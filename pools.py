from __future__ import with_statement
import subprocess
import os
import sys
import time
import numpy as np

from copy import copy
from glob import glob
from math import ceil

try:
    import progressbar
    pbar = True
except:
    pbar = False
    pass

try:
    import matplotlib.pyplot as plt
except:
    print "Could not load pyplot"

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    def debug_here(): pass


import time

class Retry(object):
    default_exceptions = (Exception)
    def __init__(self, tries, exceptions=None, delay=0):
        """
        Decorator for retrying function if exception occurs
        
        tries -- num tries
        exceptions -- exceptions to catch
        delay -- wait between retries
        """
        self.tries = tries
        if exceptions is None:
            exceptions = Retry.default_exceptions
        self.exceptions =  exceptions
        self.delay = delay

    def __call__(self, f):
        def fn(*args, **kwargs):
            exception = None
            for _ in range(self.tries):
                try:
                    return f(*args, **kwargs)
                except self.exceptions, e:
                    print "Retry, exception: "+str(e)
                    time.sleep(self.delay)
                    exception = e
            #if no success after tries, raise last exception
            raise exception
        return fn


class Pool(object):
    def __init__(self):
        self.instantiated_models = []
        self.instantiated_models_dict = {}
        self.selected_models = set()
        self.emergent_exe = None

    def queue_jobs(self):
        """Put jobs in the queue to be processed"""
        assert len(self.instantiated_models) != 0, "Insantiate models first by calling _instantiate()"
        for model in self.instantiated_models:
            split_jobs = model.split_batches()
            # Put jobs in queue
            for job in split_jobs:
                self.queue_job(job)

    def _instantiate(self, **kwargs):
        """Instantiate selected models (select via select())."""
        assert len(self.selected_models) != 0, "No models selected."
        for model in self.selected_models:
            self.instantiated_models.append(model(**kwargs))
            self.instantiated_models_dict[model.__name__] = self.instantiated_models[-1]

    def prepare(self, queue=True, **kwargs):
        self._instantiate(**kwargs)
        if queue:
            self.queue_jobs()

    def analyze(self):
        for model in self.instantiated_models:
            model.load_logs()
            model.preprocess_data()
            model.analyze()
            
    def select(self, groups=None, exclude=None, all=False):
        """Check if models are registered and instantiated. If not,
        register and instantiate them."""
        if all:
            groups = registered_models.groups.keys()

        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)
            
        # Check that all groups exist
        for group in groups:
            assert registered_models.groups.has_key(group), "Group with name " +group+ " not found.\n Available groups: "+', '.join(registered_groups.groups.keys())

        selected_groups = set.intersection(set(registered_models.groups.keys()), set(groups))

        # Exclude models
        #selected_groups = set.discard(self.selected_models, exclude)

        # Add models from selected groups
        for group in selected_groups:
            for model in registered_models.groups[group]:
                self.selected_models.add(model)


class PoolMPI(Pool):
    def __init__(self):
        self.queue = []
        self.processes = []
        super(PoolMPI, self).__init__()
            
    def queue_job(self, item):
        """Put one job in the queue to be processed"""
        self.queue.append(item)

    def start_jobs(self, run=True, analyze=True, **kwargs):
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            self.mpi_controller(run=run, analyze=analyze, **kwargs)
        else:
            self.mpi_worker()
        
    # MPI function for usage on cluster
    def mpi_controller(self, run=True, analyze=True, **kwargs):
        from mpi4py import MPI

        # Put all jobs in the queue
        self.select(all=True)
        self.prepare(**kwargs)
        
        process_list = range(1, MPI.COMM_WORLD.Get_size())
        rank = MPI.COMM_WORLD.Get_rank()
        proc_name = MPI.Get_processor_name()
        status = MPI.Status()

        print "Controller %i on %s: ready!" % (rank, proc_name)

        if run and analyze:
            raise ValueError('Either run or analyze can be true. Call this function twice.')

        if run:
            workers_done = []
            queue = iter(self.queue)
            # Feed all queued jobs to the childs
            while(True):
                # Create iterator
                status = MPI.Status()
                recv = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                print "Controller: received tag %i" % status.tag
                if status.tag == 10:
                    try:
                        task = queue.next()
                        # Send job to worker
                        print "Controller: Sending task"
                        MPI.COMM_WORLD.send(task, dest=status.source, tag=10)

                    except StopIteration:
                        # Task queue is empty
                        print "Controller: Task queue is empty"
                        if analyze:
                            workers_done.append(status.source)
                            print workers_done
                            if len(workers_done) == len(process_list):
                                break
                            else:
                                continue
                        else:
                            print "Controller: Sending kill signal"
                            MPI.COMM_WORLD.send([], dest=status.source, tag=2)
                        

                elif status.tag == 2: # Exit
                    process_list.remove(status.source)
                    print 'Process %i exited' % status.source
                    print 'Processes left: ' + str(process_list)
                else:
                    print 'Unkown tag %i with msg %s' % (status.tag, str(data))

                if len(process_list) == 0:
                    print "No processes left"
                    break

        # All jobs finished, analyze.
        if analyze:
            print "Controller: Analyzing jobs"
            iter_models = registered_models.instantiated_models_dict.iterkeys()
            while(True):
                status = MPI.Status()
                recv = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

                if status.tag == 10:
                    try:
                        # Get model
                        task = iter_models.next()
                    except StopIteration:
                        # Empty, send kill signal (coded as tag 2)
                        MPI.COMM_WORLD.send([], dest=status.source, tag=2)

                    MPI.COMM_WORLD.send(task, dest=status.source, tag=11)

                elif status.tag == 2: # Exit
                    process_list.remove(status.source)
                    print 'Process %i exited' % status.source
                    print 'Processes left: ' + str(process_list)
                else:
                    print 'Unkown tag %i with msg %s' % (status.tag, str(recv))

                if len(process_list) == 0:
                    print "No processes left"
                    break

        return False
                

    def mpi_worker(self):
        try:
            import matplotlib
        except ImportError:
            print "Failed to import matplotlib"

        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        proc_name = MPI.Get_processor_name()
        status = MPI.Status()
        
        print "Worker %i on %s: ready!" % (rank, proc_name)
        # Send ready
        MPI.COMM_WORLD.send([{'rank':rank, 'name':proc_name}], dest=0, tag=10)

        # Start main data loop
        while True:
            # Get some data
            print "Worker %i on %s: waiting for data" % (rank, proc_name)
            recv = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG, status=status)
            print "Worker %i on %s: received data, tag: %i" % (rank, proc_name, status.tag)
            
            if status.tag == 2:
                print "Worker %i on %s: recieved kill signal" % (rank, proc_name)
                MPI.COMM_WORLD.send([], dest=0, tag=2)
                return

            if status.tag == 10:
                # Run emergent
                print "Worker %i on %s: Calling emergent: %s" % (rank, proc_name, recv)
                call_emergent(dict_to_list(recv), mpi=True, emergent_exe=self.emergent_exe)

            elif status.tag == 11:
                # Analyze model
                print "Worker %i on %s: Analyzing model %s" % (rank, proc_name, recv)
                try:
                    model = registered_models.instantiated_models_dict[recv]
                    model.load_logs()
                    model.preprocess_data()
                    model.analyze()
                except Exception, err:
                    # Only log the error, but keep on processing jobs
                    sys.stderr.write("Worker %i on %s: ERROR: %s" % (rank, proc_name, str(err)))
                    

            print("Worker %i on %s: finished one job" % (rank, proc_name))
            MPI.COMM_WORLD.send([], dest=0, tag=10)
    
        MPI.COMM_WORLD.send([], dest=0, tag=2)

class PoolSSH(Pool):
    """This class contains the following containers and operates on them:
    queue: jobs to work on
    queue_output: finished jobs
    pbar: the progress bar
    processes: the workers"""
    
    def __init__(self):
        import multiprocessing
        from Queue import Empty, Full
        
        self.queue = multiprocessing.JoinableQueue()
        self.queue_output = multiprocessing.Queue()
        if pbar:
            self.pbar = progressbar.ProgressBar()
        self.processes = []

        super(PoolSSH, self).__init__()

    def run(self, hosts=None, run=True, analyze=True, groups=None, all=False, **kwargs):
        self.select(groups=groups, all=all)
        if run:
            self.prepare(**kwargs)
            self.start_and_join_workers()

        if analyze:
            self._instantiate(**kwargs)
            self.analyze()
            
            
    def queue_job(self, item):
        """Put one job in the queue to be processed"""
        self.queue.put(item)
	# Update the progress bar
        if pbar:
            self.pbar.maxval = self.queue.qsize()

    def join_workers(self):
        self.queue.join()
        # After successful completion, terminate all workers to tidy up
        self.terminate_workers()
    
    def start_and_join_workers(self, hosts=None, silent=True):
        # Convenience function
        self.start_workers(hosts=hosts, silent=silent)
        self.join_workers()

    def terminate_workers(self):
        if len(self.processes) != 0:
            for process in self.processes:
                process.terminate()
            self.processes = []

        # Empty the queue
        try:
            for item in self.queue.get_nowait():
                pass
        except Empty:
            pass
        #self.pbar.finish()

    def start_workers(self, hosts=None, silent=True):
        import multiprocessing

        if hosts is None:
            hosts = {'cycle':2, 'bike':2, 'ski':1, 'drd2':3, 'darpp32': 3, 'theta':7}
            #hosts = {'cycle':2, 'bike':2, 'ski':2, 'ride':2}
            #hosts = {'cycle':2, 'bike':2, 'ski':1, 'drd2':3}
            #hosts = {'drd2':4}

        if pbar:
            self.pbar.start()
        for host, num_threads in hosts.iteritems():
            # Create worker threads for every host
            for i in range(num_threads):
                if not silent:
                    print "Launching process for " + host
                proc = multiprocessing.Process(target=self.worker, args=(host,silent))
                proc.start()
                self.processes.append(proc)

    def worker(self, host, silent):
        from Queue import Empty, Full
        if host != 'local':
            command = ['ssh', host]
        else:
            command = None

        try:
            while(True):
                try:
                    flag = self.queue.get(timeout=20)
                except Empty:
                    flag = self.queue.get(timeout=10)
                    
                call_emergent(dict_to_list(flag), prefix=command, silent=silent, emergent_exe=self.emergent_exe)
                # Done
                self.queue.task_done()
                self.queue_output.put(flag)
                if pbar:
                    self.pbar.update(self.queue_output.qsize())
        except Empty:
            if not silent:
                print "Empty"
            return


class RegisteredModels(object):
    def __init__(self):
        self.registered_models = set()
        self.groups = {}

    def register(self, model):
        """Register model so that it can be run automatically by
        calling the pools.run() function."""
        self.registered_models.add(model)
        
        return model
    
    def register_group(self, groups):
        def reg(model):
            self.register(model)
            
            for group in groups:
                if not self.groups.has_key(group):
                    self.groups[group] = set()
                self.groups[group].add(model)
            return model
        return reg


# Convenience aliases
registered_models = RegisteredModels()
register = registered_models.register
register_group = registered_models.register_group

def dict_to_list(dict):
    """Convert dictionary to list where each item is 'key=value'.
    Used for passing command line args to emergent.
    """
    return [str(key) + '=' + str(value) for key, value in dict.iteritems()]

def run_model(model_class, run=True, analyze=True, hosts=None, **kwargs):
    """Convenience function to run and analyse individual model types.

    Arguments:
    **********
    model_class: a class derived from emergent.base (e.g. emergent.saccade).
    run (default=True): if False, the data is only analyzed (assumes previous run).
    hosts <dict>: define the hosts you want to distribute the work load
                  on as the key (passwordless ssh is required). Value is the number of
                  processes you want to run on each of them at the same time (e.g. choose the
                  number of cores). Example: hosts={'node1':2, 'node2':4}.
                  If you want to run only local, without ssh, simply set hosts={'local':x}

    additional keywoard args are passed to the model class."""
    # create instance of the model
    model = model_class(**kwargs)
    if run:
	model.queue_jobs()
	pools.start_and_join_workers(hosts=hosts)
    model.load_logs()

    if analyze:
	model.load_logs()
	model.preprocess_data()
	model.analyze()

    return model

@Retry(3)
def call_emergent(flags, prefix=None, silent=False, errors=False, mpi=False, emergent_exe=None):
    """Call emergent with the provided flags(list).
    A prefix can be provided which will be inserted before emergent."""
    import os

    if mpi:
        emergent_call = ['/gpfs/home/wiecki/BG_inhib/pyemergent/call_emergent.sh','-nogui','-ni','-p'] + flags
        # Using the older os.system() here because subprocess leads to
        # python segfaults when running on multiple nodes. No idea why.
        os.system(' '.join(emergent_call))

    else:
        if prefix is None:
            prefix = []

        # Is there an alternate emergent executable defined? (e.g. when running on a cluster)
        if emergent_exe is None:
            emergent_call = prefix + ['nice', '-n', '0', 'emergent','-nogui','-ni','-p'] + flags
        else:
            emergent_call = prefix + [emergent_exe] + ['-nogui','-ni','-p'] + flags
        
        if silent:
            if errors:
                subprocess.call(emergent_call,
                                stdout=open(os.devnull,'w'))
            else:
                subprocess.call(emergent_call,
                                stdout=open(os.devnull,'w'),
                                stderr=open(os.devnull,'w'))
        else:
            print(" ".join(emergent_call))
            subprocess.call(emergent_call)
