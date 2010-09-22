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
    pass

try:
    import multiprocessing
except:
    pass

from Queue import Empty, Full

class Pools(object):
    """This class contains the following containers and operates on them:
    queue: jobs to work on
    queue_output: finished jobs
    pbar: the progress bar
    processes: the workers"""
    
    def __init__(self):
        self.queue = multiprocessing.JoinableQueue()
        self.queue_output = multiprocessing.Queue()
        if pbar:
            self.pbar = progressbar.ProgressBar()
        self.processes = []
        self.groups = {}
        self.registered_models = set()
        self.emergent_exe = None


    def queue_jobs(self, items):
        """Put jobs in the queue to be processed"""
	assert len(items) != 0, "Job queue is empty!"
        
        for item in items:
            self.queue_job(item)
            
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
    
    def start_and_join_workers(self, ssh=True, hosts=None, silent=True):
        # Convenience function
	if ssh:
	    self.start_workers_ssh(hosts=hosts, silent=silent)
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

    def start_workers_ssh(self, hosts=None, silent=True):
        if hosts is None:
	    hosts = {'cycle':2, 'bike':2, 'ski':1, 'drd2':4, 'darpp32': 4}
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
        if host != 'local':
            command = ['ssh', host]
        else:
            command = None

        try:
            while(True):
                flag = self.queue.get(timeout=10)
                call_emergent(dict_to_list(flag), prefix=command, silent=silent)
                # Done
                self.queue.task_done()
                self.queue_output.put(flag)
                if pbar:
                    self.pbar.update(self.queue_output.qsize())
        except Empty:
            if not silent:
                print "Empty"
            return

    # MPI function for usage on cluster
    def mpi_controller(self, run=True, analyze=True):
        from mpi4py import MPI
        process_list = range(1, MPI.COMM_WORLD.Get_size())

        print "Controller: started"
        if run:
            # Feed all queued jobs to the childs
            while(True):
                status = MPI.Status()
                recv = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                print "Controller: received tag %i" % status.tag
                if status.tag == 10:
                    try:
                        task = self.queue.get(timeout=10)
                    except Empty:
                        break
                    # Send job to worker
                    print "Controller: Sending task"
                    MPI.COMM_WORLD.send(task, dest=status.source, tag=10)

                elif status.tag == 2: # Exit
                    process_list.remove(status.source)
                    print 'Process %i exited' % status.source
                else:
                    print 'Unkown tag %i with msg %s' % (status.tag, str(data))

                if len(process_list) == 0:
                    print "No processes left"
                    break

        # All jobs finished, analyze.
        if analyze:
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
                call_emergent(dict_to_list(recv), mpi=True)

            elif status.tag == 11:
                # Analyze model
                print "Worker %i on %s: Analyzing model %s" % (rank, proc_name, recv)
                model = registered_models.instantiated_models_dict[recv]
                model.load_logs()
                model.preprocess_data()
                model.analyze()

            print("Worker %i on %s: finished one job" % (rank, proc_name))
            MPI.COMM_WORLD.send([], dest=0, tag=10)
    
        MPI.COMM_WORLD.send([], dest=0, tag=2)

pools = Pools()

class RegisteredModels(object):
    def __init__(self):
        self.registered_models = []
        self.instantiated_models = []
        self.instantiated_models_dict = {}
        self.prefix = '/home/wiecki/working/projects/bg3/'
        #self.log_dir = None

    def _instantiate(self, **kwargs):
        for model in self.registered_models:
            self.instantiated_models.append(model(prefix=self.prefix, **kwargs))
            self.instantiated_models_dict[model.__name__] = self.instantiated_models[-1]

    def _queue(self):
        assert len(self.instantiated_models) != 0, "Insantiate models first by calling _instantiate()"
        
        for model in self.instantiated_models:
	    model.queue_jobs()
    
    def prepare_queue(self, groups=None, **kwargs):
        """Instantiates and queues models. This is used when we need
        to get the queue length."""
        self._check(groups=groups, **kwargs)
        self._queue()

    def _check(self, groups=None, **kwargs):
        """Check if models are registered and instantiated. If not,
        register and instantiate them."""

        if len(self.registered_models) == 0:
            # If models were not set externally, set them to the registered ones
            if groups is None:
                self.registered_models = pools.registered_models
            else:
                group_models = []
                # Gather all models from all groups and create intersection
                for group in groups:
                    assert pools.groups.has_key(group), "Group with name "+ group+ " not found"
                    group_models.append(pools.groups[group])
                self.registered_models = set.intersection(*group_models)
                    

	if len(self.instantiated_models) == 0:
	    # Models have not been instantiated yet
	    self._instantiate(**kwargs)

    def run_and_analyze(self, groups=None, silent=True, run=True, analyze=True, hosts=None, **kwargs):
        if run:
            self.run(groups=groups, silent=silent, hosts=hosts, **kwargs)
        if analyze:
            self.analyze(groups=groups, **kwargs)

        return self.instantiated_models
        
    def run(self, groups=None, silent=True, hosts=None, **kwargs):
        self.prepare_queue(groups=groups, **kwargs)
	pools.start_and_join_workers(silent=silent, hosts=hosts)
        
        return self.instantiated_models

    def analyze(self, groups=None, **kwargs):
        self._check(groups=groups, **kwargs)

        for model in self.instantiated_models:
            model.load_logs()
	    model.preprocess_data()
            model.analyze()

        return self.instantiated_models

    def run_jobs_mpi(self, run=True, analyze=False):
        from mpi4py import MPI
        #self._check(**kwargs)
        #self._queue()

        if MPI.COMM_WORLD.Get_rank() == 0:
            pools.mpi_controller(run=run, analyze=analyze)
        else:
            pools.mpi_worker()
    
    def cluster_run_model(self, idx):
        """Run MPI job for only one model (with all batches and all
        conditions.  
        Arguments: 
        idx: which model to run.
        """

        from mpi4py import MPI
        self._instantiate()

        # Put jobs of model into queue
        assert (idx+1) < len(self.instantiated_models), "Model index not found"
        model = self.instantiated_models[idx]
        model.queue_jobs()
        
        self.run_jobs_mpi(run=run, analyze=analyze)

# Convenience aliases
registered_models = RegisteredModels()
run = registered_models.run_and_analyze
register = pools.register
register_group = pools.register_group

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
    # initialize (sometimes when a run did not finish old jobs hang around in
    # the queue)
    pools.terminate_workers()
    
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

def call_emergent(flags, prefix=None, silent=False, errors=False, mpi=False):
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
        if pools.emergent_exe is None:
            emergent_call = prefix + ['nice', '-n', '19', 'emergent','-nogui','-ni','-p'] + flags
        else:
            emergent_call = prefix + [pools.emergent_exe] + ['-nogui','-ni','-p'] + flags
        
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
            if mpi:
                subprocess.call(emergent_call)
            else:
                subprocess.call(emergent_call)
