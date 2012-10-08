from __future__ import with_statement
import subprocess
import sys
import time
import math

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    def debug_here(): pass

# Retry decorator with exponential backoff
def retry(tries, delay=3, backoff=2):
  """Retries a function or method until it returns True.

  delay sets the initial delay, and backoff sets how much the delay should
  lengthen after each failure. backoff must be greater than 1, or else it
  isn't really a backoff. tries must be at least 0, and delay greater than
  0."""

  if backoff <= 1:
    raise ValueError("backoff must be greater than 1")

  tries = math.floor(tries)
  if tries < 0:
    raise ValueError("tries must be 0 or greater")

  if delay <= 0:
    raise ValueError("delay must be greater than 0")

  def deco_retry(f):
    def f_retry(*args, **kwargs):
      mtries, mdelay = tries, delay # make mutable

      rv = f(*args, **kwargs) # first attempt
      while mtries > 0:
        if rv == True: # Done on success
          return True

        mtries -= 1      # consume an attempt
        time.sleep(mdelay) # wait...
        mdelay *= backoff  # make future wait longer

        rv = f(*args, **kwargs) # Try again

      return False # Ran out of tries :-(

    return f_retry # true decorator -> decorated function
  return deco_retry  # @retry(arg[, ...]) -> true decorator


def analyze_locally(groups=None, batches=8):
    p = Pool()
    p.select(groups)
    p._instantiate(batches=batches)
    p.analyze()


class Pool(object):
    def __init__(self, prefix=None, emergent_exe=None, analyze=True, debug=False):
        self.instantiated_models = []
        self.instantiated_models_dict = {}
        self.selected_models = set()
        self.emergent_exe = emergent_exe
        self.prefix = prefix
        self.debug = debug
        self.queue = []

    def queue_jobs(self):
        """Put jobs in the queue to be processed"""
        assert len(self.instantiated_models) != 0, "Insantiate models first by calling _instantiate()"

        for model in self.instantiated_models:
            split_jobs = model.split_batches()
            # Put jobs in queue
            for job in split_jobs:
                self.queue.append(job)

    def _instantiate(self, **kwargs):
        """Instantiate selected models (select via select())."""
        assert len(self.selected_models) != 0, "No models selected."
        for model in self.selected_models:
            self.instantiated_models.append(model(prefix=self.prefix, **kwargs))
            self.instantiated_models_dict[model.__name__] = self.instantiated_models[-1]

    def prepare(self, queue=True, **kwargs):
        self._instantiate(**kwargs)
        if queue:
            self.queue_jobs()

    def run(self):
        for model in self.queue:
            print model
            call_emergent(dict_to_list(model), silent=not(self.debug), emergent_exe=self.emergent_exe, prefix=self.prefix)

    def analyze(self):
        for model in self.instantiated_models:
            model.load_logs()
            model.preprocess_data()
            model.analyze()

    def select(self, groups=None, exclude=()):
        """Check if models are registered and instantiated. If not,
        register and instantiate them."""
        if not groups:
            groups = set(registered_models.groups.keys())
        else:
            groups = set(groups)
            # Check that all groups exist
            for group in groups:
                assert registered_models.groups.has_key(group), "Group with name " +group+ " not found.\n Available groups: "+', '.join(registered_models.groups.keys())
        # Exclude models
        selected_groups = groups.difference(set(exclude))
        # Find models that meet the criteria
        for model, groups in registered_models.registered_models.iteritems():
            if selected_groups.issuperset(groups) or groups.issuperset(selected_groups):
                self.selected_models.add(model)

class PoolMPI(Pool):
    def __init__(self, **kwargs):
        self.processes = []
        super(PoolMPI, self).__init__(**kwargs)

    def start_jobs(self, run=True, analyze=True, groups=None, exclude=(), **kwargs):
        from mpi4py import MPI

        # Put all jobs in the queue
        self.select(groups=groups, exclude=exclude)
        self.prepare(**kwargs)

        self.rank = MPI.COMM_WORLD.Get_rank()
        if self.rank == 0:
            self.mpi_controller(run=run, analyze=analyze, **kwargs)
        else:
            self.mpi_worker()

    # MPI function for usage on cluster
    def mpi_controller(self, run=True, analyze=True, **kwargs):

        import progressbar
        self.pbar = progressbar.ProgressBar().start()

        if run and analyze:
            raise ValueError('Either run or analyze can be true. Call this function twice.')

        if run:
            self.mpi_controller_run()
        if analyze:
            self.mpi_controller_analyze()

    def mpi_controller_run(self):
        from mpi4py import MPI

        process_list = range(1, MPI.COMM_WORLD.Get_size())
        proc_name = MPI.Get_processor_name()
        status = MPI.Status()
        if self.debug:
            print "Controller %i on %s: ready!" % (self.rank, proc_name)

        counter=0

        if self.debug:
            print self.queue
        workers_done = []
        queue = iter(self.queue)

        self.pbar.maxval = len(self.queue)
        # Feed all queued jobs to the childs
        while(True):
            # Create iterator
            status = MPI.Status()
            recv = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if self.debug:
                print "Controller: received tag %i from %s" % (status.tag, status.source)
            if status.tag == 10:
                try:
                    task = queue.next()
                    self.pbar.update(counter)
                    counter+=1
                    # Send job to worker
                    if self.debug:
                        print "Controller: Sending task"
                    MPI.COMM_WORLD.send(task, dest=status.source, tag=10)

                except StopIteration:
                    # Task queue is empty
                    if self.debug:
                        print "Controller: Task queue is empty"
                        MPI.COMM_WORLD.send([], dest=status.source, tag=2)
                        workers_done.append(status.source)
                        if self.debug:
                            print workers_done
                        if set(workers_done) == set(process_list):
                            break
                        else:
                            continue
                    else:
                        if self.debug:
                            print "Controller: Sending kill signal"
                        MPI.COMM_WORLD.send([], dest=status.source, tag=2)


            elif status.tag == 2: # Exit
                process_list.remove(status.source)
                if self.debug:
                    print 'Process %i exited' % status.source
                    print 'Processes left: ' + str(process_list)
            else:
                print 'Unkown tag %i with msg %s' % (status.tag, str(recv))

            if len(process_list) == 0:
                self.pbar.finish()
                if self.debug:
                    print "No processes left"
                return True

        return False

    def mpi_controller_analyze(self):
        from mpi4py import MPI

        process_list = range(1, MPI.COMM_WORLD.Get_size())
        proc_name = MPI.Get_processor_name()
        status = MPI.Status()

        counter = 0

        if self.debug:
            print "Controller %i on %s: Analyzing jobs!" % (self.rank, proc_name)

        iter_models = self.instantiated_models_dict.iterkeys()
        self.pbar.maxval = len(self.instantiated_models_dict)

        while(True):
            status = MPI.Status()
            recv = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

            if status.tag == 10:
                try:
                    # Get model
                    task = iter_models.next()
                    self.pbar.update(counter)
                    counter+=1
                except StopIteration:
                    # Empty, send kill signal (coded as tag 2)
                    MPI.COMM_WORLD.send([], dest=status.source, tag=2)

                MPI.COMM_WORLD.send(task, dest=status.source, tag=11)

            elif status.tag == 2: # Exit
                process_list.remove(status.source)
                if self.debug:
                    print 'Process %i exited' % status.source
                    print 'Processes left: ' + str(process_list)
            else:
                print 'Unkown tag %i with msg %s' % (status.tag, str(recv))

            if len(process_list) == 0:
                if self.debug:
                    print "No processes left"
                self.pbar.finish()
                return True

        return False


    def mpi_worker(self):
        try:
            import matplotlib
        except ImportError:
            print "Failed to import matplotlib"

        from mpi4py import MPI

        proc_name = MPI.Get_processor_name()
        status = MPI.Status()
        if self.debug:
            print "Worker %i on %s: ready!" % (self.rank, proc_name)
        # Send ready
        MPI.COMM_WORLD.send([{'rank':self.rank, 'name':proc_name}], dest=0, tag=10)

        # Start main data loop
        while True:
            # Get some data
            if self.debug:
                print "Worker %i on %s: waiting for data" % (self.rank, proc_name)
            recv = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if self.debug:
                print "Worker %i on %s: received data, tag: %i" % (self.rank, proc_name, status.tag)

            if status.tag == 2:
                if self.debug:
                    print "Worker %i on %s: recieved kill signal" % (self.rank, proc_name)
                MPI.COMM_WORLD.send([], dest=0, tag=2)
                return

            if status.tag == 10:
                # Run emergent
                if self.debug:
                    print "Worker %i on %s: Calling emergent: %s" % (self.rank, proc_name, recv)
                    recv['debug'] = True
                call_emergent(dict_to_list(recv), silent=not(self.debug), emergent_exe=self.emergent_exe, prefix=self.prefix)

            elif status.tag == 11:
                # Analyze model
                if self.debug:
                    print "Worker %i on %s: Analyzing model %s" % (self.rank, proc_name, recv)
                try:
                    model = self.instantiated_models_dict[recv]
                    model.load_logs()
                    model.preprocess_data()
                    model.analyze()
                except Exception, err:
                    # Only log the error, but keep on processing jobs
                    sys.stderr.write("Worker %i on %s: ERROR: %s" % (self.rank, proc_name, str(err)))

            if self.debug:
                print("Worker %i on %s: finished one job" % (self.rank, proc_name))
            MPI.COMM_WORLD.send([], dest=0, tag=10)

        MPI.COMM_WORLD.send([], dest=0, tag=2)

class RegisteredModels(object):
    def __init__(self):
        self.registered_models = {}
        self.groups = {}

    def register_group(self, groups):
        def reg(model):
            self.registered_models[model] = set(groups)

            for group in groups:
                if not self.groups.has_key(group):
                    self.groups[group] = set()
                self.groups[group].add(model)
            return model
        return reg


# Convenience aliases
registered_models = RegisteredModels()
#register = registered_models.register
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

@retry(2)
def call_emergent(flags, prefix=None, silent=False, emergent_exe=None):
    """Call emergent with the provided flags(list).
    A prefix can be provided which will be inserted before emergent."""
    import os

    #if mpi:
        #emergent_call = ['/gpfs/home/wiecki/BG_inhib/pyemergent/call_emergent.sh','-nogui','-ni','-p'] + flags
        # Using the older os.system() here because subprocess leads to
        # python segfaults when running on multiple nodes. No idea why.
        #subprocess.call(emergent_call)
        #print "Calling %s" % ' '.join(emergent_call)
        #ret_val = os.system(' '.join(emergent_call))
        #print "os.system return value: %i" % ret_val
        #from mpi4py import MPI
        #print "Launching job"
        #comm = MPI.COMM_SELF.Spawn('emergent',
        #                           args=['-nogui','-ni','-p'] + flags)
        #print "Finished? Disconnecting."
        #comm.Disconnect()

    if prefix is None:
        prefix = []
    elif type(prefix) is not list:
        prefix = [prefix]

    # Is there an alternate emergent executable defined? (e.g. when running on a cluster)
    if emergent_exe is None:
        emergent_call = prefix + ['/usr/bin/emergent','-nogui','-ni','-p'] + flags
    else:
        emergent_call = [emergent_exe] + ['-nogui','-ni','-p'] + flags

    if silent:
        retcode = subprocess.call(emergent_call,
                                  stdout=open(os.devnull,'w'),
                                  stderr=open(os.devnull,'w'))
    else:
        print(" ".join(emergent_call))
        retcode = subprocess.call(emergent_call)

    if retcode != 0:
        print retcode
        print prefix

    return (retcode == 0)
