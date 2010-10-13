from __future__ import with_statement
import os

try:
    import matplotlib.pyplot as plt
except:
    print "Could not load pyplot"

import numpy as np
import sys

#import hhm
from math import ceil
from copy import copy

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    def debug_here(): pass

import pools

def sem(x, axis=0):
    x = np.asarray(x)
    n = x.shape[axis]
    s = np.std(x,axis=axis)/np.sqrt(n-1)
    return s

class Base(object):
    def __init__(self, proj_name=None, batches=8, prefix=None, log_dir=None, log_dir_abs=None, debug=False, plot=True):
	if proj_name is None:
	    #proj_name = 'BG_response_inhib_preSMA_inhib_fixation'
	    #proj_name = 'BG_IFG_striatum_split_salience4'
            proj_name = 'BG_inhib8'
	self.proj = proj_name
	
	if prefix is None:
	    prefix = '/home/wiecki/working/projects/bg_inhib/'
	self.prefix = prefix
	self.batches = batches
	self.data = {}
	self.data_type = {}
        self.plot = plot
	self.flags = []
        self.lw = 2.
	#self.log_dir_emergent = os.path.join('logs', self.__class__.__name__)
        if log_dir is None:
            self.log_dir = os.path.join(self.prefix, 'logs', self.__class__.__name__)
        elif log_dir_abs is not None:
            self.log_dir = os.path.join(log_dir_abs, self.__class__.__name__)
        else:
            self.log_dir = os.path.join(self.prefix, log_dir, self.__class__.__name__)
	self.plot_prefix_png = self.prefix + 'plots/png/' + self.__class__.__name__ + '_'
	self.plot_prefix_eps = self.prefix + 'plots/eps/' + self.__class__.__name__ + '_'
	self.plot_prefix_pdf = self.prefix + 'plots/pdf/' + self.__class__.__name__ + '_'
	self.colors = ('r','b','g','y','c')

	self.ddms = {}
	self.ddms_results = {}
	#self.hhm = hhm.hhm()
	
	self.flag = {'proj': self.prefix+self.proj + '.proj',
		     'log_dir': self.log_dir,
		     'batches': 1,
                     'debug': debug,
                     'SZ_mode': 'false',
                     'rnd_seed': 'NEW_SEED',
                     'LC_mode': 'HPLT'}

	# Check if logdir directory exists, if not, create it
	if not os.path.isdir(self.log_dir):
	    os.mkdir(self.log_dir)

    def _split_batches(self):
        """Called by queue_jobs.
        Gives each batch run its own emergent job so that each batch can run in
        an individual process for better multiprocessing."""
	new_flag = {}
	flags = []
	for flag in self.flags:
	    for batch in range(self.batches):
		new_flag = copy(flag)
		new_flag['tag'] = flag['tag'] + '_b' + str(batch)
		flags.append(new_flag)

	return flags

    def queue_jobs(self, silent=True):
	# Split the flags to include individual runs for every single batch
	# (instead of one run with multiple batches)
	split_jobs = self._split_batches()
	# Put jobs in queue
        pools.pools.queue_jobs(split_jobs)

    def _preprocess_logs(self, log_type, converters=None):
	"""Load in log files. Populates data."""
	# Read in log files
	data = {}
	for tag in self.tags:
	    data_batches = []
	    for batch in range(self.batches):
		fname = os.path.join(self.log_dir, self.proj + '_' + tag + '_b' + str(batch) + '.' + log_type + '.dat')
		print fname
		data_ind = load_log(fname)
		# Change batch number
		data_ind['batch'] = batch
		# Append row with batch number in it
		data_batches.append(data_ind)
	    data[tag] = np.hstack([data_batch for data_batch in data_batches])
		
	return data
    
    def load_logs(self, log_type=None):
	"""Load logs of a certain log type (e.g. 'trl' or'cyc').
	log_type<string> defaults to 'trl'."""
	if log_type is None:
	    log_type = 'trl'

	self.data = self._preprocess_logs(log_type)

	return self.data

    def load_logs_type(self, log_types):
	"""Load logs of a multiple log types (e.g. ['trl','cyc']).
	Arguments:
	log_types<list>
	"""
	for log_type in log_types:
	    self.data[log_type] = self._preprocess_logs(log_type)

	return self.data

    def preprocess_data(self):
	"""Base method. May be overloaded with preprocessing."""
	pass
    
    def analyze(self):
	"""Base method. To be overloaded with your analyzes."""
	pass
    
    def new_fig(self, **kwargs):
        from pylab import figure, subplot
	figure(**kwargs)
	ax = subplot(111)
	#ax.spines['left'].set_position('center')
	ax.spines['right'].set_color('none')
	#ax.spines['bottom'].set_position('center')
	ax.spines['top'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

    def save_plot(self, name):
        from pylab import savefig
	savefig(self.plot_prefix_png + name + ".png")
	savefig(self.plot_prefix_eps + name + ".eps")
	savefig(self.plot_prefix_pdf + name + ".pdf")

    def fit_hddm(self, depends_on, plot=False, **kwargs):
        import hddm
        model = hddm.Multi(self.hddm_data, depends_on=depends_on, is_subj_model=True, no_bias=False, **kwargs)
        model.mcmc()

        if plot:
            raise NotImplementedError, "TODO"

        return model

    def fit_hlba(self, depends_on, plot=False, **kwargs):
        import hddm
        model = hddm.Multi(self.hddm_data, model_type='LBA', depends_on=depends_on, is_subj_model=True, no_bias=False, **kwargs)
        model.mcmc()

        if plot:
            raise NotImplementedError, "TODO"

        return model
    

class BaseCycle(Base):
    def __init__(self, **kwargs):
	super(BaseCycle, self).__init__(**kwargs)
	self.flag['log_cycles'] = True


    def load_logs(self):
	self.load_logs_type(['trl', 'cyc'])
	
    def extract_cycles(self, tag, trl_cond, cyc_col_name, center=None, cycle=None, wind=(50,50)):
	"""Extracts cycles of individual trials.
	Trials can be specified by trl_cond. The data will be centered around
	the col 'center' if it is supplied or around the cycle number if cycle is supplied."""

	batches = self.data['trl'][tag][trl_cond]['batch']
	epochs = self.data['trl'][tag][trl_cond]['epoch']
	trials = self.data['trl'][tag][trl_cond]['trial']
	
	if cycle is not None:
	    centers = np.ones(trials.shape) * cycle
	elif center is not None:
	    centers = self.data['trl'][tag][trl_cond][center]
	else:
	    raise ValueError, "You have to supply either cycle or center."

	center_winds = []

	for batch, epoch, trial, center in zip(batches, epochs, trials, centers):
	    # Select corresponding trial (uniquely identified by it's batch_num, epoch and trial
	    data_cyc_idx = (self.data['cyc'][tag]['batch'] == batch) & \
			   (self.data['cyc'][tag]['epoch'] == epoch) & \
			   (self.data['cyc'][tag]['trial'] == trial)
	    data_cyc_ind = self.data['cyc'][tag][data_cyc_idx]
	    # Find the cycle index in which the response was made
	    data_cyc_cent_idx = np.where(data_cyc_ind['cycle'] == center)[0]

	    ##############################################
	    # Extract wind before and after this response
	    ##############################################
	    center_cycle = data_cyc_ind[data_cyc_cent_idx]['cycle'][0]
	    max_cycles = np.max(data_cyc_ind['cycle'])

	    # Detect if there is enough space to cut out the window around the response
	    pre_buf = 0
	    post_buf = wind[1]
	    if center_cycle < wind[0]:
		pre_buf = wind[0] - center_cycle
	    if wind[1] > max_cycles - center_cycle:
		post_buf = max_cycles - center_cycle

	    # How large the max window is allowed to be
	    pre_wind = wind[0] - pre_buf

	    # Actually copy over the corresponding window
	    center_wind = np.zeros((np.sum(wind)+1))
	    center_wind[pre_buf:wind[0]] = data_cyc_ind[data_cyc_cent_idx[0]-pre_wind:data_cyc_cent_idx[0]][cyc_col_name]
	    center_wind[wind[0]:wind[0]+post_buf+1] = data_cyc_ind[data_cyc_cent_idx[0]:data_cyc_cent_idx[0]+post_buf+1][cyc_col_name]

	    center_winds.append(center_wind)

	return np.array(center_winds)

    def plot_filled(self, x, data, avg=True, **kwargs):
	import matplotlib.pyplot as plt

        if avg: # Plot average line
            y = np.mean(data, axis=0)
            if len(data.shape) == 0:
                raise ValueError('Data array is empty.')
            if data.shape[1] != 0:
                sem_ = sem(data, axis=0)
                plt.fill_between(x, y-sem_, y+sem_, alpha=.5, **kwargs)
            plt.plot(x, y, **kwargs)
            
        else: # Plot all lines
            for y in data:
                plt.plot(x, y, **kwargs)
            

def load_log(fname):
    dtype = convert_header_np(fname)
    if np.__version__ == '1.5.0':
        data = np.genfromtxt(fname, dtype=dtype, skip_header=True)
    else:
        data = np.genfromtxt(fname, dtype=dtype, skiprows=1)

    return data

def convert_header_np(fname):
    """Reads in the column names from the log files provided with fnames.
    Returns a dict."""
    dt = []
    mask = []
    idx = 0

    with open(fname) as f:
        line = f.readline()

    cols = line.split()
    
    for i, col in enumerate(cols):
        prefix = col[0]
        name = col[1:]
        
        if prefix == '|':
            # Int
            dt.append((name, 'float'))
            mask.append(True)
        elif prefix == '$':
            # String
            dt.append((name, 'S16'))
            mask.append(False)
        elif prefix == '%':
            # Float
            dt.append((name, 'float'))
            mask.append(True)
        elif prefix == '#':
            # Double
            dt.append((name, 'float'))
            mask.append(True)
        elif prefix == '_':
            dt.append((name, 'S3'))
            mask.append(False)
        elif prefix == '&':
            raise NotImplementedError, 'Log files with matrices not supported'
        else:
            raise TypeError, 'Unknown prefix %s' % prefix

    return dt


def group(data, group_names):
    """Group data according to a list of group_names

    >>> test = gen_testdata()
    >>> means, sems = group(test, ['a'])
    >>> means['a']
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    >>> sems['a']
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    >>> means['b']
    array([ 12.5,  12.5,  12.5,  12.5,  12.5,  12.5,  12.5,  12.5,  12.5,  12.5])
    >>> sems['b']
    array([ 0.42257713,  0.42257713,  0.42257713,  0.42257713,  0.42257713,
            0.42257713,  0.42257713,  0.42257713,  0.42257713,  0.42257713])
    >>> means2, sems2 = group(test, ['a', 'b'])
    >>> means2['b']
    array([ 11.,  12.,  13.,  14.,  11.,  12.,  13.,  14.,  11.,  12.,  13.,
            14.,  11.,  12.,  13.,  14.,  11.,  12.,  13.,  14.,  11.,  12.,
            13.,  14.,  11.,  12.,  13.,  14.,  11.,  12.,  13.,  14.,  11.,
            12.,  13.,  14.,  11.,  12.,  13.,  14.])
    """
    
    # Define Counter object to keep track of where we are
    class Counter(object):
        def __init__(self):
            self.i = 0
        def add(self):
            self.i += 1

    row_idx = Counter()

    rows = 1

    # Calculate size
    for name in group_names:
        rows *= np.unique(data[name]).shape[0]

    # Allocate arrays
    data_mean = np.empty((rows), dtype=data.dtype)
    data_sem = np.empty((rows), dtype=data.dtype)

    # Call recursive grouping function
    group_rec(data, group_names, data_mean, data_sem, row_idx)
    
    # Check if the arrays have been filled
    if row_idx.i < rows:
        # Delete empty rows and issue a warning
        print("WARNING: Not all groups generated, data incomplete?")
        data_mean = np.delete(data_mean, np.s_[row_idx.i:rows])
        data_sem = np.delete(data_sem, np.s_[row_idx.i:rows])
        
    # Fix data_sem
    for name in group_names:
        data_sem[name] = data_mean[name]
    return data_mean, data_sem


def group_rec(data, group_names, data_mean, data_sem, row_idx):
    if len(group_names) != 0:
        col = data[group_names[0]]
	items = np.unique(data[group_names[0]])
	# Extract items from first group col
	for item in items:
	    # Select those rows with the item
	    rows_select = col == item
            
            data_slice = data[rows_select,:]

	    # Recursive call
	    group_rec(data_slice, group_names[1:], data_mean, data_sem, row_idx)

        return

    else:
	# Calculate mean and sem and return
        # We have to iterate over the items individually because mean does not
        # work on recarrays
        for name,col_type in data.dtype.descr:
            if col_type.find('S') != -1:
                # Col dtype is a string. We can not calculate meaningful stats
                # for that if there are different ones
		uniq = np.unique(data[name])
		if uniq.shape[0] > 1:
		    data_mean[name][row_idx.i] = 'N/A'
		    data_sem[name][row_idx.i] = 'N/A'
		else:
		    data_mean[name][row_idx.i] = uniq[0]
		    data_sem[name][row_idx.i] = uniq[0]
            else:
                # Col is a number (hopefully)
                col = data[name]
                data_mean[name][row_idx.i] = np.mean(col)
		# SEM of just one number is not possible
		if col.shape[0] != 1:
		    data_sem[name][row_idx.i] = sem(col)
		else:
		    data_sem[name][row_idx.i] = 0
        row_idx.add()
        return
    
def group_batch(data, group_names):
    """Convience function for group which groups over batches first and
    then over group_names so that SEM values are group wise"""
    data_mean, data_sem = group(data, ['batch'] + group_names)
    data_mean_gp, data_sem_gp = group(data_mean, group_names)

    return data_mean_gp, data_sem_gp

def gen_testdata():
    new_cols = np.array([0,0,0,0])
    for i in range(0,10):
	for j in range(11,15):
	    for k in range(16,18):
		new_row = np.hstack((i, j, k, np.random.randn(1)))
		new_cols = np.vstack((new_cols, new_row))
    new_cols = new_cols[1:,:]
    dt = np.dtype([('a', 'float'), ('b', 'float'), ('c','float'), ('d', 'float')])
    new_cols.dtype = dt
    return new_cols

def write_job(nodes, prefix, emergent, set_python_exec=None, log_dir=None, ppn=8):
    """Write .job file to be ran with qsub."""
    with open("emergent.job", "w") as f:
        f.writelines(['#!/bin/bash\n',
                      '#PBS -N emergent\n',
                      '#PBS -r n\n',
                      '#PBS -l nodes='+str(nodes)+':ppn=%i\n'%ppn,
                      '#PBS -l walltime=04:00:00\n',
                      'date\n'])
        
        #if set_python_exec is not None:
        #    f.write('source ' +set_python_exec + '\n')

        # Write jobs
        if log_dir is None:
            f.write('mpirun -machinefile $PBS_NODEFILE -np '+ str(nodes*ppn) + ' ' + set_python_exec + ' ' + os.path.os.getcwd() + '/emergent.py -m -p '+prefix+' -e '+emergent+'\n')
            f.write('mpirun -machinefile $PBS_NODEFILE -np '+ str(nodes*ppn) + ' ' + set_python_exec + ' ' + os.path.os.getcwd() + '/emergent.py -a -p '+prefix+' -e '+emergent+'\n')
        else:
            f.write('mpirun -machinefile $PBS_NODEFILE -np '+ str(nodes*ppn) + ' ' + set_python_exec + ' ' + os.path.os.getcwd() + '/emergent.py -m -p '+prefix+' -l ' + log_dir +' -e '+emergent+'\n')
            f.write('mpirun -machinefile $PBS_NODEFILE -np '+ str(nodes*ppn) + ' ' + set_python_exec + ' ' + os.path.os.getcwd() + '/emergent.py -a -p '+prefix+' -l ' + log_dir +' -e '+emergent+'\n')

        f.write('date\n')
        # Analyze data
        #f.write('python emergent.py -a -p '+prefix+'\n')

def usage():
    print("""emergent.py

    Run response inhibition model and analyze.

    Arguments:
    --individual (-i): Don't parallelize, just run serial (taks a _long_ time).

    --write (-w): write emergent.job script to be run by qsub.
    --nodes (-n) <int>: how many subprocesses to start (should be number of CPUs you want to run on).
    --jobs (-j) <int>: how many jobs should each subprocess work.

    --mpi (-m): parallelize run with MPI.
    --fraction (-f) <int>: which fraction of the whole jobs to run.
    --analyze (-a): Analyze the log files (after simulations have been run).

    """)
    
    
def main():
    import getopt

    # Set defaults
    master = False
    slave = False
    analyze = False
    prefix = None
    emergent = None
    set_python_exec = None
    log_dir = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'wmn:f:p:e:l:s:ah', ['write', 'mpi', 'nodes', 'prefix', 'emergent', 'log_dir', 'set_python_exec', 'analyze', 'help'])
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)

    for o, a in opts:
        if o in ('-w', '--write'):
            master=True
        elif o in ('-m', '--mpi'):
            slave=True
        elif o in ('-n', '--nodes'):
            nodes = int(a)
        elif o in ('-a', '--analyze'):
            analyze = True
        elif o in ('-p', '--prefix'):
            prefix = a
            pools.registered_models.prefix = prefix
        elif o in ('-e', '--emergent'):
            emergent = a
            pools.pools.emergent_exe = emergent
        elif o in ('-l', '--logdir'):
            log_dir = a
        elif o in ('-s', '--set_python_exec'):
            set_python_exec = a
        elif o in ('-h', '--help'):
            usage()

    if master or slave or analyze:

        # Queue models to find out how many jobs there are
        import antisaccade
        import stopsignal

        pools.registered_models.prepare_queue(batches=4)
   
    if master:
        if prefix is None:
            print "Please provide the prefix directory"
            sys.exit(2)
        write_job(nodes, prefix, emergent, set_python_exec=set_python_exec, log_dir=log_dir)

    elif slave or analyze:
        pools.registered_models.run_jobs_mpi(run=slave, analyze=analyze) #log_dir_abs=log_dir)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    
    main()
