from __future__ import with_statement
import matplotlib

#try:
#    import matplotlib
#    matplotlib.use('Agg')
    #import matplotlib.pyplot as plt
#except:
#    print "Could not load pyplot"

import numpy as np
import sys, os

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
    def __init__(self, proj_name=None, batches=8, prefix=None, log_dir=None, log_dir_abs=None, debug=None, plot=True):
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
	self.colors = ('k','r','b','y','c','g','m','w')

	self.ddms = {}
	self.ddms_results = {}
	#self.hhm = hhm.hhm()

        self.flag = {'proj': self.prefix+self.proj + '.proj',
		     'log_dir': self.log_dir,
		     'batches': 1,
                     'SZ_mode': 'false',
                     #'rnd_init': 'OLD_SEED',
                     'LC_mode': 'tonic',
                     'tonic_NE': 0.8,
                     'motivational_bias': 'NO_BIAS'}

        if debug is not None:
            self.flag['debug'] = debug
            self.debug = debug
        else:
            self.debug = False

	# Check if logdir directory exists, if not, create it
	if not os.path.isdir(self.log_dir):
            try:
                os.mkdir(self.log_dir)
            except OSError:
                pass

    def split_batches(self):
        """Gives each batch run its own emergent job so that each batch can run in
        an individual process for better multiprocessing."""
	new_flag = {}
	flags = []

	for flag in self.flags:
            np.random.seed(31337)
	    for batch in range(self.batches):
		new_flag = copy(flag)
		new_flag['tag'] = flag['tag'] + '_b' + str(batch)
                new_flag['init_seed'] = np.random.randint(1e5)
		flags.append(new_flag)

	return flags

    def _preprocess_logs(self, log_type, converters=None):
	"""Load in log files. Populates data."""
	# Read in log files
	data = {}
	for tag in self.tags:
	    data_batches = []
	    for batch in range(self.batches):
		fname = os.path.join(self.log_dir, self.proj + '_' + tag + '_b' + str(batch) + '.' + log_type + '.dat')
                if self.debug:
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
-	log_types<list>
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
	fig = figure(**kwargs)
	ax = subplot(111)
	#ax.spines['left'].set_position('center')
	ax.spines['right'].set_color('none')
	#ax.spines['bottom'].set_position('center')
	ax.spines['top'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

        return fig

    def save_plot(self, name):
        from pylab import savefig
	savefig(self.plot_prefix_png + name + ".png")
	savefig(self.plot_prefix_eps + name + ".eps")
	savefig(self.plot_prefix_pdf + name + ".pdf")

def fit_hddm((data, depends_on)):
    import hddm
    import hddm.sandbox

    # Select only antisaccade trials
    data = data[data['instruct']==1]

    model = hddm.sandbox.HDDMSwitch(data, depends_on=depends_on, is_group_model=False, init=True)

    model.create_nodes()
    model.map(runs=3)
    model.sample(7000, burn=2000)
    model.print_stats()
    print "Logp: %f" % model.mc.logp
    #hddm.utils.plot_posteriors(model)
    stats = model.stats()
    stats['logp'] = model.mc.logp
    return stats

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

	center_winds_batch = []
        center_winds = []

        for batch in np.unique(batches):
            # Select data of this batch
            data_batch = self.data['trl'][tag][trl_cond][self.data['trl'][tag][trl_cond]['batch'] == batch]

            epochs = data_batch['epoch']
            trials = data_batch['trial']

            if cycle is not None:
                centers = np.ones(trials.shape) * cycle
            elif center is not None:
                centers = data_batch[center]
            else:
                raise ValueError, "You have to supply either cycle or center."

            for epoch, trial, center_cyc in zip(epochs, trials, centers):
                # Select corresponding trial (uniquely identified by it's batch_num, epoch and trial
                data_cyc_idx = (self.data['cyc'][tag]['batch'] == batch) & \
                               (self.data['cyc'][tag]['epoch'] == epoch) & \
                               (self.data['cyc'][tag]['trial'] == trial)
                data_cyc_ind = self.data['cyc'][tag][data_cyc_idx]
                # Find the cycle index in which the response was made
                data_cyc_cent_idx = np.where(data_cyc_ind['cycle'] == center_cyc)[0]

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
                center_winds_batch.append(center_wind)

            if len(np.unique(batches)) == 1:
                center_winds = center_winds_batch
            else:
                center_winds.append(np.mean(center_winds_batch, axis=0))

	return np.array(center_winds)

    def plot_filled(self, x, data, avg=True, **kwargs):
	import matplotlib.pyplot as plt

        if avg: # Plot average line
            y = np.mean(data, axis=0)
            if data.shape == (0,):
                raise ValueError('Data array is empty.')
            if data.shape[1] != 0:
                sem_ = np.std(data, axis=0)
                plt.fill_between(x, y-sem_, y+sem_, alpha=.5, **kwargs)
            plt.plot(x, y, **kwargs)

        else: # Plot all lines
            for y in data:
                plt.plot(x, y, **kwargs)


def load_log(fname):
    dtype = convert_header_np(fname)
    try:
        if np.__version__.startswith('1.5.0'):
            data = np.genfromtxt(fname, dtype=dtype, skip_header=True)
        else:
            data = np.genfromtxt(fname, dtype=dtype, skiprows=1)
    except TypeError:
        raise TypeError("Error reading log file (most common not all columns generated. Check if multiple emergent runs have been writing to the same output file.\n")
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

def group_batch(data, group_names, individual_batches=False):
    """Convience function for group which groups over batches first and
    then over group_names so that SEM values are group wise"""
    data_mean, data_sem = group(data, ['batch'] + group_names)
    data_mean_gp, data_sem_gp = group(data_mean, group_names)

    if individual_batches:
        return data_mean, data_sem
    else:
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
    import pools

    # Set defaults
    mpi = False
    analyze = False
    prefix = None
    emergent = None
    set_python_exec = None
    log_dir = None
    run = False
    groups = []
    batches = 4
    verbose = False
    exclude = []

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'mg:b:f:p:e:l:rahvx:', ['mpi', 'group', 'batches', 'prefix', 'emergent', 'log_dir', 'run', 'analyze', 'help', 'verbose', 'exclude'])
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)

    for o, a in opts:
        if o in ('-m', '--mpi'):
            mpi=True
        elif o in ('-g', '--group'):
            groups.append(a)
        elif o in ('-r', '--run'):
            run = True
        elif o in ('-a', '--analyze'):
            analyze = True
        elif o in ('-p', '--prefix'):
            prefix = a
        elif o in ('-e', '--emergent'):
            emergent = a
        elif o in ('-l', '--logdir'):
            log_dir = a
        elif o in ('-b', '--batches'):
            batches = int(a)
        elif o in ('-h', '--help'):
            usage()
        elif o in ('-v', '--verbose'):
            verbose = True
        elif o in ('-x', '--exclude'):
            exclude.append(a)
        else:
            print "Command option %s not recognized."%o

    # Queue models
    import antisaccade
    import stopsignal

    if mpi:
        pool = pools.PoolMPI(emergent_exe=emergent, prefix=prefix, debug=verbose)
        pool.start_jobs(run=run, groups=groups, analyze=analyze, batches=batches, exclude=exclude) #log_dir_abs=log_dir)

    else: # Run locally
        pool = pools.Pool(emergent_exe=emergent, prefix=prefix, debug=verbose)
        pool.select(groups, exclude=exclude)
        pool.prepare(batches=batches, debug=verbose)
        if run:
            pool.run() #raise NotImplementedError, "Running locally not implemented."
        else:
            pool.analyze()

if __name__ == '__main__':
    #import doctest
    #doctest.testmod()

    main()
