import numpy as np

from copy import copy

try:
    import matplotlib
    import matplotlib.pyplot as plt
except:
    print "Could not load pyplot"

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    pass

import pools
import emergent

from math import modf, floor

def quantize(x, bins=7):
    hist, limits = np.histogram(x, bins=bins)
    out = np.empty(7)
    idx = []
    for i in range(bins):
        limit_lower = limits[i]
        limit_upper = limits[i+1]
        idx.append((x>=limit_lower) & (x<limit_upper))
        out[i] = np.mean(x[idx[-1]])

    return (out, np.asarray(idx))

class Saccade(emergent.Base):
    def __init__(self, pre_trial_cue=True, intact=True, pretrial=False, SZ=False, PD=False, NE=False, STN=False, motivation=False, dlpfc_connectivity=False, max_epoch=50, task=None, **kwargs):
	super(Saccade, self).__init__(**kwargs)

        self.ms = 4.

	self.width=.2
	self.data_trial = {}
	self.data_trial_idx = {}
	self.tags = []
        self.names = []
        if task is None:
            self.flag['task'] = 'SACCADE'
        else:
            self.flag['task'] = task

        self.flag['motivational_bias'] = 'NO_BIAS'
        self.flag['LC_mode'] = 'phasic'
        self.flag['test_SSD_mode'] = 'false'

	if pre_trial_cue:
	    self.flag['max_epoch'] = max_epoch
	    self.flag['antisaccade_block_mode'] = True
            #self.flag['DLPFC_speed_mean'] = .1
            #self.flag['DLPFC_speed_std'] = 0.0 # 0.05

	else:
	    self.flag['max_epoch'] = max_epoch
	    self.flag['antisaccade_block_mode'] = False
            #self.flag['DLPFC_speed_mean'] = .1
            #self.flag['DLPFC_speed_std'] = .0 # 0.05

        if intact:
            self.flags.append(copy(self.flag))
            self.tags.append('intact')
            self.flags[-1]['tag'] = '_' + self.tags[-1]

	if SZ:
            self.names.append('$\uparrow$tonic\nDA\nact')
            self.flags.append(copy(self.flag))
	    self.tags.append('Increased_tonic_DA')
	    self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['tonic_DA_intact'] = 0.032
	    self.flags[-1]['SZ_mode'] = 'true'

	if PD:
            self.names.append('$\downarrow$tonic\nDA\nact')
            self.flags.append(copy(self.flag))
            self.tags.append('Decreased_tonic_DA')
	    self.flags[-1]['tag'] = '_' + self.tags[-1]
	    self.flags[-1]['SZ_mode'] = 'false'
	    self.flags[-1]['tonic_DA_intact'] = 0.029

        # if NE:
        #     for tonic_NE in np.linspace(0.3,.4,8):
        #         self.names.append('NE\n%.2f' % tonic_NE)
        #         self.flags.append(copy(self.flag))
        #         self.tags.append('Tonic_NE_%f'%tonic_NE)
        #         self.flags[-1]['tag'] = '_' + self.tags[-1]
        #         self.flags[-1]['LC_mode'] = 'tonic'
        #         self.flags[-1]['tonic_NE'] = tonic_NE

        if NE:
            self.names.append('tonic\nNE\nact')
            self.flags.append(copy(self.flag))
            self.tags.append('Tonic_NE')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['LC_mode'] = 'tonic'
            self.flags[-1]['tonic_NE'] = 0.3

        if STN:
            self.names.append('$\downarrow$STN-SNr\ncons')
            self.flags.append(copy(self.flag))
            self.tags.append('DBS_on')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
	    self.flags[-1]['STN_lesion'] = .5

        if motivation:
            self.names.append('$\uparrow$preSMA-\nstriatum\ncons')
            self.flags.append(copy(self.flag))
            self.tags.append('Speed')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['motivational_bias'] = 'SPEED_BIAS'

            self.names.append('$\downarrow$preSMA-\nstriatum\ncons')
            self.flags.append(copy(self.flag))
            self.tags.append('Accuracy')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['motivational_bias'] = 'ACC_BIAS'

        if dlpfc_connectivity:
            self.names.append('$\downarrow$DLPFC\ncons')
            self.flags.append(copy(self.flag))
            self.tags.append('DLPFC_connectivity')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['dlpfc_connectivity'] = -.4

        if pretrial:
            self.names.append('$\uparrow$DLPFC\nspeed')
            self.flags.append(copy(self.flag))
            self.tags.append('Pretrial')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['DLPFC_speed_mean_mod'] = .1

	return
	# SZ run + D1 antag
	self.flag['tag'] = '_' + self.tags[2]
	self.flag['SZ_mode'] = 'true'
	self.flag['D1_antag'] = .25
	self.flag['D2_antag'] = 0
	self.flags.append(copy(self.flag))

	# SZ run + D2 antag
	self.flag['tag'] = '_' + self.tags[3]
	self.flag['SZ_mode'] = 'true'
	self.flag['D1_antag'] = 0
	self.flag['D2_antag'] = .5
	self.flags.append(copy(self.flag))

    def analyze(self):
        self.new_fig()
	self.plot_RT_histo()
	self.save_plot("RT_histo")

        self.new_fig()
        self.plot_RT()
        self.save_plot("RT")

        self.new_fig()
        self.plot_error(inhibited_as_error=True)
        self.save_plot("error")

        self.new_fig()
        self.plot_error(inhibited_as_error=False)
        self.save_plot("error_no_inhib")

        #self.new_fig()
        #self.plot_preSMA_act()

        self.new_fig()
        self.plot_RT_vs_accuracy()
        self.save_plot("RT_vs_accuracy")

        return


	self.new_fig()
	self.plot_block_influence(error=True)
	self.save_plot('block_influence_error')

	self.new_fig()
	self.plot_block_influence(IFG_act=True)
	self.save_plot('block_influence_IFG_act')

	self.new_fig()
	self.plot_block_influence(SPE=True)
	self.save_plot('block_influence_SPE')

	return

	self.new_fig()
	self.plot_go_act()
	self.save_plot("Go_act")

	return

    def select_ps_as(self, data_mean, data_sem, name):
	idx = data_mean['trial_name'] == '"Antisaccade"'
	as_mean = data_mean[idx][name]
	as_sem = data_sem[idx][name]
	idx = data_mean['trial_name'] == '"Prosaccade"'
	ps_mean = data_mean[idx][name]
	ps_sem = data_sem[idx][name]
	return ((ps_mean, as_mean), (ps_sem, as_sem))


    def quantize_cdf(self, data):
        # Select non-inhibited, correct trials.
        idx_as = (data['inhibited']==0) & (data['trial_name'] == '"Antisaccade"')
        #debug_here()
        quant, idx_quant = quantize(data[idx_as]['minus_cycles'])
        cdf = np.empty_like(quant)
        for i,idx in enumerate(idx_quant):
            cdf[i] = 1-np.mean(data[idx_as][idx]['error'])

        return (quant, cdf)

    def plot_RT_vs_accuracy(self):
        for i,tag in enumerate(self.tags):
            data = self.data[tag]

            quant, cdf = self.quantize_cdf(data)
            # Plot
            plt.plot(quant*self.ms, cdf, label=tag, color=self.colors[i])
            plt.plot(quant*self.ms, cdf, 'o', color=self.colors[i])

        plt.xlabel('Mean RT (ms)')
        plt.ylabel('Accuracy (%)')
        plt.legend(loc=0, fancybox=True)


    def plot_RT_delta(self):
        for i,tag in enumerate(self.tags):
            data = self.data[tag]

            idx_corr = (data['inhibited']==0) & (data['error']==1) & (data['trial_name'] == '"Antisaccade"')
            idx_err = (data['inhibited']==0) & (data['error']==0) & (data['trial_name'] == '"Antisaccade"')

            quant_err = quantize(data[idx_err]['minus_cycles'])
            quant_corr = quantize(data[idx_corr]['minus_cycles'])

            diff = quant_corr - quant_err
            mean_ = np.mean(np.vstack((quant_err, quant_corr)), axis=0)
            # Plot
            plt.plot(mean_, diff, label=tag)

        plt.xlabel('Mean RT')
        plt.ylabel('Delta RT (cycles)')
        plt.legend(loc=0, fancybox=True)

    def plot_RT(self):
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.2)

        tags = self.tags[1:]
        base_mean, base_sem = emergent.group_batch(self.data['intact'], ['trial_name'], individual_batches=True)
        ordered_mean_base, dummy = self.select_ps_as(base_mean, base_sem, 'minus_cycles')

	for i,tag in enumerate(tags):
	    data_mean, data_sem = emergent.group_batch(self.data[tag], ['trial_name', 'inhibited'], individual_batches=True)
            # Select non-inhibited, correct trials.
	    idx = (np.logical_not(data_mean['inhibited'])) # & np.logical_not(data_mean['error']))
	    ordered_mean, ordered_sem = self.select_ps_as(data_mean[idx], data_sem[idx], 'minus_cycles')
            assert len(ordered_mean) == 2, "No responses made in at least one condition. Tag: %s" % (tag)
            pro = ordered_mean[0] - ordered_mean_base[0]
            anti = ordered_mean[1] - ordered_mean_base[1]
	    l1 = plt.bar(0.25+i, np.mean(pro, axis=0)*self.ms,
                         yerr=emergent.sem(pro)*self.ms,
                         label='Prosaccade',
                         width=self.width, color='.7', ecolor='k')

            l2 = plt.bar(.5+i, np.mean(anti, axis=0)*self.ms,
                         yerr=emergent.sem(anti)*self.ms,
                         label='Antisaccade',
                         width=self.width, color='k', ecolor='k')

            #plt.title('Pro/Antisaccade: RTs')
            plt.ylabel('RTs (ms) relative to intact')
            #plt.xlabel('Task Condition')
            #plt.xticks((0.1, 0.35, 1.1, 1.35, 2.1, 2.35), ("Pro", "Anti", "Pro", "Anti", "Pro", "Anti"))
            plt.xticks(np.arange(len(self.names))+.5, self.names)
            plt.xlim((-0.05,1.+i))
                #plt.ylim((0, 200))
            ax = plt.gca()
            fontsize = 16
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(fontsize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(fontsize)

            plt.legend((l1, l2), ('Prosaccade', 'Antisaccade'), loc='best', frameon=False)

    def plot_error(self, inhibited_as_error=False):
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.2)

        tags = self.tags[1:]
        base_mean, base_sem = emergent.group_batch(self.data['intact'], ['trial_name'], individual_batches=True)
        ordered_mean_base, dummy = self.select_ps_as(base_mean, base_sem, 'error')

	for i,(tag,name) in enumerate(zip(tags,self.names)):
            if inhibited_as_error:
                data_mean, data_sem = emergent.group_batch(self.data[tag], ['trial_name'], individual_batches=True)
            else:
                data_mean, data_sem = emergent.group_batch(self.data[tag], ['trial_name', 'inhibited'], individual_batches=True)
                idx = data_mean['inhibited'] == 0
                data_mean = data_mean[idx]
                data_sem = data_sem[idx]


	    ordered_mean, ordered_sem = self.select_ps_as(data_mean, data_sem, 'error')

            pro = ordered_mean[0] - ordered_mean_base[0]
            anti = ordered_mean[1] - ordered_mean_base[1]

    	    plt.bar(i-.4, np.mean(anti, axis=0),
		   yerr=emergent.sem(anti),
		   label=name, color='.7', ecolor='k')  # self.colors[i], ecolor='k')

        #plt.title('Pro/Antisaccade: Errors')
        plt.ylabel('Error rate relative to intact')
        #plt.xlabel('Task Condition')
        ax = plt.gca()
        fontsize = 16
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)

        plt.xticks(range(len(self.names)), self.names) #np.linspace(0.5,len(self.tags),len(self.tags)-.5), self.tags)


    def plot_error_obsolete(self, inhibited_as_error=False):
	for i,tag in enumerate(self.tags):
            if inhibited_as_error:
                data_mean, data_sem = emergent.group_batch(self.data[tag], ['trial_name'])
            else:
                data_mean, data_sem = emergent.group_batch(self.data[tag], ['trial_name', 'inhibited'])
                idx = data_mean['inhibited'] == 0
                data_mean = data_mean[idx]
                data_sem = data_sem[idx]

	    ordered_mean, ordered_sem = self.select_ps_as(data_mean, data_sem, 'error')
	    #idx = data_mean['inhibited'] == 0
    	    plt.bar([0+i,.25+i], ordered_mean,
		   yerr=ordered_sem,
		   label=tag.replace("_", " "),
		   width=self.width, color=self.colors[i], ecolor='k')
	    plt.title('Pro/Antisaccade: Errors')
	    plt.ylabel('mean error')
	    plt.xlabel('Task Condition')
	    plt.xticks((0.1, 0.35, 1.1, 1.35, 2.1, 2.35), ("Pro", "Anti", "Pro", "Anti", "Pro", "Anti"))
	    plt.xlim((-0.05,.5+i))
            #plt.ylim((0,0.6))
	plt.legend(loc=0)

    def plot_go_act(self):
	for i,tag in enumerate(self.tags):
	    data_mean, data_sem =emergent.group_batch(self.data[tag], ['trial_name'])
            ordered_mean, ordered_sem = self.select_ps_as(data_mean, data_sem, 'Go_acts_avg')
	    plt.bar([0+i,.25+i],
		   ordered_mean,
		   yerr=ordered_sem,
		   label=tag, width=self.width, color=plt.cm.spectral(i), ecolor='k')
	    plt.title('Pro/Antisaccade: Go activity averaged across batches')
	    plt.ylabel('Go activity')
	    plt.xlabel('Task Condition')
	    plt.xticks((0.1, 0.35, 1.1, 1.35, 2.1, 2.35), ("Pro", "Anti", "Pro", "Anti", "Pro", "Anti"))
	    plt.xlim((-0.05,.5+i))

    def plot_preSMA_act(self):
        for i,tag in enumerate(self.tags):
            data_mean, data_sem = emergent.group_batch(self.data[tag], ['trial_name'])
            ordered_mean, ordered_sem = self.select_ps_as(data_mean, data_sem, 'pre_SMA_acts_avg')
            plt.bar([0+i, .25+i],
                   ordered_mean,
                   yerr=ordered_sem,
                   label=tag, width=self.width, color=self.colors[i], ecolor='k')
            plt.title('Pro/Antisaccade: pre-SMA activity averaged across batches')
            plt.ylabel('pre-SMA activity')
            plt.xlabel('Task Condition')
	    plt.xticks((0.1, 0.35, 1.1, 1.35, 2.1, 2.35), ("Pro", "Anti", "Pro", "Anti", "Pro", "Anti"))
	    plt.xlim((-0.05,.5+i))


    def plot_RT_histo(self, bins=50, range=(0,200), save=True, grid=False, cutoff=None, pro=False, ylim_max=16):
        if pro:
            saccade = '"Prosaccade"'
        else:
            saccade = '"Antisaccade"'

        tags = ['intact']

        num_tags = len(self.tags)
        cells = int(np.ceil(np.sqrt(num_tags)))
	for i,tag in enumerate(tags):
            if grid and i==0:
                fig = self.new_fig()
            if not grid:
                fig = self.new_fig()

	    #ax = fig.add_subplot(111, projection='frameaxes')
            if grid:
                ax = fig.add_subplot(cells, cells, i+1)
            else:
                ax = fig.add_subplot(111)
	    plt.hold(True)

	    # Correct
	    data=self.data[tag]
            if cutoff is not None:
                idx = (data['trial_name'] == saccade) & (data['error'] == 0.0) & (data['inhibited'] == 0) & (data['minus_cycles'] > cutoff)
            else:
                idx = (data['trial_name'] == saccade) & (data['error'] == 0.0) & (data['inhibited'] == 0)
            #print "%s: %f"%(tag, np.mean(data['error'][(data['trial_name'] == saccade) & (data['inhibited'] == 0)]))
	    histo = np.histogram(data['minus_cycles'][idx],
				 bins=bins, range=range)
            num_trials = float(len(data['minus_cycles'][idx]))
	    x = histo[1][:-1]*self.ms
	    y = histo[0]
	    ax.plot(x, (y*100) / num_trials, label="Correct", color='r')
	    ax.fill_between(x, 0, (y*100) / num_trials, color='r')

	    # Errors
            if cutoff is not None:
                idx = (data['trial_name'] == saccade) & (data['error'] == 1.0) & (data['inhibited'] == 0) & (data['minus_cycles'] > cutoff)
            else:
                idx = (data['trial_name'] == saccade) & (data['error'] == 1.0) & (data['inhibited'] == 0)

	    histo = np.histogram(data['minus_cycles'][idx],
				 bins=bins, range=range)
	    y = histo[0]
	    ax.plot(x, (y*100.) / num_trials, label="Error", alpha=0.5, color='.7')
	    ax.fill_between(x, 0, (y*100.) / num_trials, color='.7', alpha=.5)

            #if not grid:
            #    plt.legend(loc=0, fancybox=True)

            if not grid:
                #plt.title("%s errors and correct responses %s" %(saccade[2:-2], tag))
                plt.ylabel("Percentage of trials")
                plt.xlabel("Response time (ms)")
                plt.ylim((0, ylim_max))
            else:
                plt.title(tag)

    def plot_block_influence(self, error=False, IFG_act=False, SPE=False):
	"""Plot trial number of block against error (if IFG_act is false)
	or against IFG_act (if True)
	or against SPE (if True)
	"""
	for tag in ['intact']:
	    data_mean, data_sem = emergent.group_batch(self.data[tag], ['trial_name', 'trial'])
	    # select AS trials
	    data_as_mean = data_mean[data_mean['trial_name'] == '"Antisaccade"']
            data_as_sem = data_sem[data_mean['trial_name'] == '"Antisaccade"']
	    if error:
		col_name = 'error'
	    elif IFG_act:
		col_name = 'IFG_trl_acts_avg'
	    elif SPE:
		col_name = 'prediction_error'
	    else:
		col_name = 'error' # Default to error

	    plt.errorbar(data_as_mean['trial'], data_as_mean[col_name],
			yerr=data_as_sem[col_name], label=tag, lw=2)

	plt.title(col_name)

#@pools.register_group(['saccade', 'pretrial', 'nocycle', 'NE'])
class SaccadeNE(Saccade):
    def __init__(self, **kwargs):
        super(SaccadeNE, self).__init__(pre_trial_cue=True, **kwargs)

        self.tags = ['phasic', 'tonic']
        self.flag['tag'] = '_' + self.tags[0]
        self.flag['LC_mode'] = self.tags[0]
        self.flags.append(copy(self.flag))

        self.flag['tag'] = '_' + self.tags[1]
        self.flag['LC_mode'] = self.tags[1]
        self.flags.append(copy(self.flag))



@pools.register_group(['saccade', 'pretrial', 'nocycle'])
class Saccade_pretrial(Saccade):
    def __init__(self, SZ=False, **kwargs):
	super(Saccade_pretrial, self).__init__(pre_trial_cue=True, SZ=True, PD=True, NE=False, STN=True, pretrial=False, dlpfc_connectivity=True, max_epoch=50, **kwargs)

@pools.register_group(['saccade2', 'pretrial', 'nocycle'])
class Saccade_pretrial2(Saccade):
    def __init__(self, SZ=False, **kwargs):
	super(Saccade_pretrial, self).__init__(proj_name='BG_inhib9.proj', pre_trial_cue=True, SZ=True, PD=True, NE=False, STN=True, pretrial=False, dlpfc_connectivity=True, max_epoch=50, **kwargs)


@pools.register_group(['flanker', 'pretrial', 'nocycle'])
class Flanker_pretrial(Saccade):
    def __init__(self, SZ=False, **kwargs):
	super(Flanker_pretrial, self).__init__(pre_trial_cue=True, intact=True, motivation=True, STN=True, task='FLANKER', max_epoch=50, **kwargs)


#######
# DDM #
#######

class SaccadeDDMBase(Saccade):
    def __init__(self, **kwargs):
        if not kwargs.has_key('fit_ddm'):
            self.fit_ddm = True
        else:
            self.fit_ddm = kwargs['fit_ddm']
            del kwargs['fit_ddm']

        if not kwargs.has_key('mcmc'):
            self.mcmc = True
        else:
            self.mcmc = kwargs['mcmc']
            del kwargs['mcmc']

        super(SaccadeDDMBase, self).__init__(pre_trial_cue=True, intact=False, **kwargs)
        if not kwargs.has_key('depends'):
            self.depends = ['a', 'vpp', 'vcc', 'tcc']
        else:
            self.depends = kwargs['depends']

        self.condition = 'none'

    def preprocess_data(self, cutoff=0):
	# Construct data structure for ddm fits.
	# Response 0 -> prosaccade
	# Response 1 -> antisaccade

	self.stimulus = {}
	self.response = {}
	self.rt = {}
        self.subj_idx = {}
        self.tags_array = {}
        # Convert to something in the domain of seconds so we can use
        # the HDDM parameter range.
        norm = 50.

	for tag in self.tags:
	    data = self.data[tag][(self.data[tag]['inhibited'] == 0) & (self.data[tag]['minus_cycles']>cutoff)]
            if len(data) == 0:
                print "No responses made on condition: " + tag
	    self.stimulus[tag] = np.copy(data['trial_name'])
	    self.response[tag] = 1-np.copy(data['error'])
            self.subj_idx[tag] = np.copy(data['batch'])
            self.tags_array[tag] = np.empty(data.shape[0], dtype='S32')
            self.tags_array[tag][:] = tag

	    self.rt[tag] = np.copy(data['minus_cycles'])*self.ms/1000.

        # Create array with data across tags (tag becomes stimulus)
        self.stimulus_all = np.hstack([self.stimulus[tag]=='"Antisaccade"' for tag in self.tags])
        self.stimulus_name_all = np.hstack([self.stimulus[tag] for tag in self.tags])
        self.response_all = np.hstack([self.response[tag] for tag in self.tags])
        # Flip responses for prosaccades (coding is "reversed" for pro and antisaccades)
        self.response_all[self.stimulus_all=='"Prosaccade"'] = 1-self.response_all[self.stimulus_all=='"Prosaccade"']

        self.rt_all = np.hstack([self.rt[tag] for tag in self.tags])
        self.subj_idx_all = np.hstack([self.subj_idx[tag] for tag in self.tags])
        self.tags_array_all = np.hstack([self.tags_array[tag] for tag in self.tags])

        dtype = np.dtype([('instruct',np.int16), ('response', np.int16),
                          ('rt', np.float), ('subj_idx', np.int16),
                          ('dependent', 'S32'), ('trial_type', 'S32')])

        self.hddm_data = np.rec.fromarrays([self.stimulus_all, self.response_all, self.rt_all,
                                            self.subj_idx_all, self.tags_array_all, self.stimulus_name_all], dtype=dtype)

        # Create tag array with tag names for every line in *_all variables
        self.tag_all = np.hstack([tag for tag in self.tags for stimulus in self.stimulus[tag]])

    def analyze(self):
        #self.plot_RT_histo()#(cutoff=50)
        #self.save_plot('RT_histo')

        if self.fit_ddm:
            # Fitting DDM
            print "Fitting DDM."
            self.new_fig()
            self.fit_and_analyze_ddm()

            self.new_fig()
            self.plot_param_influences()

            self.new_fig()
            self.plot_hddm_fit()
        else:
            self.new_fig()
            self.plot_hddm_fit(plot_fit=False)

    def plot_hddm_fit(self, range_=(-1., 1.), bins=150., plot_fit=True):
        import hddm
        import copy
        pro_anti_param = 'none' #'a'
        x = np.linspace(range_[0], range_[1], 200)
        test_params = ('vpp', 'vcc', 'a', 'tcc')
        params = ('vpp', 'vcc', 'a', 'tcc', 't') #, 'T')
        # Plot parameters
        for i,test_param in enumerate(test_params):
            #print test_param
            plt.figure()
            for j, dep_val in enumerate(self.x):
                param_vals = {}
                param_vals['Vcc'] = 0
                param_vals['T'] = 0
                if test_param == pro_anti_param:
                    tag = "%s('%s_%.4f', \'\"Antisaccade\"\')" %(test_param, self.condition, dep_val)
                    dep_tag = '%s_%.4f'%(self.condition, dep_val)
                else:
                    tag = "%s('%s_%.4f',)" %(test_param, self.condition, dep_val)
                    dep_tag = '%s_%.4f'%(self.condition, dep_val)

                plt.subplot(3,3,j+1)

                data = self.hddm_data[(self.hddm_data['dependent']==dep_tag) & (self.hddm_data['instruct']==1)]
                data = hddm.utils.flip_errors(data)

                # Plot histogram
                hist = hddm.utils.histogram(data['rt'], bins=bins, range=range_,
                                            density=True)[0]
                plt.plot(np.linspace(range_[0], range_[1], bins), hist)

                # Plot fit
                if plot_fit:
                    fitted_params = copy.deepcopy(self.stats_dict[test_param])
                    for param in params:
                        if param == test_param:
                            param_vals[param] = fitted_params[tag]['mean']
                        else:
                            param_name = param# +'_group'
                            param_vals[param] = fitted_params[param_name]['mean']

                    fit = hddm.likelihoods.wfpt_switch.pdf(x, param_vals['vpp'], param_vals['vcc'], param_vals['Vcc'], param_vals['a'], .5, param_vals['t'], param_vals['tcc'], param_vals['T'])
                    plt.plot(x, fit)

                plt.title(tag)

            if not plot_fit:
                # Bail, no need to replot the same thing
                self.save_plot('hddm_fit')
                return

            self.save_plot('hddm_fit_%s'%test_param)

    def fit_ddm_comparison(self):
        results = {}
        results['switch'] = emergent.fit_hddm_no_deps(self.hddm_data, switch=True)
        results['nobias'] = emergent.fit_hddm_no_deps(self.hddm_data, switch=False)
        results['bias'] = emergent.fit_hddm_no_deps(self.hddm_data, switch=False, bias=True)

        print results

    def fit_and_analyze_ddm(self, retry=False):
        from multiprocessing import Pool

        self.hddm_models = {}
        test_params = ['a', 'vpp', 'vcc', 'tcc']

        experiments = [(self.hddm_data, {param: 'dependent'}) for param in test_params]
        pool = Pool(processes=len(test_params))

        stats = pool.map(emergent.fit_hddm, experiments)

        self.stats_dict = {}

        for param, stat in zip(test_params, stats):
            self.stats_dict[param] = stat

        self.new_fig()

        logps = [self.stats_dict[param]['logp'] for param in test_params]
        print "Logps:"
        for param, logp in zip(test_params, logps):
            print "& %s &  %d\\" %(param, round(logp))
        print "End"

        plt.plot(logps, lw=self.lw)
        plt.ylabel('logp')
        plt.xticks(np.arange(5), ['threshold', 'prepotent_drift', 'pfc_drift', 'tcc'])
        plt.title('HDDM model fits for different varying parameters')

        #plt.subplot(212)
        #if self.mcmc:
        #    plt.plot([self.hddm_models[test_param].DIC for test_param in test_params], lw=self.lw)
        #    plt.ylabel('DIC')
        #else:
        #    plt.plot([self.hddm_models[test_param].BIC for test_param in test_params], lw=self.lw)
        #    plt.ylabel('BIC')
        #plt.xticks(np.arange(5), ['threshold', 'prepotent_drift', 'pfc_drift', 'tcc'])
        #plt.title('HDDM model fits for different varying parameters')
        self.save_plot('model_probs')


    def plot_param_influences(self):
        pro_anti_param = 'none'
        test_params = ['a', 'vpp', 'vcc', 'tcc']

        # Plot parameters
        for i,test_param in enumerate(test_params):
            y = []
            yerr = []
            for x in self.x:
                if test_param == pro_anti_param:
                    tag = "%s('%s_%.4f', \'\"Antisaccade\"\')" %(test_param, self.condition, x)
                else:
                    tag = "%s('%s_%.4f',)" %(test_param, self.condition, x)
                y.append(self.stats_dict[test_param][tag]['mean'])
                yerr.append(self.stats_dict[test_param][tag]['standard deviation'])
                #yerr.append(self.hddm_models[test_param].params_est_std[tag])
            plt.subplot(2,2,i+1)
            plt.errorbar(self.x, y=y, yerr=yerr, label=test_param, lw=self.lw, color='.8')
            plt.plot(self.x, y, 'o', color='.8')
            plt.ylabel(test_param)
        self.save_plot('param_influences')

    def plot_var(self, var, inhibited_as_error=False): #, param):
        values = []
        #params = []
        for tag in self.tags:
            if inhibited_as_error:
                data_mean, data_sem = emergent.group(self.data[tag], ['trial_name'])
                idx = (data_mean['trial_name'] == '"Antisaccade"')
            else:
                data_mean, data_sem = emergent.group(self.data[tag], ['trial_name', 'inhibited'])
                idx = (data_mean['trial_name'] == '"Antisaccade"') & (data_mean['inhibited'] == 0)

            if np.all(idx == False):
                values.append(1.)
            else:
                values.append(data_mean[idx][var])
            #params.append(self.ddm.params[param+"_"+tag])

        plt.plot(self.x, values, 'o')

@pools.register_group(['DDM', 'DLPFC', 'mean'])
class SaccadeDDMDLPFC_mean(SaccadeDDMBase):
    def __init__(self, start=-0.022, stop=0.0, samples=6, **kwargs):
        super(SaccadeDDMDLPFC_mean, self).__init__(**kwargs)
        self.set_flags_condition('DLPFC_speed_mean_mod', start, stop, samples)

@pools.register_group(['DDM', 'DLPFC', 'connect'])
class SaccadeDDMDLPFC_connect(SaccadeDDMBase):
    def __init__(self, start=-0.4, stop=0.2, samples=6, **kwargs):
        super(SaccadeDDMDLPFC_connect, self).__init__(**kwargs)
        self.set_flags_condition('dlpfc_connectivity', start, stop, samples)

@pools.register_group(['DDM', 'STN'])
class SaccadeDDMSTN(SaccadeDDMBase):
    def __init__(self, start=0.0, stop=.75, samples=5, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.set_flags_condition('STN_lesion', start, stop, samples)

#@pools.register_group(['DDM', 'speed_acc'])
class SaccadeDDMSpeedAcc(SaccadeDDMBase):
    def __init__(self, **kwargs):
        super(SaccadeDDMSpeedAcc, self).__init__(**kwargs)
        self.tags.append('speed')
        self.flag['tag'] = '_' + self.tags[-1]
        self.flag['motivational_bias'] = 'SPEED_BIAS'
        self.flags.append(copy(self.flag))

        self.tags.append('accuracy')
        self.flag['tag'] = '_' + self.tags[-1]
        self.flag['motivational_bias'] = 'ACC_BIAS'
        self.flags.append(copy(self.flag))

        self.x = [0,1]
        self.condition = 'SpeedAcc'

@pools.register_group(['DDM', 'pro_anti'])
class SaccadeDDMpro_anti(SaccadeDDMBase):
    def __init__(self, **kwargs):
        super(SaccadeDDMpro_anti, self).__init__(**kwargs)
        self.tags.append('intact')
        self.flag['tag'] = '_' + self.tags[-1]
        self.flags.append(copy(self.flag))

    def analyze(self):
        pass


#@pools.register_group(['saccade', 'DDM', 'thalam', 'nocycle'])
class SaccadeDDMThalam(SaccadeDDMBase):
    def __init__(self, start=0.65, stop=.8, samples=5, **kwargs):
        super(SaccadeDDMThalam, self).__init__(**kwargs)
        self.set_flags_condition('thalam_thresh', start, stop, samples)

@pools.register_group(['DDM', 'DA'])
class SaccadeDDMDA(SaccadeDDMBase):
    def __init__(self, start=0.027, stop=0.032, samples=6, **kwargs):
        super(SaccadeDDMDA, self).__init__(**kwargs)
        self.set_flags_condition('tonic_DA_SZ', start, stop, samples)
        for flag in self.flags:
            flag['SZ_mode'] = True
            #flag['thalam_thresh'] = 0.7

@pools.register_group(['DDM', 'prepotent', 'striatum'])
class SaccadeDDMPrepotentStriatum(SaccadeDDMBase):
    def __init__(self, start=-.3, stop=.24, samples=6, **kwargs):
        super(SaccadeDDMPrepotentStriatum, self).__init__(**kwargs)
        self.set_flags_condition('prepotent_bias', start, stop, samples)

@pools.register_group(['DDM', 'prepotent', 'PFC'])
class SaccadeDDMPrepotentPFC(SaccadeDDMBase):
    def __init__(self, start=-.2, stop=.2, samples=6, **kwargs):
        super(SaccadeDDMPrepotentPFC, self).__init__(**kwargs)
        self.set_flags_condition('prepotent_bias_pfc', start, stop, samples)

#@pools.register_group(['DDM', 'prepotent', 'PFC+striatum', 'nocycle'])
class SaccadeDDMPrepotent(SaccadeDDMBase):
    def __init__(self, start=-.2, stop=.2, samples=5, **kwargs):
        super(SaccadeDDMPrepotent, self).__init__(**kwargs)
        self.set_flags_condition(['prepotent_bias_pfc', 'prepotent_bias'], start, stop, samples, tag='prepotent_both')

@pools.register_group(['DDM', 'preSMA'])
class SaccadeDDMpreSMA(SaccadeDDMBase):
    def __init__(self, start=-.2, stop=.1, samples=6, **kwargs):
        super(SaccadeDDMpreSMA, self).__init__(**kwargs)
        self.set_flags_condition('preSMA_striatum_bias', start, stop, samples)


@pools.register_group(['DDM', 'compare'])
class SaccadeDDMCompare(SaccadeDDMBase):
    def __init__(self, **kwargs):
        super(SaccadeDDMCompare, self).__init__(**kwargs)
        self.tags.append('intact')
        self.flag['tag'] = '_' + self.tags[-1]
        self.flags.append(copy(self.flag))

    def analyze(self):
        self.fit_ddm_comparison()

############################################
# Cycle
###########################################

class SaccadeBaseCycle(emergent.BaseCycle):
    def __init__(self, pre_trial_cue=True, task=None, **kwargs):
	super(SaccadeBaseCycle, self).__init__(**kwargs)

        # Converting factor
        self.ms = 4

        if task is None:
            task = 'SACCADE'

	self.tags = ['intact']
        self.flag['max_epoch'] = 300
        self.flag['task'] = task
        if pre_trial_cue:
            self.flag['antisaccade_block_mode'] = True
        else:
            self.flag['antisaccade_block_mode'] = False

	self.flag['tag'] = '_' + self.tags[-1]
	self.flags.append(copy(self.flag))

        #self.tags.append('NE_tonic')
        #self.flag['tag'] = '_' + self.tags[-1]
	#self.flags.append(copy(self.flag))
        self.lw = 3
        self.thalam_thresh = 0.85

    def analyze(self):
	self.new_fig()
        self.analyse_Go_NoGo_left_right()
	self.save_plot("GoNoGo_act_anti_err")

	self.new_fig()
	self.analyse_preSMA_act()
	self.save_plot("preSMA_act")

        self.new_fig()
	self.analyse_ACC_act()
	self.save_plot("ACC_act")

	self.new_fig()
	self.analyse_STN_act()
	self.save_plot("STN_act")

        self.new_fig()
        self.analyse_SC_act()
        self.save_plot('SC_act')

        self.new_fig()
        self.analyse_PFC_act()
        self.save_plot('PFC_act')

        self.new_fig()
        self.analyse_act_diff('PFC_acts_avg')
        self.save_plot('PFC_diff')

        self.new_fig()
        self.analyse_act_diff('ACC_act')
        self.save_plot('ACC_diff')

        self.new_fig()
        self.analyse_preSMA_correct_error()
        self.save_plot('preSMA_act_correct_error')

	#self.new_fig()
	#self.analyse_preSMA_act_anti_pro()
	#self.save_plot("preSMA_act_pro_anti")

    def plot_RT_histo(self, bins=75, range=(0,200), save=False, grid=True, cutoff=None, pro=False):
        if pro:
            saccade = '"Prosaccade"'
        else:
            saccade = '"Antisaccade"'

        num_tags = len(self.tags)
        cells = int(np.ceil(np.sqrt(num_tags)))

        tags = ['intact']

	for i,tag in enumerate(tags):
            if grid and i==0:
                fig = plt.figure()
            if not grid:
                fig = plt.figure()

	    #ax = fig.add_subplot(111, projection='frameaxes')
            if grid:
                ax = fig.add_subplot(cells, cells, i+1)
            else:
                ax = fig.add_subplot(111)
	    plt.hold(True)

	    # Correct
	    data = self.data['trl'][tag]
            if cutoff is not None:
                idx = (data['trial_name'] == saccade) & (data['error'] == 0.0) & (data['inhibited'] == 0) & (data['minus_cycles'] > cutoff)
            else:
                idx = (data['trial_name'] == saccade) & (data['error'] == 0.0) & (data['inhibited'] == 0)

	    histo = np.histogram(data['minus_cycles'][idx],
				 bins=bins, range=range)

            #print "%s: %f"%(tag, np.mean(data['error'][(data['trial_name'] == saccade) & (data['inhibited'] == 0)]))

	    x = histo[1][:-1]
	    y = histo[0]
	    ax.plot(x,y, label="Correct", color='.3')
	    ax.fill_between(x, 0, y, color='.7')
            #print "%s: %f"%(tag, np.mean(data['error']))
	    # Errors
            if cutoff is not None:
                idx = (data['trial_name'] == saccade) & (data['error'] == 1.0) & (data['inhibited'] == 0) & (data['minus_cycles'] > cutoff)
            else:
                idx = (data['trial_name'] == saccade) & (data['error'] == 1.0) & (data['inhibited'] == 0)

	    histo = np.histogram(data['minus_cycles'][idx],
				 bins=bins, range=range)
	    y = histo[0]
	    ax.plot(x, y, label="Error", alpha=0.5, color='r')
	    ax.fill_between(x, 0, y, color='r', alpha=.5)

            if not grid:
                plt.legend(loc=0, fancybox=True)

            if not grid:
                plt.title("%s errors and correct responses %s" %(saccade[2:-2], tag))
                plt.ylabel("Number of trials")
                plt.xlabel("Response time (cycles)")
                #plt.ylim((0, np.max(y)+20))
            else:
                plt.title(tag)

    def analyse_AS_correct_error(self, column, wind, cycle=None, center=None, tag=None):
        if tag is None:
            tag = 'intact'

        data = self.data['trl'][tag]

        AS_corr = self.extract_cycles(
            'intact',
            ((data['trial_name'] == '"Antisaccade"') &
             (data['error'] == 0) &
             (data['inhibited'] == 0)),
            column,
            wind=wind,
            cycle=cycle,
            center=center)

        AS_err = self.extract_cycles(
            'intact',
            ((data['trial_name'] == '"Antisaccade"') &
             (data['error'] == 1) &
             (data['inhibited'] == 0)),
            column,
            wind=wind,
            cycle=cycle,
            center=center)

        return (AS_corr,AS_err)

    def analyse_PS_AS(self, column, wind, cycle=None, center=None, tag=None):
        if tag is None:
            tag = 'intact'

        data = self.data['trl'][tag]

        PS = self.extract_cycles(
            tag,
            ((data['trial_name'] == '"Prosaccade"') &
             (data['error'] == 0) &
             (data['inhibited'] == 0)),
            column,
            wind=wind,
            cycle=cycle,
            center=center)

        AS = self.extract_cycles(
            tag,
            ((data['trial_name'] == '"Antisaccade"') &
             (data['error'] == 0) &
             (data['inhibited'] == 0)),
            column,
            wind=wind,
            cycle=cycle,
            center=center)

        return (PS, AS)

    def analyse_PS_AS_left_right(self, column, wind, cycle=None, center=None, tag=None):
        if tag is None:
            tag = 'intact'

        data = self.data['trl'][tag]

        PS_left = self.extract_cycles(
            tag,
            ((data['trial_name'] == '"Prosaccade"') &
             (data['group_name'] == '"left"') &
             (data['error'] == 0) &
             (data['inhibited'] == 0)),
            column,
            wind=wind,
            cycle=cycle,
            center=center)

        PS_right = self.extract_cycles(
            tag,
            ((data['trial_name'] == '"Prosaccade"') &
             (data['group_name'] == '"right"') &
             (data['error'] == 0) &
             (data['inhibited'] == 0)),
            column,
            wind=wind,
            cycle=cycle,
            center=center)

        AS_left = self.extract_cycles(
            tag,
            ((data['trial_name'] == '"Antisaccade"') &
             (data['group_name'] == '"left"') &
             (data['error'] == 0) &
             (data['inhibited'] == 0)),
            column,
            wind=wind,
            cycle=cycle,
            center=center)

        AS_right = self.extract_cycles(
            tag,
            ((data['trial_name'] == '"Antisaccade"') &
             (data['group_name'] == '"right"') &
             (data['error'] == 0) &
             (data['inhibited'] == 0)),
            column,
            wind=wind,
            cycle=cycle,
            center=center)

        return (PS_left, PS_right, AS_left, AS_right)


    def analyse_preSMA_act(self):
	wind = (100,150)

        preSMA_AS_corr, preSMA_AS_err = self.analyse_AS_correct_error('Motor__acts_avg', wind=wind, center='minus_cycles')
        preSMA_PS, preSMA_AS = self.analyse_PS_AS('Motor__acts_avg', wind=wind, center='minus_cycles')

	x=np.linspace(-wind[0],wind[1],np.sum(wind)+1)*self.ms
        # self.plot_filled(x, preSMA_AS_corr, label="Antisaccade Correct", color='r')
        # self.plot_filled(x, preSMA_AS_err, label="Antisaccade Error", color='b')
        # self.plot_filled(x, preSMA_PS, label="Prosaccade", color='k')
        plt.plot(x, np.mean(preSMA_AS_corr, axis=0), label="Antisaccade Correct", color='r', lw=3.)
        plt.plot(x, np.mean(preSMA_AS_err, axis=0), label="Antisaccade Error", color='b', lw=3.)
        plt.plot(x, np.mean(preSMA_PS, axis=0), label="Prosaccade", color='k', lw=3.)

	#plt.xlabel('Cycles around response')
	plt.ylabel('Average pre-SMA activity')
	#plt.title('pre-SMA activity during pro- and anti-saccades')
        plt.axvline(x=0, color='k')
	plt.legend(loc=0, fancybox=True)

    def analyse_PFC_act(self):
	wind = (100,150)

        preSMA_AS_corr, preSMA_AS_err = self.analyse_AS_correct_error('PFC_acts_avg', wind=wind, center='minus_cycles')
        preSMA_PS, preSMA_AS = self.analyse_PS_AS('PFC_acts_avg', wind=wind, center='minus_cycles')

	x=np.linspace(-wind[0],wind[1],np.sum(wind)+1)*self.ms
        plt.plot(x, np.mean(preSMA_AS_corr, axis=0), label="Antisaccade Correct", color='r', lw=3.)
        plt.plot(x, np.mean(preSMA_AS_err, axis=0), label="Antisaccade Error", color='b', lw=3.)
        plt.plot(x, np.mean(preSMA_PS, axis=0), label="Prosaccade", color='k', lw=3.)

	#plt.xlabel('Cycles around response')
	plt.ylabel('Average PFC activity')
	#plt.title('pre-SMA activity during pro- and anti-saccades')
        plt.axvline(x=0, color='k')
	plt.legend(loc=0, fancybox=True)

    def analyse_act_diff(self, layer, name):
        import scipy as sp
        import scipy.interpolate
        import scipy.signal
	wind = (100,150)

        AS_corr, AS_err = self.analyse_AS_correct_error(layer, wind=wind, center='minus_cycles')
        PS, AS = self.analyse_PS_AS(layer, wind=wind, center='minus_cycles')

	x = np.linspace(-wind[0],wind[1],np.sum(wind)+1)*self.ms
        y = np.mean(AS_corr, axis=0)
        deriv = np.diff(y)
        gauss = sp.signal.gaussian(20, 3)
        deriv = sp.convolve(deriv, gauss, mode='same')
        print len(deriv)
        print len(x)
        plt.plot(x[1:], deriv, label="Incongruent correct", color='r', lw=3.)

        y = np.mean(AS_err, axis=0)
        deriv = np.diff(y)
        deriv = sp.convolve(deriv, gauss, mode='same')
        plt.plot(x[1:], deriv, label="Incongruent error", color='b', lw=3.)

        y = np.mean(PS, axis=0)
        deriv = np.diff(y)
        deriv = sp.convolve(deriv, gauss, mode='same')
        plt.plot(x[1:], deriv, label="Congruent", color='k', lw=3.)

	plt.xlabel('ms relative to response')
	plt.ylabel('Average {name} activity'.format(name=name))
	#plt.title('pre-SMA activity during pro- and anti-saccades')
        plt.axvline(x=0, color='k')
	plt.legend(loc=0, fancybox=True)


    def analyse_preSMA_correct_error(self):
	wind = (0,100)

        preSMA_AS_corr_corr, preSMA_AS_err_corr = self.analyse_AS_correct_error('correct_preSMA', wind=wind, cycle=0)
        preSMA_AS_corr_err, preSMA_AS_err_err = self.analyse_AS_correct_error('error_preSMA', wind=wind, cycle=0)

	x=np.linspace(-wind[0],wind[1],np.sum(wind)+1)*self.ms
        plt.plot(x, np.mean(preSMA_AS_corr_corr, axis=0), label="Correct response unit, correct trial", color='r', lw=3.)
        plt.plot(x, np.mean(preSMA_AS_err_corr, axis=0), label="Correct response unit, error trial", color='b', lw=3.)
        plt.plot(x, np.mean(preSMA_AS_corr_err, axis=0), label="Error response unit, correct trial", color='c', lw=3.)
        plt.plot(x, np.mean(preSMA_AS_err_err, axis=0), label="Error response unit, error trial", color='k', lw=3.)

	plt.xlabel('ms from stimulus onset')
	plt.ylabel('FEF activity')
	#plt.title('pre-SMA activity during pro- and anti-saccades')
        plt.axvline(x=0, color='k')
	plt.legend(loc=4, fancybox=True)


    def analyse_ACC_act(self, tag=None):
	wind = (100,150)

        preSMA_AS_corr, preSMA_AS_err = self.analyse_AS_correct_error('ACC_act', wind=wind, center='minus_cycles', tag=tag)
        preSMA_PS, preSMA_AS = self.analyse_PS_AS('ACC_act', wind=wind, center='minus_cycles', tag=tag)

	x=np.linspace(-wind[0],wind[1],np.sum(wind)+1)*self.ms
        # self.plot_filled(x, preSMA_AS_corr, label="Antisaccade Correct", color='r')
        # self.plot_filled(x, preSMA_AS_err, label="Antisaccade Error", color='b')
        # self.plot_filled(x, preSMA_PS, label="Prosaccade", color='k')
        plt.plot(x, np.mean(preSMA_AS_corr, axis=0), label="Antisaccade Correct", color='r', lw=3.)
        plt.plot(x, np.mean(preSMA_AS_err, axis=0), label="Antisaccade Error", color='b', lw=3.)
        plt.plot(x, np.mean(preSMA_PS, axis=0), label="Prosaccade", color='k', lw=3.)

	#plt.xlabel('')
	plt.ylabel('Average ACC activity')
	#plt.title('ACC activity during pro- and anti-saccades')
        plt.axvline(x=0, color='k')
	plt.legend(loc=0, fancybox=True)

    def analyse_preSMA_act_anti_pro(self):
	wind = (0,214)
	cycle=0

        preSMA_PS, preSMA_AS = self.analyse_PS_AS('Motor__acts_avg', wind=wind, cycle=cycle)

	x = np.linspace(cycle-wind[0],cycle+wind[1],np.sum(wind)+1)*self.ms
        self.plot_filled(x, preSMA_AS, label="Antisaccade")
        self.plot_filled(x, preSMA_PS, label="Prosaccade")
	plt.xlabel('Cycles around response')
	plt.ylabel('Average pre-SMA activity')
	plt.title('pre-SMA activity during pro- and antisaccade trials')
	plt.legend(loc=0, fancybox=True)

    def analyse_SC_act(self):
	wind = (0,150)
        cycle = 0
        thalam_AS_corr, thalam_AS_corr_err = self.analyse_AS_correct_error('Thalam_unit_corr', wind=wind, cycle=cycle)#center='minus_cycles')
        thalam_AS_incorr, thalam_AS_incorr_err = self.analyse_AS_correct_error('Thalam_unit_incorr', wind=wind, cycle=cycle)#center='minus_cycles')

	x = np.linspace(-wind[0],wind[1],np.sum(wind)+1)*self.ms
        plt.plot(x, np.mean(thalam_AS_corr, axis=0), color='g', label="correct + executed", lw=3.)
        plt.plot(x, np.mean(thalam_AS_corr_err, axis=0), color='b', label="correct + not executed", lw=3.)
        plt.plot(x, np.mean(thalam_AS_incorr, axis=0), color='c', label="error + not executed", lw=3.)
        plt.plot(x, np.mean(thalam_AS_incorr_err, axis=0), color='r', label="error + executed", lw=3.)

	plt.xlabel('Time from stimulus onset (ms)')
	plt.ylabel('SC activity')
	#plt.title('SC activity during antisaccades and antisaccade errors')
	plt.legend(loc=0)
        plt.axhline(y=self.thalam_thresh, color='k')

    def analyse_STN_act(self, center=False):
        if center:
            wind = (150, 150)
            cycle = None
            center = 'minus_cycles'
            x=np.linspace(-wind[0],wind[1],np.sum(wind)+1)*self.ms
        else:
            wind = (0,150)
            cycle = 0
            center = None
            x=np.linspace(cycle-wind[0],cycle+wind[1],np.sum(wind)+1)*self.ms

        STN_AS_corr, STN_AS_err = self.analyse_AS_correct_error('STN_acts_avg', wind=wind, cycle=cycle, center=center)

        STN_PS, STN_AS = self.analyse_PS_AS('STN_acts_avg', wind=wind, cycle=cycle, center=center)


        # self.plot_filled(x, STN_AS_corr, label="Antisaccade Correct", color='r')
        # self.plot_filled(x, STN_AS_err, label="Antisaccade Error", color='0.75')
        # self.plot_filled(x, STN_PS, label="Prosaccade", color='b')
        plt.plot(x, np.mean(STN_AS_corr, axis=0), label="Antisaccade Correct", color='r', lw=3.)
        plt.plot(x, np.mean(STN_AS_err, axis=0), label="Antisaccade Error", color='0.7', lw=3.)
        plt.plot(x, np.mean(STN_PS, axis=0), label="Prosaccade", color='b', lw=3.)

        if center:
            plt.xlabel('Time from response onset (ms)')
        else:
            plt.xlabel('Time from cue onset (ms)')

	plt.ylabel('Average STN activity')
	#plt.title('STN activity during successfully and erronous antisaccades')
	plt.legend(loc=0, fancybox=True)


    def analyse_Go_NoGo_left_right(self):
        wind = (135,135)
        center = 'minus_cycles'

        y_lim_lower = -.1

        Go_PS_left_left, Go_PS_left_right, Go_AS_left_left, Go_AS_left_right = self.analyse_PS_AS_left_right('left_Go', wind=wind, center=center)
        Go_PS_right_left, Go_PS_right_right, Go_AS_right_left, Go_AS_right_right = self.analyse_PS_AS_left_right('right_Go', wind=wind, center=center)

        NoGo_PS_left_left, NoGo_PS_left_right, NoGo_AS_left_left, NoGo_AS_left_right = self.analyse_PS_AS_left_right('left_NoGo', wind=wind, center=center)
        NoGo_PS_right_left, NoGo_PS_right_right, NoGo_AS_right_left, NoGo_AS_right_right = self.analyse_PS_AS_left_right('right_NoGo', wind=wind, center=center)

	x=np.linspace(-wind[0],wind[1],np.sum(wind)+1)*self.ms

        # RESP_STIM
        ax1=plt.subplot(221)
	ax1.spines['right'].set_color('none')
	#ax.spines['bottom'].set_position('center')
	ax1.spines['top'].set_color('none')
	ax1.xaxis.set_ticks_position('bottom')
	ax1.yaxis.set_ticks_position('left')
        plt.setp(ax1.get_xticklabels(), visible=False)

        try:
            # self.plot_filled(x, Go_PS_left_left, label='left', color='r')
            # self.plot_filled(x, Go_PS_left_right, label='right', color='b')
            plt.plot(x, np.mean(Go_PS_left_left, axis=0), label='left', color='r', lw=3.)
            plt.plot(x, np.mean(Go_PS_left_right, axis=0), label='right', color='b', lw=3.)

            plt.axvline(x=0, color='k')
            #plt.legend(loc=0)
            plt.ylim(y_lim_lower,1.)
        except ValueError:
            pass


        ax2 = plt.subplot(222)
	ax2.spines['right'].set_color('none')
	#ax.spines['bottom'].set_position('center')
	ax2.spines['top'].set_color('none')
	ax2.xaxis.set_ticks_position('bottom')
	ax2.yaxis.set_ticks_position('left')
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        try:
            #self.plot_filled(x, Go_AS_left_right, color='r')
            #self.plot_filled(x, Go_AS_right_right, color='b')
            plt.plot(x, np.mean(Go_AS_left_right, axis=0), color='r', lw=3.)
            plt.plot(x, np.mean(Go_AS_right_right, axis=0), color='b', lw=3.)
            plt.axvline(x=0, color='k')
            plt.ylim(y_lim_lower, 1.)
        except ValueError:
            pass
        #plt.legend(loc=0)


        ax3 = plt.subplot(223)
	ax3.spines['right'].set_color('none')
	#ax.spines['bottom'].set_position('center')
	ax3.spines['top'].set_color('none')
	ax3.xaxis.set_ticks_position('bottom')
	ax3.yaxis.set_ticks_position('left')

        try:
            # self.plot_filled(x, NoGo_PS_left_left, color='r')
            # self.plot_filled(x, NoGo_PS_left_right, color='b')
            plt.plot(x, np.mean(NoGo_PS_left_left, axis=0), color='r', lw=3.)
            plt.plot(x, np.mean(NoGo_PS_left_right, axis=0), color='b', lw=3.)
            plt.ylim(y_lim_lower, .5)
            plt.axvline(x=0, color='k')
        except ValueError:
            pass
        #plt.legend(loc=0)


        ax4 = plt.subplot(224)
	ax4.spines['right'].set_color('none')
	#ax.spines['bottom'].set_position('center')
	ax4.spines['top'].set_color('none')
	ax4.xaxis.set_ticks_position('bottom')
	ax4.yaxis.set_ticks_position('left')
        plt.setp(ax4.get_yticklabels(), visible=False)
        try:
            # self.plot_filled(x, NoGo_AS_left_right, color='r')
            # self.plot_filled(x, NoGo_AS_right_right, color='b')
            plt.plot(x, np.mean(NoGo_AS_left_right, axis=0), color='r', lw=3.)
            plt.plot(x, np.mean(NoGo_AS_right_right, axis=0), color='b', lw=3.)
            plt.axvline(x=0, color='k')
            plt.ylim(y_lim_lower, .5)
        except ValueError:
            pass
        #plt.legend(loc=0)

@pools.register_group(['saccade', 'cycle', 'brown'])
class SaccadeCycleBrown(SaccadeBaseCycle):
    def __init__(self, **kwargs):
        super(SaccadeCycleBrown, self).__init__(pre_trial_cue=True, **kwargs)
        self.flags[-1]['brown_indirect'] = True

    def analyze(self):
        self.new_fig()
        self.analyse_Go_NoGo_left_right()
        self.save_plot('striatum')

@pools.register_group(['saccade', 'cycle', 'intact'])
class SaccadeCyclePreTrial(SaccadeBaseCycle):
    def __init__(self, **kwargs):
        super(SaccadeCyclePreTrial, self).__init__(pre_trial_cue=True, **kwargs)

@pools.register_group(['saccade', 'cycle', 'pretrial', 'SC_STN'])
class SaccadeCycleSCSTN(SaccadeBaseCycle):
    def __init__(self, **kwargs):
        super(SaccadeCycleSCSTN, self).__init__(pre_trial_cue=True, **kwargs)
        # Add SC->STN connection.
        self.flags[-1]['SC_STN_con'] = 1.

    def analyze(self):
	self.new_fig()
	self.analyse_STN_act()
	self.save_plot("STN_act_stim_aligned")

	self.new_fig()
	self.analyse_STN_act(center=True)
	self.save_plot("STN_act_response_aligned")


@pools.register_group(['saccade', 'prepotent', 'PFC', 'cycle'])
class SaccadePrepotentPFCConflict(SaccadeBaseCycle):
    def __init__(self, start=-.1, stop=.1, samples=4, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.set_flags_condition('prepotent_bias_pfc', start, stop, samples)


    def analyze(self):
        self.new_fig()
        self.analyse_prepotent('ACC_act')
	plt.ylabel('Average conflict activity')
        plt.axvline(x=0, color='k')
	plt.legend(loc=0, fancybox=True)
        self.save_plot('prepotent_ACC')

        self.new_fig()
        self.analyse_prepotent('PFC_acts_avg')
	plt.ylabel('Average conflict activity')
        plt.axvline(x=0, color='k')
	plt.legend(loc=0, fancybox=True)
        self.save_plot('prepotent_DLPFC')

        self.new_fig()
        self.analyse_prepotent('ACC_act', func=np.std)
	plt.ylabel('Std conflict activity')
        plt.axvline(x=0, color='k')
	plt.legend(loc=0, fancybox=True)
        self.save_plot('prepotent_ACC_std')

        self.new_fig()
        self.analyse_prepotent('PFC_acts_avg', func=np.std)
	plt.ylabel('Std conflict activity')
        plt.axvline(x=0, color='k')
	plt.legend(loc=0, fancybox=True)
        self.save_plot('prepotent_DLPFC_std')

    def analyse_prepotent(self, layer, func=None):
        if func is None:
            func = np.mean

	wind = (100,150)

        for i, tag in enumerate(self.tags):
            if tag == 'intact':
                continue
            preSMA_AS_corr, preSMA_AS_err = self.analyse_AS_correct_error(layer, wind=wind, center='minus_cycles', tag=tag)
            preSMA_PS, preSMA_AS = self.analyse_PS_AS(layer, wind=wind, center='minus_cycles', tag=tag)

            x=np.linspace(-wind[0],wind[1],np.sum(wind)+1)*self.ms
            alpha = (i+1.)/len(self.tags)
            plt.plot(x, func(preSMA_AS_corr, axis=0), label=str(alpha), color='r', lw=3., alpha=alpha)
            plt.plot(x, func(preSMA_AS_err, axis=0), color='b', lw=3., alpha=alpha)
            plt.plot(x, func(preSMA_PS, axis=0), color='k', lw=3., alpha=alpha)

	#plt.xlabel('Cycles around response')
	plt.ylabel('Average DLPFC activity')
	#plt.title('pre-SMA activity during pro- and anti-saccades')
        plt.axvline(x=0, color='k')
	plt.legend(loc=0, fancybox=True)


@pools.register_group(['saccade', 'fast', 'PFC', 'conf'])
class SaccadePrepotentFastDLPFC(Saccade):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.flags[-1]['DLPFC_speed_mean_mod'] = 1


    def analyze(self):
        self.new_fig()
	self.plot_RT_histo(ylim_max=54)
	self.save_plot("RT_histo")

        self.new_fig()
        self.plot_RT_vs_accuracy()
	self.save_plot("RT_vs_accuracy")

@pools.register_group(['saccade', 'fast', 'PFC', 'noconf'])
class SaccadePrepotentFastDLPFCNoConf(Saccade):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(proj_name='BG_inhib8_fast_pfc',**kwargs)
        #self.flags[-1]['DLPFC_speed_mean_mod'] = 1


    def analyze(self):
        self.new_fig()
	self.plot_RT_histo(ylim_max=54)
	self.save_plot("RT_histo")
