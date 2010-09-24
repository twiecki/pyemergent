import numpy as np
    
from copy import copy

try:
    import matplotlib.pyplot as plt
except:
    print "Could not load pyplot"

try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    pass

import pools
import emergent


class Saccade(emergent.Base):
    def __init__(self, pre_trial_cue=True, intact=True, SZ=False, PD=False, max_epoch=50, **kwargs):
	super(Saccade, self).__init__(**kwargs)

	# Set ddm to have fixed starting point and that drift rate
	# depends on the stimulus
	#self.ddm_params['set'] = 'z 0.5'
	#self.ddm_params['depends'] = 'v stimulus'
	#self.ddm_params['format'] = 'RESPONSE TIME'
	
	self.width=.2
	self.data_trial = {}
	self.data_trial_idx = {}
	self.tags = []
	self.flag['task'] = 'SACCADE'
        self.flag['motivational_bias'] = 'NO_BIAS'
        self.flag['test_SSD_mode'] = 'false'

	if pre_trial_cue:
	    self.flag['max_epoch'] = max_epoch
	    self.flag['antisaccade_block_mode'] = True
            self.flag['DLPFC_speed_mean'] = .02
            self.flag['DLPFC_speed_std'] = 0.01 # 0.05

	else:
	    self.flag['max_epoch'] = max_epoch
	    self.flag['antisaccade_block_mode'] = False
            self.flag['DLPFC_speed_mean'] = .01
            self.flag['DLPFC_speed_std'] = .1 # 0.05

        if intact:
            # Intact run
            self.tags.append('intact')
            self.flag['tag'] = '_' + self.tags[-1]
            self.flag['SZ_mode'] = 'false'
            self.flag['D1_antag'] = 0
            self.flag['D2_antag'] = 0
            self.flags.append(copy(self.flag))

	if SZ:
	    # SZ run
	    self.tags.append('Increased_tonic_DA')
	    self.flag['tag'] = '_' + self.tags[-1]
	    self.flag['SZ_mode'] = 'true'
	    self.flag['D1_antag'] = 0
	    self.flag['D2_antag'] = 0
 	    self.flags.append(copy(self.flag))

	if PD:
	    self.tags.append('Decreased_tonic_DA')
	    self.flag['tag'] = '_' + self.tags[3]
	    self.flag['SZ_mode'] = 'false'
	    self.flag['tonic_DA_intact'] = 0.01
	    self.flag['D1_antag'] = 0
	    self.flag['D2_antag'] = 0
	    self.flag['salience'] = 1
	    self.flags.append(copy(self.flag))

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
	self.plot_RT_histogram()
	self.save_plot("Hikosaka_switch_histo")

        self.new_fig()
        self.plot_RT()

        self.new_fig()
        self.plot_error()

        self.new_fig()
        self.plot_preSMA_act()

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

	return (np.concatenate((ps_mean, as_mean)), np.concatenate((ps_sem, as_sem)))
	    
	
    def plot_RT(self):
	for i,tag in enumerate(self.tags):
	    data_mean, data_sem = emergent.group_batch(self.data[tag], ['trial_name', 'inhibited', 'error'])
            # Select non-inhibited, correct trials.
	    idx = (np.logical_not(data_mean['inhibited']) & np.logical_not(data_mean['error']))
	    ordered_mean, ordered_sem = self.select_ps_as(data_mean[idx], data_sem[idx], 'minus_cycles')
            assert len(ordered_mean) == 2, "No responses made in at least one condition"

	    plt.bar([0+i,.25+i], ordered_mean,
		   yerr=ordered_sem,
		   label=tag.replace("_", " "),
		   width=self.width, color=self.colors[i], ecolor='k')
	    plt.title('Pro/Antisaccade: RTs')
	    plt.ylabel('RTs (cycles)')
	    plt.xlabel('Task Condition')
	    plt.xticks((0.1, 0.35, 1.1, 1.35, 2.1, 2.35), ("Pro", "Anti", "Pro", "Anti", "Pro", "Anti"))
	    plt.xlim((-0.05,.5+i))
	    #plt.ylim((0, 200))

	plt.legend(loc=0, fancybox=True)

    def plot_error(self):
	for i,tag in enumerate(self.tags):
	    data_mean, data_sem = emergent.group_batch(self.data[tag], ['trial_name', 'inhibited'])
            idx = data_mean['inhibited'] == 0
	    ordered_mean, ordered_sem = self.select_ps_as(data_mean[idx], data_sem[idx], 'error')
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
            plt.ylim((0,0.6))
	plt.legend(loc=2)

    def plot_go_act(self):
	for i,tag in enumerate(self.tags):
	    data_mean, data_sem =emergent.group_batch(self.data[tag], ['trial_name'])
            ordered_mean, ordered_sem = self.select_ps_as(data_mean, data_sem, 'Go_acts_avg')
	    plt.bar([0+i,.25+i],
		   ordered_mean,
		   yerr=ordered_sem,
		   label=tag, width=self.width, color=self.colors[i], ecolor='k')
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
            

    def plot_RT_histogram(self, bins=75, range=(0,200)):
	for i,tag in enumerate(self.tags):
	    fig = plt.figure()

	    #ax = fig.add_subplot(111, projection='frameaxes')
	    ax = fig.add_subplot(111)
	    plt.hold(True)

	    # Correct
	    data=self.data[tag]
	    idx = (data['trial_name'] == '"Antisaccade"') & (data['error'] == 0.0) & (data['inhibited'] == 0)
	    histo = np.histogram(data['minus_cycles'][idx],
				 bins=bins, range=range)
	    x = histo[1][:-1]
	    y = histo[0]
	    ax.plot(x,y, label="Correct", color=self.colors[0])
	    ax.fill_between(x, 0, y, color=self.colors[0])

	    # Errors
	    idx = (data['trial_name'] == '"Antisaccade"') & (data['error'] == 1.0) & (data['inhibited'] == 0)
	    histo = np.histogram(data['minus_cycles'][idx],
				 bins=bins, range=range)
	    y = histo[0]
	    ax.plot(x, y, label="Error", alpha=0.5, color=self.colors[1])
	    ax.fill_between(x, 0, y, color=self.colors[1], alpha=.5)
	    
	    plt.legend(loc=0, fancybox=True)

	    plt.title("Antisaccade errors and correct responses " + tag)
	    plt.ylabel("Number of trials")
	    plt.xlabel("Response time (cycles)")
            plt.ylim((0, np.max(y)+20))

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


@pools.register_group(['saccade', 'ontrial', 'nocycle'])
class Saccade_ontrial(Saccade):
    def __init__(self, **kwargs):
	super(Saccade_ontrial, self).__init__(pre_trial_cue=False, **kwargs)
        self.tags.append('Lower_salience_detection')
        self.flag['tag'] = '_' + self.tags[-1]
        self.flag['SZ_mode'] = 'false'
        self.flag['D1_antag'] = 0
        self.flag['D2_antag'] = 0
        self.flag['salience'] = .25
        self.flags.append(copy(self.flag))

@pools.register_group(['saccade', 'pretrial', 'nocycle'])
class Saccade_pretrial(Saccade):
    def __init__(self, SZ=False, **kwargs):
	super(Saccade_pretrial, self).__init__(pre_trial_cue=True, SZ=SZ, max_epoch=500, **kwargs)



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

        if not kwargs.has_key('fit_lba'):
            self.fit_lba = True
        else:
            self.fit_ddm = kwargs['fit_lba']
            del kwargs['fit_lba']
            
        super(SaccadeDDMBase, self).__init__(pre_trial_cue=True, intact=False, **kwargs)
        if not kwargs.has_key('depends'):
            self.depends = ['a', 'v', 'z']
        else:
            self.depends = kwargs['depends']

    def set_flags_condition(self, condition, start, stop, samples):
        self.x = np.linspace(start, stop, samples)
        self.condition = condition
        for i in self.x:
            tag = 'DDM_' + condition + '_' + str(i)
            self.tags.append(tag)
            self.flag['tag'] = '_' + tag
            self.flag[condition] = i
            self.flags.append(copy(self.flag))

    def preprocess_data(self):
	# Construct data structure for ddm fits.
	# Response 0 -> prosaccade
	# Response 1 -> antisaccade

	self.stimulus = {}
	self.response = {}
	self.rt = {}
        self.subj_idx = {}
        self.tags_array = {}
        # fastdm expects reaction times in the domain of 5 secs (max)
        max_rt = 300
        norm = max_rt/5.

	for tag in self.tags:
	    data = self.data[tag][(self.data[tag]['inhibited'] == 0) & (self.data[tag]['trial_name'] == '"Antisaccade"')]
            if len(data) == 0:
                print "No responses made on condition: " + tag
	    self.stimulus[tag] = np.copy(data['trial_name'])
	    self.response[tag] = 1-np.copy(data['error'])
            self.subj_idx[tag] = np.copy(data['batch'])
            self.tags_array[tag] = np.empty(data.shape[0], dtype='S32')
            self.tags_array[tag][:] = tag
            
	    self.rt[tag] = np.copy(data['minus_cycles'])/norm
	    #idx_as = self.stimulus[tag] == '"Antisaccade"'
	    # Invert errors of antisaccade trials, that way,
	    # successful antisaccades are 1, while errors
	    # (i.e. prosaccades) are 0
	    #self.response[tag][idx_as] = 1 - self.response[tag][idx_as]

        # Create array with data across tags (tag becomes stimulus)
        self.stimulus_all = np.hstack([self.stimulus[tag] for tag in self.tags])
        self.response_all = np.hstack([self.response[tag] for tag in self.tags])
        self.rt_all = np.hstack([self.rt[tag] for tag in self.tags])
        self.subj_idx_all = np.hstack([self.subj_idx[tag] for tag in self.tags])
        self.tags_array_all = np.hstack([self.tags_array[tag] for tag in self.tags])

        dtype = np.dtype([('stim','S18'), ('response', np.int16),
                          ('rt', np.float), ('subj_idx', np.int16),
                          ('dependent', 'S32')])

        self.hddm_data = np.rec.fromarrays([self.stimulus_all, self.response_all, self.rt_all,
                                   self.subj_idx_all, self.tags_array_all], dtype=dtype)
        
        # Create tag array with tag names for every line in *_all variables
        self.tag_all = np.hstack([tag for tag in self.tags for stimulus in self.stimulus[tag]])

    def analyze(self):
        #ps_ddms, as_ddms = self.fit_ddm_as_separate(plot=True, PS=False, AS=True)

        #self.ddm = self.fit_ddm_across_conds(depends=self.depends, plot=plot, AS=True)
        if self.fit_ddm:
            self.hddm_model_a = self.fit_hddm(depends_on={'a':['dependent']})
            self.hddm_model_z = self.fit_hddm(depends_on={'z':['dependent']})
            self.hddm_model_v = self.fit_hddm(depends_on={'v':['dependent']})

            print 'logp a: %f' % self.hddm_model_a.mcmc_model.logp
            print 'logp v: %f' % self.hddm_model_v.mcmc_model.logp
            print 'logp z: %f' % self.hddm_model_z.mcmc_model.logp

        if self.fit_lba:
            self.lba_model_a = self.fit_hlba(depends_on={'a':['dependent']})
            self.lba_model_z = self.fit_hlba(depends_on={'z':['dependent']})
            self.lba_model_v = self.fit_hlba(depends_on={'v0':['dependent'], 'v1':['dependent']})

            print 'logp a: %f' % self.lba_model_a.mcmc_model.logp
            print 'logp v: %f' % self.lba_model_v.mcmc_model.logp
            print 'logp z: %f' % self.lba_model_z.mcmc_model.logp
           
        
        #print self.hddm_model.params_est
        if self.plot:
            self.plot_RT_histogram()
            self.plot_var('error')

    def plot_var(self, var): #, param):
        values = []
        #params = []
        for tag in self.tags:
            data_mean, data_sem = emergent.group(self.data[tag], ['trial_name', 'inhibited'])
            idx = (data_mean['trial_name'] == '"Antisaccade"') & (data_mean['inhibited'] == 0)
            if np.all(idx == False):
                values.append(1.)
            else:
                values.append(data_mean[idx][var])
            #params.append(self.ddm.params[param+"_"+tag])

        self.new_fig()
        plt.plot(self.x, values, 'o')
        
        #self.new_fig()
        #plt.plot(self.x, p
        
@pools.register_group(['saccade', 'DDM', 'DLPFC', 'nocycle', 'mean'])
class SaccadeDDMDLPFC_mean(SaccadeDDMBase):
    def __init__(self, start=0.01, stop=0.04, samples=5, **kwargs):
        super(SaccadeDDMDLPFC_mean, self).__init__(**kwargs)
        self.set_flags_condition('DLPFC_speed_mean', start, stop, samples)
        #for flag in self.flags:
        #    flag['DLPFC_speed_std'] = 0.01

#@pools.register_group(['saccade', 'DDM', 'DLPFC', 'nocycle', 'std'])
class SaccadeDDMDLPFC_std(SaccadeDDMBase):
    def __init__(self, start=0.0, stop=0.9, samples=7, **kwargs):
        super(SaccadeDDMDLPFC_std, self).__init__(**kwargs)
        self.depends = ['z', 'sz', 'a']
        self.set_flags_condition('DLPFC_speed_std', start, stop, samples)

@pools.register_group(['saccade', 'DDM', 'thalam', 'nocycle'])
class SaccadeDDMThalam(SaccadeDDMBase):
    def __init__(self, start=0.3, stop=.85, samples=5, **kwargs):
        super(SaccadeDDMThalam, self).__init__(**kwargs)
        self.set_flags_condition('thalam_thresh', start, stop, samples)

@pools.register_group(['saccade', 'DDM', 'DA', 'nocycle'])
class SaccadeDDMDA(SaccadeDDMBase):
    def __init__(self, start=0.027, stop=0.04, samples=10, **kwargs):
        super(SaccadeDDMDA, self).__init__(**kwargs)
        self.set_flags_condition('tonic_DA_SZ', start, stop, samples)
        for flag in self.flags:
            flag['SZ_mode'] = True
            flag['thalam_thresh'] = 0.7

# @pools.register_group(['saccade', 'DDM', 'STN', 'nocycle'])
class SaccadeDDMSTN(SaccadeDDMBase):
     def __init__(self, start=0, stop=.5, samples=5, **kwargs):
         super(SaccadeDDMSTN, self).__init__(**kwargs)
         self.set_flags_condition('stn_bias', start, stop, samples)


#@pools.register_group(['saccade', 'DDM', 'compare', 'nocycle'])
class SaccadeDDMCompare(SaccadeDDMBase):
    def __init__(self, DA=0.032, DLPFC_speed_mean=0.1, **kwargs):
        super(SaccadeDDMCompare, self).__init__(**kwargs)
        
        # Run DA model
        tag_DA = 'DDM_DA_'+str(DA)
        self.tags.append(tag_DA)
        self.flag['tag'] = '_' + tag_DA
        self.flag['tonic_DA_SZ'] = DA
        self.flag['SZ_mode'] = True
        self.flags.append(copy(self.flag))

        # Run DLPFC model
        tag_DLPFC = 'DDM_DLPFC_'+str(DLPFC_speed_mean)
        self.tags.append(tag_DLPFC)
        self.flag['tag'] = '_' + tag_DLPFC
        self.flag['tonic_DA_SZ'] = 0.028
        self.flag['DLPFC_speed_mean'] = DLPFC_speed_mean
        self.flag['SZ_mode'] = False
        self.flags.append(copy(self.flag))

        self.x = np.array([0,1])
        self.condition = 'Compare'

    def analyze(self):
        #self.fit_ddm_as_joint()
        
        self.new_fig()
        self.plot_error()

        self.plot_RT_histogram()

class compare_DDM_models(object):
    def __init__(self, DA_model, DLPFC_model):
        self.err_DA, self.param_DA = self.get_params(DA_model)
        self.err_DLPFC, self.param_DLPFC = self.get_params(DLPFC_model)
        
    def get_params(self, model):
        errors = []
        params = []

        for tag in enumerate(model.tags):
            # Save errors
            data = emergent.group(model.data[tag], ['trial_name'])
            errors.append(data[data['trial_name'] == '"Antisaccade"']['error'])
            
            # Save fitted DDM params
            params.append({})
            for param in model.params:
                params[-1][param] = model.ddm.params[param+"_"+tag]

        return (errors, params)

    def plot_correlation(self, errors, params, param_name):
        for error, param in zip(errors, params):
            pass
        

############################################
# Cycle
###########################################

class SaccadeBaseCycle(emergent.BaseCycle):
    def __init__(self, pre_trial_cue=True, **kwargs):
	super(SaccadeBaseCycle, self).__init__(**kwargs)
	self.tags = ['intact']
        self.flag['max_epoch'] = 50
        self.flag['task'] = 'SACCADE'
        if pre_trial_cue:
            self.flag['antisaccade_block_mode'] = True
        else:
            self.flag['antisaccade_block_mode'] = False

	self.flag['tag'] = '_' + self.tags[0]
        self.flag['DLPFC_speed_mean'] = .02
        self.flag['DLPFC_speed_std'] = .01
	self.flags.append(copy(self.flag))

        self.lw = 2
        self.thalam_thresh = 0.8

    def analyze(self):
	self.new_fig()
        self.analyse_Go_NoGo_left_right()
	self.save_plot("GoNoGo_act_anti_err")

	self.new_fig()
	self.analyse_preSMA_act()
	self.save_plot("preSMA_act")
	
	self.new_fig()
	self.analyse_STN_act()
	self.save_plot("STN_act")

        self.new_fig()
        self.analyse_SC_act()
        self.save_plot('SC_act')
        
	#self.new_fig()
	#self.analyse_preSMA_act_anti_pro()
	#self.save_plot("preSMA_act_pro_anti")


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

	x=np.linspace(-wind[0],wind[1],np.sum(wind)+1)
        self.plot_filled(x, preSMA_AS_corr, label="Antisaccade Correct", color='g')
        self.plot_filled(x, preSMA_AS_err, label="Antisaccade Error", color='r')
        self.plot_filled(x, preSMA_PS, label="Prosaccade", color='b')
	plt.xlabel('Cycles around response')
	plt.ylabel('Average pre-SMA activity')
	plt.title('pre-SMA activity during pro- and anti-saccades')
        plt.axvline(x=0, color='k')
	plt.legend(loc=0, fancybox=True)

    def analyse_preSMA_act_anti_pro(self):
	wind = (0,214)
	cycle=0

        preSMA_PS, preSMA_AS = self.analyse_PS_AS('Motor__acts_avg', wind=wind, cycle=cycle)

	x=np.linspace(cycle-wind[0],cycle+wind[1],np.sum(wind)+1)
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
        
	x=np.linspace(-wind[0],wind[1],np.sum(wind)+1)
        self.plot_filled(x, thalam_AS_corr, color='g', label="Correct_exec")
        self.plot_filled(x, thalam_AS_corr_err, color='b', label="Correct_not_exec")
        self.plot_filled(x, thalam_AS_incorr, color='c', label="Error_not_exec")
        self.plot_filled(x, thalam_AS_incorr_err, color='r', label="Error_exec")

	plt.xlabel('Cycles around antisaccade reponse')
	plt.ylabel('Average SC activity')
	plt.title('SC activity during antisaccades and antisaccade errors')
	plt.legend(loc=2)
        plt.axhline(y=self.thalam_thresh, color='k')

    def analyse_STN_act(self):
	wind = (0,200)
	cycle = 0
        STN_AS_corr, STN_AS_err = self.analyse_AS_correct_error('STN_acts_avg', wind=wind, cycle=cycle)

        STN_PS, STN_AS = self.analyse_PS_AS('STN_acts_avg', wind=wind, cycle=0) #center='minus_cycles')

	x=np.linspace(cycle-wind[0],cycle+wind[1],np.sum(wind)+1)
        self.plot_filled(x, STN_AS_corr, label="Antisaccade Correct", color='g')
        self.plot_filled(x, STN_AS_err, label="Antisaccade Error", color='r')
        self.plot_filled(x, STN_PS, label="Prosaccade", color='b')
	plt.xlabel('Cycles around response')
	plt.ylabel('Average STN activity')
	plt.title('STN activity during successfully and erronous antisaccades')
	plt.legend(loc=0, fancybox=True)

        
    def analyse_Go_NoGo_left_right(self):
        wind = (150,150)
        center = 'minus_cycles'
        
        Go_PS_left_left, Go_PS_left_right, Go_AS_left_left, Go_AS_left_right = self.analyse_PS_AS_left_right('left_Go', wind=wind, center=center)
        Go_PS_right_left, Go_PS_right_right, Go_AS_right_left, Go_AS_right_right = self.analyse_PS_AS_left_right('right_Go', wind=wind, center=center)

        NoGo_PS_left_left, NoGo_PS_left_right, NoGo_AS_left_left, NoGo_AS_left_right = self.analyse_PS_AS_left_right('left_NoGo', wind=wind, center=center)
        NoGo_PS_right_left, NoGo_PS_right_right, NoGo_AS_right_left, NoGo_AS_right_right = self.analyse_PS_AS_left_right('right_NoGo', wind=wind, center=center)

	x=np.linspace(-wind[0],wind[1],np.sum(wind)+1)

        # RESP_STIM
        plt.subplot(221)
        try:
            self.plot_filled(x, Go_PS_left_left, label='left', color='r')
            self.plot_filled(x, Go_PS_left_right, label='right', color='b')
            plt.axvline(x=0, color='k')
            plt.legend(loc=0)
        except ValueError:
            pass
                

        plt.subplot(222)
        try:
            self.plot_filled(x, Go_AS_left_right, color='r')
            self.plot_filled(x, Go_AS_right_right, color='b')
            plt.axvline(x=0, color='k')
        except ValueError:
            pass
        #plt.legend(loc=0)

        #plt.subplot(323)
        #self.plot_filled(x, Go_PS_left_left, color='r')
        #self.plot_filled(x, Go_PS_left_right, color='b')
        #plt.axvline(x=0, color='k')
        #plt.legend(loc=0)

        #plt.subplot(324)
        #self.plot_filled(x, Go_AS_left_right, color='r')
        #self.plot_filled(x, Go_AS_right_right, color='b')
        #plt.axvline(x=0, color='k')
        #plt.legend(loc=0)

        plt.subplot(223)
        try:
            self.plot_filled(x, NoGo_PS_left_left, color='r')
            self.plot_filled(x, NoGo_PS_left_right, color='b')
            plt.axvline(x=0, color='k')
        except ValueError:
            pass
        #plt.legend(loc=0)

        plt.subplot(224)
        try:
            self.plot_filled(x, NoGo_AS_left_right, color='r')
            self.plot_filled(x, NoGo_AS_right_right, color='b')
            plt.axvline(x=0, color='k')
        except ValueError:
            pass
        #plt.legend(loc=0)


@pools.register_group(['saccade', 'cycle', 'pretrial'])
class SaccadeCyclePreTrial(SaccadeBaseCycle):
    def __init__(self, **kwargs):
        super(SaccadeCyclePreTrial, self).__init__(pre_trial_cue=True, **kwargs)

#    def analyze(self):
#         self.analyse_Go_NoGo_left_right()



#@pools.register_group(['saccade', 'cycle', 'ontrial'])
class SaccadeCycleOnTrial(SaccadeBaseCycle):
    def __init__(self, **kwargs):
        super(SaccadeCycleOnTrial, self).__init__(pre_trial_cue=False, **kwargs)

