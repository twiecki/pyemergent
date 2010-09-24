from __future__ import division
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
from emergent import sem

def calc_SSRT(GoRT, SSD, numtrials=None):
    """Calculate the SSRT for a give GoRT distribution (array) and a given staircase run,
    the 50% inhibitory interval is computed using numtrials last trials of staircase"""
    median = np.median(GoRT)
    if numtrials is None:
        p_inhib = np.mean(SSD)
    else:
        p_inhib = np.mean(SSD[-numtrials:])
    return median-p_inhib

def calc_cond_mean_std(data, cond, col):
    cond_idx = np.where(cond)[0]
    cond_data = data[cond_idx]
    cond_data_mean = np.mean(cond_data[col], axis=0)
    cond_data_median = np.median(cond_data[col], axis=0)
    cond_data_sem = sem(cond_data[col], axis=0)
    return (cond_data, cond_data_mean, cond_data_median, cond_data_sem)


class StopSignalBase(emergent.Base):
    def __init__(self, **kwargs):
	super(StopSignalBase, self).__init__(**kwargs)
	self.flag['task'] = 'STOP_SIGNAL'
        self.flag['DLPFC_speed_mean'] = 0.01
        self.flag['DLPFC_speed_std'] = 0
	self.SSRT = {}
	self.GoRT = {}
	self.GoRTcode = {}
	self.data_settled = {}

	self.b_data = {}
	self.resp_data = {}
	self.resp_noss_data = {}
	self.GoRT = {}
        self.RT = {}
        self.response_prob = {}
	self.SSD = {}
	self.resp_ss_data = {}
	self.resp_go_data = {}

	self.pt_code = {0: 'GoTrial_resp',
			1: 'GoTrial_noresp',
			2: 'SS_inhib',
			3: 'SS_resp'}

    def _preprocess_data(self, data, tag):
        self.SSRT[tag] = []

        uniq_batches = np.unique(data['batch'])

        # initialize data containers
        self.b_data[tag] = []
        self.resp_data[tag] = []
        self.resp_noss_data[tag] = []
        self.GoRT[tag] = []
        self.RT[tag] = []
        self.SSD[tag] = []
        self.data_settled[tag] = []
        self.response_prob[tag] = []

        for b,batch in enumerate(uniq_batches):
            # Make list with individual batches
            b_idx = data['batch'] == batch
            self.b_data[tag].append(data[b_idx][50:])
            self.data_settled[tag].append(data[b_idx][20:])

            # Slice out trials in which a response was made
            resp_idx = self.b_data[tag][-1]['inhibited'] == 0
            self.resp_data[tag].append(self.b_data[tag][-1][resp_idx])

            self.response_prob[tag].append(np.sum((self.data_settled[tag][-1]['inhibited'] == 0) &
                                                   (self.data_settled[tag][-1]['SS_presented'] == 1)) /
                                           np.sum((self.data_settled[tag][-1]['SS_presented'] == 1)))
            # Slice out trials in which a response was made and no SS was presented
            resp_noss_idx = self.resp_data[tag][-1]['SS_presented'] == 0
            self.resp_noss_data[tag].append(self.resp_data[tag][-1][resp_noss_idx])

            self.GoRT[tag].append(self.resp_noss_data[tag][-1]['minus_cycles'])
            self.RT[tag].append(self.resp_data[tag][-1]['minus_cycles'])
            self.SSD[tag].append(self.data_settled[tag][-1]['SSD'])
            self.SSRT[tag].append(calc_SSRT(self.GoRT[tag][-1], self.SSD[tag][-1]))

        # Analyze SSRTs for SS and Go_resp
        resp_ss_idx = (data['SS_presented'] == 1) & (data['inhibited'] == 0.0)
        self.resp_ss_data[tag] = data[resp_ss_idx]

        resp_go_idx = (data['SS_presented'] == 0) & (data['inhibited'] == 0.0)
        self.resp_go_data[tag] = data[resp_go_idx]

        # Convert list data_settled to continous array
        self.data_settled[tag] = np.concatenate(self.data_settled[tag])

    def preprocess_data(self):
	for t,tag in enumerate(self.tags):
	    self._preprocess_data(self.data[tag], tag)
            
    def go_ddm(self, plot=True):
	"""Fit and plot DDM to data of Go-trials"""
        return
	for t,tag in enumerate(self.tags):
	    # Select Go trials
	    data = self.data[tag]
	    idx = data['SS_presented'] == False
	    self.fit_ddm_data(data[idx][''],
			      data[idx][''],
			      data[idx]['minus_cycles'],
			      plot=plot,
			      tag=tag,
			      color=t)
	    
	    
    def plot_staircase(self):
	for t,tag in enumerate(self.tags):
	    # Plot staircase
	    for b_idx, b_data in enumerate(self.b_data[tag]):
		if b_idx == 0: # If first, add label
		    plt.plot(b_data['SSD'], self.colors[t], label=self.names[t])
		else:
		    break
		    plt.plot(b_data['SSD'], self.colors[t])
		plt.title('Staircases')
		plt.xlabel('Trials')
		plt.ylabel('SSD')
		leg = plt.legend(loc='best', fancybox=True)
		leg.get_frame().set_alpha(.5)
		plt.show()

    def plot_seq_effects(self):
	for t,tag in enumerate(self.tags):
	    # Analyze each individual trial code (i.e. what was the previous trial?)
	    data_mean, data_sem = emergent.group_batch(self.data_settled[tag], ['prev_trial_code', 'inhibited', 'SS_presented'])
	    # Select those where a response was made and no stop signal was presented
	    idx = (data_mean['inhibited'] == 0.0) & (data_mean['SS_presented'] == 0.0)
	    plt.errorbar(data_mean[idx]['prev_trial_code'], data_mean[idx]['minus_cycles'], color=self.colors[t], yerr=data_sem[idx]['minus_cycles'], label=self.names[t])
	    plt.title('RTs depending on previous trial')
	    plt.xticks(np.arange(len(self.pt_code.values())), self.pt_code.values())
	    plt.ylabel('RTs')
	    plt.xlabel('Previous Trial Type')
	    plt.legend(loc=2)
	    plt.xlim((-.5, 3.5))
	    #plt.ylim((60,180))

    def plot_GoRespVsInhibResp(self):
	for t,tag in enumerate(self.tags):
	    data = self.data_settled[tag]
	    # Analyze RTs in successfully inhibited trials and go trials
	    (ss_data, ss_data_mean, ss_data_median, ss_data_std) = calc_cond_mean_std(data, (data['SS_presented'] == 1) & (data['inhibited'] == 0.0), 'minus_cycles')
	    (go_data, go_data_mean, go_data_median, go_data_std) = calc_cond_mean_std(data, (data['SS_presented'] == 0) & (data['inhibited'] == 0.0), 'minus_cycles')
	    # Plot
	    plt.errorbar([0, 1], [ss_data_median, go_data_median], yerr=[ss_data_std, go_data_std], color=self.colors[t], label=self.names[t])
	    plt.title('Median RTs for Go_resp and SS_resp trials')
	    plt.xticks((0,1), ('SS_resp', 'Go_resp'))
	    plt.ylabel('Median RT')
	    plt.xlim((-.5,1.5))
	    plt.ylim((60,120))
	    plt.legend(loc=2)


    def plot_GoRTs(self):
	i=1
	for t,tag in enumerate(self.tags):
	    data = self.data_settled[tag]

	    plt.subplot(len(self.tags),3,i)
	    plt.hist(self.resp_ss_data[tag]['minus_cycles'], bins=100, range=(0,200))
	    plt.title('RTs for StopSignal trials')
	    plt.xlabel('RT')
	    plt.subplot(len(self.tags),3,i+1)
	    plt.hist(self.resp_ss_data[tag]['SSD'], bins=100, range=(0,200))
	    plt.title('SSDs in StopSignal trials')
	    plt.xlabel('SSD')
	    
	    plt.subplot(len(self.tags),3,i+2)
	    plt.hist(self.resp_go_data[tag]['minus_cycles'], bins=100, range=(0,200))
	    plt.title('RTs for Go trials')
	    plt.xlabel('RT')
	    i+=3

    def plot_3dhisto(self):
	for t,tag in enumerate(self.tags):
	    # Plot GoRT histogram
	    GoHist = []
	    for GoDist in self.GoRT[tag]:
		GoHist.append(np.histogram(GoDist, bins=75, range=(0,200))[0])
	    ml.figure(t)
	    chart = ml.barchart(GoHist)

#@pools.register_group(['atomoxetine'])
class Atomoxetine(StopSignalBase):
    def __init__(self, dec_array=(), mag_array=(), **kwargs):
	"""Test effects of the norepinephrine (NE) reuptake inhibitor Atomoxetine on StopSignal performance

	dec_array: array of parameters to check for how fast the NE burst is goes back to tonic
	mag_array: array of parameters to check for how big an NE burst is."""
	
	super(Atomoxetine, self).__init__(**kwargs)

	if len(dec_array) == 0:
	    dec_array = np.linspace(0,0.1,5)
	if len(mag_array) == 0:
	    mag_array = np.linspace(0,1,5)
	    
	self.flag['atomoxetine'] = True
	self.flag['staircase_mode'] = True
	self.flag['SS_prob'] = .25
	
	self.tags = []

	self.dec_array = dec_array
	self.mag_array = mag_array
	
	for dec in self.dec_array:
	    for mag in self.mag_array:
		self.flag['atomoxetine_dec'] = dec
		self.flag['atomoxetine_mag'] = mag
		self.tags.append(str(dec) + '_' + str(mag))
		self.flag['tag'] = '_' + self.tags[-1]
		self.flags.append(copy(self.flag))

    def analyze(self):
	atomox_SSRT = np.empty((len(self.dec_array), len(self.mag_array)))

	# Create array with SSRTs
	for i,dec in enumerate(self.dec_array):
	    for j,mag in enumerate(self.mag_array):
		atomox_SSRT[i,j] = np.mean(self.SSRT[str(dec) + '_' + str(mag)])
		#data = self.data[str(dec) + '_' + str(mag)]
		#data_gp, data_gp_idx = emergent.group_batch(data, self.data_idx, ['SS_presented'])
		#idx = np.where(data_gp[:,data_gp_idx['SS_presented']] == 1)
		#atomox_SSRT[i,j] = data_gp[idx, data_gp_idx['SSD_mean_mean']]

	# Create meshgrid for displaying
	X, Y = np.meshgrid(self.dec_array, self.mag_array)

	self.new_fig()
	plt.contourf(X, Y, atomox_SSRT)
	plt.title('Atomoxetine effects on StopSignal Reaction Time (SSRT)')
	plt.xlabel('NE burst decrease speed')
	plt.ylabel('NE burst magnitude')
	plt.xlim((0,0.1))
	plt.ylim((0,1))
	plt.colorbar()

	self.save_plot("2D-SSRT")

@pools.register_group(['stopsignal'])
class Salience(StopSignalBase):
    def __init__(self, detection_probs=None, **kwargs):
	super(Salience, self).__init__(**kwargs)
	self.tags = []

	if detection_probs is None:
	    self.detection_probs = np.linspace(0,1,11)
	else:
	    self.detection_probs = detection_probs

	self.flag['staircase_mode'] = True
	self.flag['SS_prob'] = .25

	for detection_prob in self.detection_probs:
	    self.flag['salience'] = detection_prob
	    tag = 'salience_' + str(detection_prob)
	    self.tags.append(tag)
	    self.flag['tag'] = '_' + tag
	    self.flags.append(copy(self.flag))

    def analyze(self):
	self.new_fig()
	SSRT_mean = [np.mean(self.SSRT[tag]) for tag in self.tags]
	SSRT_sem = [sem(self.SSRT[tag]) for tag in self.tags]
	
	plt.errorbar(self.detection_probs, SSRT_mean, yerr=SSRT_sem)
	plt.title('Salience detection influence on SSRTs')
	plt.xlabel('Stop-Signal detection probability')
	plt.ylabel('SSRT')
	self.save_plot('SSRT')
	    
	    
@pools.register_group(['stopsignal'])
class StopSignal_IFGlesion(StopSignalBase):
    def __init__(self, IFG_lesions=(0.,0.5,0.75,1), **kwargs):
	super(StopSignal_IFGlesion, self).__init__(**kwargs)
	self.tags = []
	self.IFGs = []
	self.names = []
	
	self.flag['staircase_mode'] = True
	self.flag['SS_prob'] = .25
	
	for IFG_lesion in IFG_lesions:
	    self.flag['IFG_lesion'] = IFG_lesion
	    tag = 'IFG_' + str(IFG_lesion)
	    self.tags.append(tag)
	    self.names.append(str(int(IFG_lesion*100)) + '%' + ' IFG lesion')
	    self.IFGs.append(IFG_lesion)
	    self.flag['tag'] = '_' + tag
	    self.flags.append(copy(self.flag))

    def analyze(self):
	self.new_fig()
	self.plot_staircase()
	self.save_plot("Staircase")
	
	self.new_fig()
	self.plot_GoRespVsInhibResp()
	self.save_plot("RTs_Go_vs_inhib")
	
	self.new_fig()
	self.plot_GoRTs()
	self.save_plot("GoRT_histo")
	
	self.new_fig()
	self.plot_seq_effects()
	self.save_plot("Seq_effects")

	return

    def plot_seq_effects_sel(self):
	self.new_fig()
	self.plot_seq_go_RT()
	self.save_plot("Seq_Go_RT")
	
	self.new_fig()
	self.plot_seq_stop_RT()
	self.save_plot("Seq_Stop_RT")
	
	self.new_fig()
	self.plot_seq_stop_inhib()
	self.save_plot("Seq_Stop_SS_inhib")
	
    def plot_seq_go_RT(self):
	for i,tag in enumerate(self.tags):
	    # Group data accordingly
	    data_mean, data_sem = emergent.group_batch(self.data[tag], ['stimulus_changed', 'SS_presented', 'inhibited', 'prev_trial_code'])

	    # Select subsets
	    # Trials that follow Go trials and the stimuli match
	    match_go_idx = (data_mean['stimulus_changed'] == 0) & \
			   (data_mean['SS_presented'] == 0) & \
			   (data_mean['prev_trial_code'] <= 1) & \
			   (data_mean['inhibited'] == 0)

	    data_match_go = data_mean[data_match_go_idx]

	    # Trials that follow Go trials and the stimuli don't match
	    nomatch_go_idx = (data['stimulus_changed'] == 1) & \
			     (data['SS_presented'] == 0) & \
			     (data['prev_trial_code'] <= 1) & \
			     (data['inhibited'] == 0)
	    
	    data_nomatch_go = data_mean[data_nomatch_go_idx, :]

	    # TODO: check for bug.
	    plt.errorbar([0,1], (np.mean(data_mean[match_go_idx]['minus_cycles'], 0),
				np.mean(data_mean[nomatch_go_idx]['minus_cycles'],0)),
			yerr=(data_sem[match_go_idx]['minus_cycles']),
			label=self.names[i])

	plt.xticks([0,1], ('No match', 'Match'))
	plt.xlim((-.25, 1.25))
	plt.title('Sequential effects of stimulus matching after Go trials')
	plt.ylabel('Mean RTs (Cycles)')
	plt.legend(loc=0, fancybox=True)
	    
    def plot_seq_stop_RT(self):
	for i,tag in enumerate(self.tags):
	    # Group data accordingly
	    data_gp, data_gp_idx = emergent.group(self.data[tag], self.data_idx, ['batch_num', 'stimulus_changed', 'SS_presented', 'inhibited', 'prev_trial_code'])
	    data, data_idx = emergent.group(data_gp, data_gp_idx, ['stimulus_changed', 'SS_presented', 'inhibited', 'prev_trial_code'])
	

	    
	    # Trials that follow Stop trials and the stimuli match
	    data_match_stop_idx = np.where((data[:, data_idx['stimulus_changed']] == 0) &
					   (data[:, data_idx['SS_presented']] == 0) &
					   (data[:, data_idx['prev_trial_code']] > 1) &
					   (data[:, data_idx['inhibited']] == 0))[0]
	    data_match_stop = data[data_match_stop_idx, :]

	    # Trials that follow Stop trials and the stimuli don't match
	    data_nomatch_stop_idx = np.where((data[:, data_idx['stimulus_changed']] == 1) &
					     (data[:, data_idx['SS_presented']] == 0) &
					     (data[:, data_idx['prev_trial_code']] > 1) &
					     (data[:, data_idx['inhibited']] == 0))[0]
	    data_nomatch_stop = data[data_nomatch_stop_idx, :]
	    
	    RT_mean_col = data_idx['minus_cycles_mean_mean']
	    RT_sem_col = data_idx['minus_cycles_mean_sem']
	    plt.errorbar([0,1], (np.mean(data_nomatch_stop[:, RT_mean_col],0), np.mean(data_match_stop[:, RT_mean_col],0)),
		    yerr=(data_nomatch_stop[0, RT_sem_col]), label=self.names[i])

	plt.xticks([0,1], ('No match', 'Match'))
	plt.xlim((-.25, 1.25))
	plt.title('Sequential effects of stimulus matching after Stop Signals')
	plt.ylabel('Mean RTs (cycles)')
	plt.legend(loc=0, fancybox=True)


    def plot_seq_stop_inhib(self):
	for i,tag in enumerate(self.tags):
	    # Group data accordingly
	    (data_gp, data_gp_idx) = emergent.group(self.data[tag], self.data_idx, ['batch_num', 'stimulus_changed', 'SS_presented', 'prev_trial_code'])
	    data, data_idx = emergent.group(data_gp, data_gp_idx, ['stimulus_changed', 'SS_presented', 'prev_trial_code'])
	

	    
	    # Trials that follow Stop trials and the stimuli match
	    data_match_stop_idx = np.where((data[:, data_idx['stimulus_changed']] == 0) &
					   (data[:, data_idx['SS_presented']] == 1) &
					   (data[:, data_idx['prev_trial_code']] > 1))[0]

	    data_match_stop = data[data_match_stop_idx, :]

	    # Trials that follow Stop trials and the stimuli don't match
	    data_nomatch_stop_idx = np.where((data[:, data_idx['stimulus_changed']] == 1) &
					     (data[:, data_idx['SS_presented']] == 1) &
					     (data[:, data_idx['prev_trial_code']] > 1))[0]
	    data_nomatch_stop = data[data_nomatch_stop_idx, :]
	    
	    RT_mean_col = data_idx['inhibited_mean_mean']
	    RT_sem_col = data_idx['inhibited_mean_sem']
	    plt.errorbar([0,1], (np.mean(data_nomatch_stop[:, RT_mean_col],0), np.mean(data_match_stop[:, RT_mean_col],0)),
		    yerr=(data_nomatch_stop[0, RT_sem_col]), label=self.names[i])

	plt.xticks([0,1], ('No match', 'Match'))
	plt.xlim((-.25, 1.25))
	plt.title('Sequential effects of stimulus matching on stop signal inhibition probability')
	plt.ylabel('SS_inhibit (probability)')
	plt.legend(loc=0, fancybox=True)

@pools.register_group(['stopsignal', 'motivation'])
class MotivationalEffects(StopSignalBase):
    def __init__(self, **kwargs):
	super(MotivationalEffects, self).__init__(**kwargs)
	#self.tags = ['NO_BIAS', 'ACC_BIAS', 'SPEED_BIAS']
	self.tags = ['SPEED_BIAS', 'ACC_BIAS']
	self.names = ['Speed', 'Accuracy']
	

	self.flag['staircase_mode'] = True
	self.flag['SS_prob'] = .25
	for tag in self.tags:
	    self.flag['tag'] = '_' + tag
	    self.flag['motivational_bias'] = tag
	    self.flags.append(copy(self.flag))

    def analyze(self):
	plt.subplot(121)#, projection='frameaxes')
	self.plot_motivation_GoRT()
	plt.subplot(122)#, projection='frameaxes')
	self.plot_motivation_SSRT()
	self.save_plot("RT_and_SSRT")
	
	self.new_fig()
	self.plot_motivation_SSRT()
	self.save_plot("SSRT")
				   
	

    def plot_motivation_GoRT(self):
	for i,tag in enumerate(self.tags):
	    data_mean, data_sem = emergent.group(self.data[tag], ['SS_presented', 'inhibited'])

	    noss_resp = (data_mean['SS_presented'] == 0) & \
			(data_mean['inhibited'] == 0)
				 
	    plt.bar(i, data_mean[noss_resp]['minus_cycles'],
		   yerr = data_sem[noss_resp]['minus_cycles'],
		   color=self.colors[i], ecolor='k', width=0.8)

	plt.title('Motivational influences on GoRT')
	plt.ylabel('Response Time (Cycles)')
	plt.xticks(np.array(range(len(self.tags)))+.5, self.names)

    def plot_motivation_SS_inhib(self):
	for i,tag in enumerate(self.tags):
	    data_mean, data_sem = emergent.group(self.data[tag], ['SS_presented'])

	    ss_pres = data_mean['SS_presented'] == 1

	    plt.bar(i, data_mean[ss_pres]['inhibited'],
		   yerr = data_sem[ss_pres]['inhibited'],
		   color=self.colors[i], ecolor='k', width=0.8)

	plt.title('Motivational influences on Stop accuracy')
	plt.ylabel('Stop Response Rate')
	plt.xticks(np.array(range(len(self.tags)))+.5, self.names)

    def plot_motivation_SSRT(self):
	for i,tag in enumerate(self.tags):
	    plt.bar(i, np.mean(self.SSRT[tag]), yerr = sem(self.SSRT[tag]),
		   color=self.colors[i], ecolor='k', width=0.8)

	plt.title('Motivational influences on SSRTs')
	plt.ylabel('SSRT')
	plt.xticks(np.array(range(len(self.tags)))+.5, self.names)

@pools.register_group(['stopsignal', 'ifg_lesion'])
class IFGLesion(StopSignalBase):
    def __init__(self, IFG_lesions=(0,0.25,0.5,0.75,1.), **kwargs):
	super(IFGLesion, self).__init__(**kwargs)
	self.flag['test_SSD_mode'] = True
        self.flag['SSD_start'] = 0
        self.flag['SSD_stop'] = 70
	#self.flag['SS_prob'] = 1.
	self.tags = []
	self.names = []
	self.IFG_lesions = IFG_lesions

	for IFG_lesion in self.IFG_lesions:
	    self.flag['IFG_lesion'] = IFG_lesion
	    tag = 'IFG_' + str(IFG_lesion)
	    self.names.append('%i IFG lesion'%(int(IFG_lesion*100)))
	    self.tags.append(tag)
	    self.flag['tag'] = '_' + tag
	    self.flags.append(copy(self.flag))

    def analyze(self):
	self.new_fig()
        debug_here()
	for i,lesion in enumerate(self.IFG_lesions):
            tag = 'IFG_' + str(lesion)
	    data_mean, data_sem = emergent.group_batch(self.data[tag], ['SSD', 'SS_presented'])
            idx = data_mean['SS_presented'] == 1
	    plt.errorbar(data_mean['SSD'][idx],
			data_mean['inhibited'][idx],
			yerr = data_sem['inhibited'][idx],
			#color=colors[i],
			label=self.names[i], lw=self.lw)

	plt.title('IFG lesion effects on response inhibition')
	plt.xlabel('SSD')
	plt.ylabel('P(inhib|signal)')
	plt.legend(loc=0, fancybox=True)

	self.save_plot("SSD_VS_SS_inhib")

class Crit_VS_Noncrit_stop(StopSignalBase):
    def __init__(self, **kwargs):
	super(Crit_VS_Noncrit_stop, self).__init__(**kwargs)
	self.tags = ['critical']

	self.flag['tag'] = '_' + self.tags[0]
	
	# Run critical direction
	self.flag['critical'] = True
	self.flag['staircase_mode'] = True
	self.flag['SS_prob'] = .25
	self.flags.append(copy(self.flag))

    def analyze(self):
	self.new_fig()
	# TODO Check if this makes sense, why not use group_batch?
	data_mean, data_sem = emergent.group_batch(self.data['critical'], ['SS_presented', 'trial_name', 'prev_trial_code'])

	# Cut out critical direction trials
	data_crit_idx = np.where((data[:,data_idx['SS_presented']] == 1.0) & # Stop-Signal presented and
				 (data[:,data_idx['trial_name']] == 1.0) &   # Non-critical direction and
				 (data[:,data_idx['prev_trial_code']] == 0))[0] # GoTrail repsonse
	
	data_go_idx = np.where((data[:,data_idx['SS_presented']] == 0.0) & # No Stop-Signal presented and
			       (data[:,data_idx['prev_trial_code']] == 0))[0] # GoTrial response
				
	data_crit = data[data_crit_idx, :]
	data_go = data[data_go_idx, :]

	RT_col = data_idx['minus_cycles_mean']
	means = [np.mean(data_crit[:, RT_col]), np.mean(data_go[:, RT_col])]

	sems = [sem(data_crit[:, RT_col]), sem(data_go[:, RT_col])]
		 
	plt.errorbar([0,1], means, yerr=sems)
	plt.title('Stop-Signal during non-critical direction trials')
	plt.xlabel('Trial type')
	plt.ylabel('GoRT')
	plt.xlim([-.5,1.5])
	plt.xticks([0,1], ['Non-critical direction + Stop-Signal', 'Regular Go-Trial'])
	
	self.save_plot("RTs")

@pools.register_group(['stopsignal', 'cycle'])
class StopSignal_cycle(emergent.BaseCycle, StopSignalBase):
    def __init__(self, **kwargs):
	super(StopSignal_cycle, self).__init__(**kwargs)

	self.SSRT = {}
	self.GoRT = {}
        self.RT = {}
	self.GoRTcode = {}
	self.data_settled = {}
        self.response_prob = {}

	self.b_data = {}
	self.resp_data = {}
	self.resp_noss_data = {}
	self.GoRT = {}
	self.SSD = {}
	self.resp_ss_data = {}
	self.resp_go_data = {}

        self.SSD_set = 50
        self.SC_thr = .80

	self.tags = ['intact'] #, 'fixed_SSD']
        self.flag['task'] = 'STOP_SIGNAL'
        self.flag['rnd_seed'] = 'OLD_SEED'
        #self.flag['max_epoch'] = 200
	self.flag['tag'] = '_' + self.tags[0]
	self.flag['staircase_mode'] = True
        self.flag['test_SSD_mode'] = False
        self.flag['SS_prob'] = .25
	self.flags.append(copy(self.flag))

	#self.flag['tag'] = '_' + self.tags[1]
	#self.flag['staircase_mode'] = False
        #self.flag['test_SSD_mode'] = True
        #self.flag['SSD_start'] = self.SSD_set
        #self.flag['SSD_stop'] = self.SSD_set
        #self.flag['SS_prob'] = 0.
	#self.flags.append(copy(self.flag))

    def preprocess_data(self):
	for t,tag in enumerate(self.tags):
            self._preprocess_data(self.data['trl'][tag], tag)
            
    def analyze(self):
        self.new_fig()
        self.analyze_STN_act(tag='intact')
        self.save_plot("STN_act")
        
        self.new_fig()
        self.analyze_SC_act_avg(tag='intact')
        self.save_plot("SC_act_intact")

        #self.analyze_SC_act_ind(tag='fixed_SSD', SSDs=(self.SSD_set,))
        #self.save_plot("SC_act_ind_fixed")

    def analyze_STN_act(self, tag=None):
        if tag is None:
            tag = 'intact'
	wind=(50,50)
	STN_ss_resp = self.extract_cycles(
	    tag, 
	    ((self.data['trl'][tag]['SS_presented'] == 1) &
	     (self.data['trl'][tag]['inhibited'] == 0)),
	    'STN_acts_avg',
	    center='SSD',
	    wind=wind)
	
	STN_ss_inhib = self.extract_cycles(
	    tag,
	    ((self.data['trl'][tag]['SS_presented'] == 1) &
	     (self.data['trl'][tag]['inhibited'] == 1)),
	    'STN_acts_avg',
	    center='SSD',
	    wind=wind)

	x=np.linspace(-wind[0],wind[1],np.sum(wind)+1)
        self.plot_filled(x, STN_ss_inhib, label='SS_inhib', color='g')
        self.plot_filled(x, STN_ss_resp, label='SS_resp', color='r')
#	plt.plot(x, np.mean(STN_ss_Go, axis=0))
	plt.xlabel('Cycles around Stop-Signal')
	plt.ylabel('Average STN activity')
	plt.title('STN activity in SS_inhib and SS_resp trials: %s'%tag)
	plt.legend(loc=0, fancybox=True)

    def analyze_SC_act_avg(self, tag=None):
        if tag is None:
            tag = 'intact'
        start_cycle = 25
	wind = (0,100)
        #wind = (100,100)
        # From emergent, SC threshold:
        data_grp_mean, data_grp_sem = emergent.group_batch(self.data['trl'][tag], ['SS_presented'])

        idx = data_grp_mean['SS_presented'] == 1
        SSD = self.data_settled[tag]['SSD'].mean()

        thalam_ss_resp = self.extract_cycles(
            tag,
            ((self.data['trl'][tag]['SS_presented'] == 1) &
             (self.data['trl'][tag]['inhibited'] == 0) &
             (self.data['trl'][tag]['epoch'] > 20)),
            'Thalam_unit_corr', cycle=start_cycle,
            #center='SSD',
            wind=wind)

        thalam_ss_inhib = self.extract_cycles(
            tag,
            ((self.data['trl'][tag]['SS_presented'] == 1) &
             (self.data['trl'][tag]['inhibited'] == 1) &
             (self.data['trl'][tag]['epoch'] > 20)),
            'Thalam_unit_corr', cycle=start_cycle,
            #center='SSD',
            wind=wind)

        x=np.linspace(wind[0]+start_cycle,wind[1]+start_cycle,np.sum(wind)+1)

        #thr_cross = np.where(np.mean(thalam_ss_resp, axis=0) > thr)[0][0]
        self.plot_filled(x, thalam_ss_inhib, label='SS_inhib', color='g')
        self.plot_filled(x, thalam_ss_resp, label='SS_resp', color='r')

        plt.axhline(y=self.SC_thr, color='k')
        #plt.axvline(x=thr_cross, color='k')
        #plt.axvline(x=np.mean(self.SSD['intact'])+np.mean(self.SSRT['intact']), color='k')
        #plt.axvline(x=np.mean(self.SSD['intact']), color='k')
        plt.axvline(x=SSD, color='k')
        plt.axvline(x=SSD + np.mean(self.SSRT['intact']), color='k')

        plt.xlabel('Cycles')
        plt.ylabel('Average SC activity')
        plt.title('SC activity during inhibited and not-inhibited stop trials: %s'%tag)
        plt.legend(loc=0)

    def analyze_SC_act_ind(self, SSDs=None, tag=None, plot_ind=False):
        if tag is None:
            tag = 'intact'
	wind = (0,100)
        start_cycle = 25
        skip_epochs = 20
        if SSDs is None:
            SSDs = np.unique(self.data['trl']['intact']['SSD'])
        for SSD in SSDs:
            data_grp_mean, data_grp_sem = emergent.group_batch(self.data['trl'][tag], ['SS_presented'])

            idx = data_grp_mean['SS_presented'] == 1
            SSD_mean = data_grp_mean[idx]['SSD']

            # Select responded and inhibited trials
            resp = ((self.data['trl'][tag]['inhibited'] == 0) &
                       (self.data['trl'][tag]['SSD'] == SSD) &
                       (self.data['trl'][tag]['epoch'] > skip_epochs))
            ss_resp = ((self.data['trl'][tag]['SS_presented'] == 1) &
                       (self.data['trl'][tag]['inhibited'] == 0) &
                       (self.data['trl'][tag]['SSD'] == SSD) &
                       (self.data['trl'][tag]['epoch'] > skip_epochs))
            ss_inhib = ((self.data['trl'][tag]['SS_presented'] == 1) &
                        (self.data['trl'][tag]['inhibited'] == 1) &
                        (self.data['trl'][tag]['SSD'] == SSD) &
                        (self.data['trl'][tag]['epoch'] > skip_epochs))
            
            # Calculate proportion of inhibited vs error trials
            mean_ss_resp = (np.sum(ss_inhib)/np.sum(((self.data['trl'][tag]['SS_presented'] == 1) &
                                                                         (self.data['trl'][tag]['SSD'] == SSD) &
                                                                         (self.data['trl'][tag]['epoch'] > skip_epochs))))
            
            print "Mean responded trials: %f" % mean_ss_resp

            if mean_ss_resp == 0. or mean_ss_resp == 1.:
                continue # No need to plot SSDs to which no or all responses where inhibited.

            self.new_fig()
            thalam_ss_resp = self.extract_cycles(tag, resp, 'Thalam_unit_corr', cycle=start_cycle, wind=wind)

            thalam_ss_inhib = self.extract_cycles(tag, ss_inhib, 'Thalam_unit_corr', cycle=start_cycle, wind=wind)

            x=np.linspace(wind[0]+start_cycle,wind[1]+start_cycle,np.sum(wind)+1)

            #thr_cross = np.where(np.mean(thalam_ss_resp, axis=0) > thr)[0][0]
            self.plot_filled(x, thalam_ss_inhib, label='SS_inhib', color='g')
            self.plot_filled(x, thalam_ss_resp, label='SS_resp', color='r')

            plt.axhline(y=self.SC_thr, color='k')
            plt.axvline(x=SSD, color='k')
            plt.axvline(x=SSD + np.mean(self.SSRT['intact']), color='k')

            plt.xlabel('Stop-Signal')
            plt.ylabel('Average SC activity')
            plt.title('SC activity during inhibited and not-inhibited stop trials\n: %s, SSD: %i, mean response rate: %f'%(tag, SSD, mean_ss_resp))
            plt.legend(loc=0)

            if plot_ind:
                self.new_fig()
                self.plot_filled(x, thalam_ss_inhib, avg=False, label='SS_inhib', color='g')
                self.plot_filled(x, thalam_ss_resp, avg=False, label='SS_resp', color='r')

                plt.axhline(y=self.SC_thr, color='k')
                plt.axvline(x=SSD, color='k')
                plt.axvline(x=SSD + np.mean(self.SSRT['intact']), color='k')

                plt.xlabel('Stop-Signal')
                plt.ylabel('Average SC activity')
                plt.title('SC activity during inhibited and not-inhibited stop trials\n: %s, SSD: %i, mean response rate: %f'%(tag, SSD, mean_ss_resp))



@pools.register_group(['stopsignal', 'DDM'])
class StopSignalDDM(StopSignalBase):
    def __init__(self, **kwargs):
	super(StopSignalDDM, self).__init__(**kwargs)
	self.tags = ['intact']
	self.flag['tag'] = '_' + self.tags[0]
	self.flag['staircase_mode'] = True
	self.flags.append(copy(self.flag))

    def preprocess_data(self):
	# Construct data structure for ddm fits.

	self.stimulus = {}
	self.response = {}
	self.rt = {}

	for tag in self.tags:
            data = self.data[tag]
            # Go trials following go trials
	    idx_post_go = ((data['SS_presented'] == 0) & (data['inhibited'] == 0) & (data['prev_trial_code'] <= 1))
            data_post_go = data[idx_post_go]
            # Go trials following stop-signal trials
	    idx_post_stop = ((data['SS_presented'] == 0) & (data['inhibited'] == 0) & (data['prev_trial_code'] > 1))
            data_post_stop = data[idx_post_stop]

	    self.stimulus[tag] = np.copy(data['trial_name'])
	    self.response[tag] = np.copy(data['error'])
	    self.rt[tag] = np.copy(data['minus_cycles'])/150.
	    idx = self.stimulus[tag] == '"Antisaccade"'
	    # Invert errors of antisaccade trials, that way,
	    # successful antisaccades are 1, while errors
	    # (i.e. prosaccades) are 0
	    self.response[tag][idx] = 1 - self.response[tag][idx]
