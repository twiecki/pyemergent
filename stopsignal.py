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
    def debug_here(): pass

import pools
import emergent
from emergent import sem

def calc_SSRT(GoRT, SSD, numtrials=-150):
    """Calculate the SSRT for a give GoRT distribution (array) and a given staircase run,
    the 50% inhibitory interval is computed using numtrials last trials of staircase"""
    median_go = np.median(GoRT)
    mean_go = np.mean(GoRT)
    mean_ssd = np.mean(SSD[np.diff(SSD)!=0])
    mean_ssd = np.mean(SSD[numtrials:])
    return mean_go-mean_ssd

def calc_cond_mean_std(data, cond, col):
    cond_idx = np.where(cond)[0]
    cond_data = data[cond_idx]
    cond_data_mean = np.mean(cond_data[col], axis=0)
    cond_data_median = np.median(cond_data[col], axis=0)
    cond_data_sem = sem(cond_data[col], axis=0)
    return (cond_data, cond_data_mean, cond_data_median, cond_data_sem)


class StopSignalBase(emergent.Base):
    def __init__(self, intact=True, pretrial=False, SZ=False, PD=False, NE=False, STN=False, motivation=False, IFG=False, salience=False, **kwargs):
	super(StopSignalBase, self).__init__(**kwargs)
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

	self.pt_code = {0: 'Go trial',
			#1: 'GoTrial_noresp',
			1: 'SS inhib',
			2: 'SS resp'}

        self.flag['task'] = 'STOP_SIGNAL'
            
        self.flag['test_SSD_mode'] = 'false'
        self.flag['max_epoch'] = 200
	self.flag['SS_prob'] = .25

        self.tags = []

        if intact:
            self.flags.append(copy(self.flag))
            self.tags.append('intact')
            self.flags[-1]['LC_mode'] = 'phasic'
            self.flags[-1]['tag'] = '_' + self.tags[-1]

	if SZ:
            self.flags.append(copy(self.flag))
	    self.tags.append('Increased_tonic_DA')
	    self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['tonic_DA_intact'] = 0.032
	    self.flags[-1]['SZ_mode'] = 'true'

	if PD:
            self.flags.append(copy(self.flag))
            self.tags.append('Decreased_tonic_DA')
	    self.flags[-1]['tag'] = '_' + self.tags[-1]
	    self.flags[-1]['SZ_mode'] = 'false'
	    self.flags[-1]['tonic_DA_intact'] = 0.029

        # if NE:
        #     for tonic_NE in np.linspace(0,.5,6):
        #         self.flags.append(copy(self.flag))
        #         self.tags.append('Tonic_NE_%f'%tonic_NE)
        #         self.flags[-1]['tag'] = '_' + self.tags[-1]
        #         self.flags[-1]['LC_mode'] = 'tonic'
        #         self.flags[-1]['tonic_NE'] = tonic_NE

        if NE:
            self.flags.append(copy(self.flag))
            self.tags.append('Tonic_NE')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['LC_mode'] = 'tonic'
            self.flags[-1]['tonic_NE'] = 0.4

        if STN:
            self.flags.append(copy(self.flag))
            self.tags.append('DBS_on')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
	    self.flags[-1]['tonic_DA_intact'] = 0.03
	    self.flags[-1]['STN_lesion'] = .8

        if motivation:
            self.flags.append(copy(self.flag))
            self.tags.append('Speed')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['motivational_bias'] = 'SPEED_BIAS'

            self.flags.append(copy(self.flag))
            self.tags.append('Accuracy')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['motivational_bias'] = 'ACC_BIAS'

        if IFG:
            self.flags.append(copy(self.flag))
            self.tags.append('IFG_lesion')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['IFG_lesion'] = .5

        if salience:
            self.flags.append(copy(self.flag))
            self.tags.append('salience')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['salience'] = .75

    def _preprocess_data(self, data, tag, cutoff=-150):
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

            # Test if model meets criteria of 50%
            settled = data[b_idx][cutoff:]
            prob = np.sum((settled['inhibited'] == 0) &
                          (settled['SS_presented'] == 1)) / np.sum((settled['SS_presented'] == 1))

            #if prob < .45 or prob > 0.55:
            #    continue

            self.response_prob[tag].append(prob)

            self.data_settled[tag].append(data[b_idx][cutoff:])
            self.b_data[tag].append(data[b_idx][cutoff:])

            # Slice out trials in which a response was made
            resp_idx = self.b_data[tag][-1]['inhibited'] == 0
            self.resp_data[tag].append(self.b_data[tag][-1][resp_idx])

            # Slice out trials in which a response was made and no SS was presented
            resp_noss_idx = (self.resp_data[tag][-1]['SS_presented'] == 0) & (self.resp_data[tag][-1]['prev_trial_code'] == 0)
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

        print tag
        print self.response_prob[tag]

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
		    plt.plot(b_data['SSD'], color=plt.cm.prism(t), label=self.names[t])
		else:
		    #break
		    plt.plot(b_data['SSD'], color=plt.cm.prism(t))
		plt.title('Staircases')
		plt.xlabel('Trials')
		plt.ylabel('SSD')
		leg = plt.legend(loc='best', fancybox=True)
		leg.get_frame().set_alpha(.5)

    def plot_seq_effects(self):
	for t,tag in enumerate(self.tags):
	    # Analyze each individual trial code (i.e. what was the previous trial?)
	    data_mean, data_sem = emergent.group_batch(self.data_settled[tag], ['prev_trial_code', 'inhibited', 'SS_presented'])
	    # Select those where a response was made and no stop signal was presented
	    idx = (data_mean['inhibited'] == 0.0) & (data_mean['SS_presented'] == 0.0) & ((data_mean['prev_trial_code'] != 1) )
            #data_mean[idx]['prev_trial_code']
	    plt.errorbar([0,1,2], data_mean[idx]['minus_cycles'], color=self.colors[t], yerr=data_sem[idx]['minus_cycles'], label=self.names[t], lw=self.lw)
	    plt.title('RTs depending on previous trial')
	    plt.xticks(np.arange(len(self.pt_code.values())), self.pt_code.values())
	    plt.ylabel('RTs')
	    plt.xlabel('Previous Trial Type')
	    plt.legend(loc=0)
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

    def plot_SSDs(self):
        for t,tag in enumerate(self.tags):
            plt.bar(t-.5, np.mean([np.mean(subj) for subj in self.SSD[tag]]), 
                    yerr=sem([np.mean(subj) for subj in self.SSD[tag]]), color=self.colors[t], label=tag, ecolor='k')

        plt.xticks(range(len(self.tags)), self.tags) #np.linspace(0.5,len(self.tags),len(self.tags)-.5), self.tags)

        plt.title('SSD across conditions')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    def plot_SSRTs(self):
        for t,tag in enumerate(self.tags):
            plt.bar(t-.5, np.mean(self.SSRT[tag]), yerr=sem(self.SSRT[tag]), color=self.colors[t], label=tag, ecolor='k')

        plt.xticks(range(len(self.tags)), self.tags) #np.linspace(0.5,len(self.tags),len(self.tags)-.5), self.tags)

        plt.title('SSRT across conditions')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    def plot_GoRTs(self):
        for t,tag in enumerate(self.tags):
            plt.bar(t-.5, np.mean([np.mean(subj) for subj in self.GoRT[tag]]), 
                    yerr=sem([np.mean(subj) for subj in self.GoRT[tag]]), color=self.colors[t], label=tag, ecolor='k')

        plt.xticks(range(len(self.tags)), self.tags) #np.linspace(0.5,len(self.tags),len(self.tags)-.5), self.tags)

        plt.title('GoRT across conditions')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
    def plot_RT_dist(self):
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

    def plot_SSD_vs_inhib(self, i, tag):
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

    def plot_3dhisto(self):
	for t,tag in enumerate(self.tags):
	    # Plot GoRT histogram
	    GoHist = []
	    for GoDist in self.GoRT[tag]:
		GoHist.append(np.histogram(GoDist, bins=75, range=(0,200))[0])
	    ml.figure(t)
	    chart = ml.barchart(GoHist)

@pools.register_group(['stopsignal', 'staircase', 'all'])
class StopSignal(StopSignalBase):
    def __init__(self, **kwargs):
        super(StopSignal, self).__init__(intact=True, NE=True, STN=True, PD=True, motivation=True, IFG=True, **kwargs)

        self.names = self.tags
        
    def analyze(self):
        self.new_fig()
        self.plot_GoRTs()
        self.save_plot('GoRTs')

        self.new_fig()
        self.plot_SSRTs()
        self.save_plot('SSRTs')

        self.new_fig()
        self.plot_SSDs()
        self.save_plot('SSDs')

        self.new_fig()
        self.plot_staircase()
        self.save_plot('staircase')



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
        self.SC_thr = .85

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
        start_cycle = 0
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
             (self.data['trl'][tag]['epoch'] > 30)),
            'Thalam_unit_corr', cycle=start_cycle,
            #center='SSD',
            wind=wind)

        thalam_ss_inhib = self.extract_cycles(
            tag,
            ((self.data['trl'][tag]['SS_presented'] == 1) &
             (self.data['trl'][tag]['inhibited'] == 1) &
             (self.data['trl'][tag]['epoch'] > 30)),
            'Thalam_unit_corr', cycle=start_cycle,
            #center='SSD',
            wind=wind)

        x=np.linspace(wind[0]+start_cycle,wind[1]+start_cycle,np.sum(wind)+1)

        #thr_cross = np.where(np.mean(thalam_ss_resp, axis=0) > thr)[0][0]
        print 1
        self.plot_filled(x, thalam_ss_inhib, label='SS_inhib', color='g')
        print 2
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



#@pools.register_group(['stopsignal', 'DDM'])
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
