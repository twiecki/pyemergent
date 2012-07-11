from __future__ import division
import numpy as np
import pandas as pd

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
    def __init__(self, intact=True, pretrial=False, SZ=False, PD=False, NE=False, STN=False, motivation=False, IFG=False, salience=False, decay_ifg=0, num_trials=200, SS_prob=.25, test_ssd_mode=False, **kwargs):
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

        self.ms = 4.

	self.pt_code = {0: 'Go trial',
			#1: 'GoTrial_noresp',
			1: 'SS inhib',
			2: 'SS resp'}

        self.flag['task'] = 'STOP_SIGNAL'

        self.flag['test_SSD_mode'] = test_ssd_mode
        self.flag['max_epoch'] = num_trials
	self.flag['SS_prob'] = SS_prob
	self.flag['decay_ifg'] = decay_ifg

        self.tags = []
        self.names = []

        if intact:
            self.flags.append(copy(self.flag))
            self.tags.append('intact')
            self.flags[-1]['LC_mode'] = 'phasic'
            self.flags[-1]['tag'] = '_' + self.tags[-1]

	if SZ:
            self.names.append('$\uparrow$tonic\nDA\nact.')
            self.flags.append(copy(self.flag))
	    self.tags.append('Increased_tonic_DA')
	    self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['tonic_DA_SZ'] = 0.032
	    self.flags[-1]['SZ_mode'] = 'true'

	if PD:
            self.names.append('$\downarrow$tonic\nDA\nact.')
            self.flags.append(copy(self.flag))
            self.tags.append('Decreased_tonic_DA')
	    self.flags[-1]['tag'] = '_' + self.tags[-1]
	    self.flags[-1]['SZ_mode'] = 'true'
	    self.flags[-1]['tonic_DA_SZ'] = 0.029

        # if NE:
        #     for tonic_NE in np.linspace(0,.5,6):
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

        if motivation:
            self.names.append('$\uparrow$tonic\nrIFG\nact')
            self.flags.append(copy(self.flag))
            self.tags.append('Accuracy')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['motivational_bias'] = 'ACC_BIAS'

            self.names.append('$\uparrow$preSMA-\nstriatum\ncons')
            self.flags.append(copy(self.flag))
            self.tags.append('Speed')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['motivational_bias'] = 'SPEED_BIAS'


        if IFG:
            self.names.append('$\downarrow$IFG-\nSTN\ncons')
            self.flags.append(copy(self.flag))
            self.tags.append('IFG_lesion')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['IFG_lesion'] = .4

        if STN:
            self.names.append('$\downarrow$STN-\nSNr\ncons')
            self.flags.append(copy(self.flag))
            self.tags.append('DBS_on')
            self.flags[-1]['tag'] = '_' + self.tags[-1]
	    self.flags[-1]['tonic_DA_intact'] = 0.03
	    self.flags[-1]['STN_lesion'] = .5


        if salience:
            self.names.append('Reduced\nsaliency\ndetection')
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
            self.b_data[tag].append(data[b_idx])

            # Slice out trials in which a response was made
            resp_idx = self.data_settled[tag][-1]['inhibited'] == 0
            self.resp_data[tag].append(self.data_settled[tag][-1][resp_idx])

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

    def plot_RT_dist_SSD(self):
        for tag in self.tags:
            data = self.data[tag]
            ssds = np.unique(data['SSD'][data['SSD'] != -1])
            fig = plt.figure()
            ax = fig.add_subplot(len(ssds) + 1, 1, 1)
            max_rt = data[data['inhibited'] == False]['minus_cycles'].max() * self.ms*3
            go_rts = data[(data['SS_presented'] == False) & (data['inhibited'] == False)]['minus_cycles'] * self.ms*3
            ax.hist(go_rts, range=(0, max_rt), bins=100)
            for i, ssd in enumerate(ssds):
                go_rts = data[(data['SSD'] == ssd) & (data['inhibited'] == False)]['minus_cycles'] * self.ms*3
                if len(go_rts) == 0:
                    continue

                ax = fig.add_subplot(len(ssds) + 1, 1, i+2)
                ax.hist(go_rts, range=(0, max_rt), bins=100)
                ax.axvline(ssd * self.ms * 3)
            self.save_plot('RT_dist_SSD_%s' % tag)

    def plot_staircase(self):
	for t,(tag, color) in enumerate(zip(self.tags, ['k', '.7'])):
	    # Plot staircase
	    for b_idx, b_data in enumerate(self.b_data[tag]):
		if b_idx == 0: # If first, add label
		    plt.plot(b_data['SSD'], color=color, label=self.names[t])
		else:
		    #break
		    plt.plot(b_data['SSD'], color=color)
		#plt.title('Staircases')
		plt.xlabel('Trial')
		plt.ylabel('SSD (ms)')
		leg = plt.legend(loc='best', frameon=False)
		#leg.get_frame().set_alpha(.5)

    def plot_seq_effects(self):
	for t,tag in enumerate(self.tags):
	    # Analyze each individual trial code (i.e. what was the previous trial?)
	    data_mean, data_sem = emergent.group_batch(self.data_settled[tag], ['prev_trial_code', 'inhibited', 'SS_presented'])
	    # Select those where a response was made and no stop signal was presented
	    idx = (data_mean['inhibited'] == 0.0) & (data_mean['SS_presented'] == 0.0) & ((data_mean['prev_trial_code'] != 1) )
            #data_mean[idx]['prev_trial_code']
	    plt.errorbar([0,1,2], data_mean[idx]['minus_cycles']*self.ms, color=self.colors[t], yerr=data_sem[idx]['minus_cycles']*self.ms, label=tag, lw=3)
	    #plt.title('RTs depending on previous trial')
	    plt.xticks(np.arange(len(self.pt_code.values())), self.pt_code.values())
	    plt.ylabel('RTs (ms)')
	    plt.xlabel('Previous Trial Type')
	    plt.legend(loc=0)
	    plt.xlim((-.5, 2.5))
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
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.2)

        base = np.array([np.mean(subj) for subj in self.SSD['intact']])

        tags = self.tags[1:]

        for t,tag in enumerate(tags):
            plt.bar(t-.4, np.mean([np.mean(subj) for subj in self.SSD[tag]]-base)*self.ms,
                    yerr=sem([np.mean(subj) for subj in self.SSD[tag]])*self.ms, color='.7', label=tag, ecolor='k')

        plt.xticks(range(len(self.names)), self.names) #np.linspace(0.5,len(self.tags),len(self.tags)-.5), self.tags)

        plt.ylabel('SSD relative to intact (ms)')
        #plt.tick_params(labelsize='medium')
        plt.ylim(-20*self.ms,20*self.ms)

        ax = plt.gca()
        fontsize = 13
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)


    def plot_SSRTs(self):
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.2)

        base = np.array(self.SSRT['intact'])

        tags = self.tags[1:]

        for t,tag in enumerate(tags):
            diff_scores = self.SSRT[tag]-base
            plt.bar(t-.4, np.mean(diff_scores)*self.ms, yerr=sem(diff_scores)*self.ms, color='.7', label=tag, ecolor='k')

        plt.xticks(range(len(self.names)), self.names) #np.linspace(0.5,len(self.tags),len(self.tags)-.5), self.tags)
        plt.ylabel('SSRT relative to intact (ms)')
        plt.ylim(-20*self.ms,20*self.ms)
        #plt.tick_params(labelsize='medium')
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax = plt.gca()
        fontsize = 13
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)


    def plot_GoRTs(self):
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.2)

        base = np.array([np.mean(subj) for subj in self.GoRT['intact']])

        tags = self.tags[1:]
        for t,tag in enumerate(tags):
            diff_scores = [np.mean(subj) for subj in self.GoRT[tag]]-base
            plt.bar(t-.4, np.mean(diff_scores)*self.ms,
                    yerr=sem(diff_scores)*self.ms, color='.7', label=tag, ecolor='k')

        plt.xticks(range(len(self.names)), self.names) #np.linspace(0.5,len(self.tags),len(self.tags)-.5), self.tags)

        plt.ylabel('GoRT relative to intact (ms)')
        plt.ylim(-20*self.ms,20*self.ms)
        #plt.tick_params(labelsize='medium')
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax = plt.gca()
        fontsize = 13
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)

    def plot_cum_RT_dist(self):
        for t,tag in enumerate(['intact']):
            #plt.figure()

            SSD = np.median(self.data_settled[tag]['SSD'])

	    data_settled = self.data_settled[tag][self.data_settled[tag]['inhibited'] == 0]
	    data_ssd = data_settled # data_settled[data_settled['SSD'] == SSD]

            bins = 140
            upper = 150

            inhib_prob = np.mean(self.data[tag][self.data[tag]['SS_presented'] == 1]['inhibited'])
            x = np.linspace(0, upper, bins)
            cdf_go = np.cumsum(np.histogram(data_settled[data_settled['SS_presented'] == 0]['minus_cycles'], density=True, bins=bins, range=(0, upper))[0])

            cdf_go /= cdf_go[-1]

            cdf_ss = np.cumsum(np.histogram(data_ssd[data_ssd['SS_presented'] == 1]['minus_cycles'], density=True, bins=bins, range=(0, upper))[0])

            cdf_ss /= cdf_ss[-1]
            cdf_ss *= inhib_prob


            plt.plot(x*self.ms, cdf_go, color='k', label='Go trials')
            plt.plot(x*self.ms, cdf_ss, color='g', label='Stop trials')
            plt.legend(loc=0)
            plt.axhline(inhib_prob, color='b', linestyle='--')
            plt.axvline(np.mean(self.SSD[tag])*self.ms, color='r', linestyle='-')
            plt.axvline((np.mean(self.SSD[tag])+np.mean(self.SSRT[tag]))*self.ms, color='r', linestyle='--')
	    plt.xlabel('RT (ms)')
            plt.ylabel('Cumulative probability')
            plt.title('Model %s' % tag)


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

@pools.register_group(['stopsignal', 'all','nocycle'])
class StopSignal(StopSignalBase):
    def __init__(self, **kwargs):
        super(StopSignal, self).__init__(NE=True, STN=True, PD=True, motivation=True, IFG=True, SZ=True, **kwargs)
        #super(StopSignal, self).__init__(intact=True, NE=False, STN=False, PD=False, motivation=True, IFG=False, SZ=False, **kwargs)

        #self.names = self.tags

    def analyze(self):
        self.new_fig()
        self.plot_cum_RT_dist()
        self.save_plot('cum_RT')

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
        self.plot_seq_effects()
        self.save_plot('seq_effects')

        #self.new_fig()
        #self.plot_staircase()
        #self.save_plot('staircase')

@pools.register_group(['stopsignal', 'staircase'])
class StopSignalStaircase(StopSignalBase):
    def __init__(self, **kwargs):
        super(StopSignalStaircase, self).__init__(intact=True, NE=False, STN=False, PD=False, motivation=False, IFG=True, **kwargs)

        for flag in range(len(self.flags)):
            self.flags[flag]['max_epoch'] = 600
            self.flags[flag]['SS_prob'] = 0.1
            self.flags[flag]['SSD_start'] = 80

        self.names = ['intact', 'Reduced IFG-STN\nconnect.']


    def analyze(self):
        self.new_fig()
        self.plot_staircase()
        self.save_plot('staircase')

@pools.register_group(['stopsignal', 'cycle'])
class StopSignal_cycle(emergent.BaseCycle, StopSignalBase):
    def __init__(self, **kwargs):
	super(StopSignal_cycle, self).__init__(**kwargs)

        self.ms = 4

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


        for flag_id in range(len(self.flags)):
            self.flags[flag_id]['log_cycles'] = True

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
        self.analyze_act_SS_onset(name='STN')
        self.save_plot("STN_act")

        self.new_fig()
        self.analyze_act_SS_onset(name='ACC', field='ACC_act', wind=(20, 100))
        self.save_plot("ACC_act_SS_onset")

        self.new_fig()
        self.analyze_act_stim_onset(tag='intact')
        self.save_plot("SC_act_stim_onset")

        self.new_fig()
        self.analyze_act_stim_onset(tag='intact', name='ACC', field='ACC_act')
        self.save_plot("ACC_act_stim_onset")


        self.new_fig()
        self.analyze_act_post(nucleus='STN')
        self.save_plot('STN_post_ss')

        self.new_fig()
        self.analyze_act_post(nucleus='SC')
        self.save_plot('SC_post_ss')

        #self.analyze_SC_act_ind(tag='fixed_SSD', SSDs=(self.SSD_set,))
        #self.save_plot("SC_act_ind_fixed")

    def analyze_act_SS_onset(self, name='STN', field='STN_acts_avg', tag='intact', wind=(50, 50)):
	ss_resp = self.extract_cycles(
	    tag,
	    ((self.data['trl'][tag]['SS_presented'] == 1) &
	     (self.data['trl'][tag]['inhibited'] == 0)),
            field,
	    center='SSD',
	    wind=wind)

	ss_inhib = self.extract_cycles(
            tag,
	    ((self.data['trl'][tag]['SS_presented'] == 1) &
	     (self.data['trl'][tag]['inhibited'] == 1)),
            field,
	    center='SSD',
	    wind=wind)

	x=np.linspace(-wind[0],wind[1],np.sum(wind)+1)
        plt.plot(x, np.mean(ss_inhib, axis=0), label='SS_inhib', color='g', lw=3)
        plt.plot(x, np.mean(ss_resp, axis=0), label='SS_resp', color='r', lw=3.)
	plt.xlabel('Cycles around Stop-Signal')
	plt.ylabel('Average %s activity' % name)
	plt.legend(loc=0, fancybox=True)

    def analyze_act_stim_onset(self, name='SC', field='Thalam_unit_corr', tag='intact'):
        start_cycle = 0
	wind = (0,125)
        #wind = (100,100)
        # From emergent, SC threshold:
        data_grp_mean, data_grp_sem = emergent.group_batch(self.data['trl'][tag], ['SS_presented'])


        idx = data_grp_mean['SS_presented'] == 1
        SSD = np.median(self.data_settled[tag]['SSD'])

        ss_resp = self.extract_cycles(
            tag,
            ((self.data['trl'][tag]['SS_presented'] == 1) &
             (self.data['trl'][tag]['inhibited'] == 0) &
             (self.data['trl'][tag]['epoch'] > 30) &
             (self.data['trl'][tag]['SSD'] == SSD)),
            field, cycle=start_cycle,
            #center='SSD',
            wind=wind)

        ss_inhib = self.extract_cycles(
            tag,
            ((self.data['trl'][tag]['SS_presented'] == 1) &
             (self.data['trl'][tag]['inhibited'] == 1) &
             (self.data['trl'][tag]['epoch'] > 30) &
             (self.data['trl'][tag]['SSD'] == SSD)),
            field, cycle=start_cycle,
            #center='SSD',
            wind=wind)

        x=np.linspace(wind[0]+start_cycle,wind[1]+start_cycle,np.sum(wind)+1)*self.ms

        #thr_cross = np.where(np.mean(thalam_ss_resp, axis=0) > thr)[0][0]
        plt.plot(x, np.mean(ss_inhib, axis=0), label='canceled stop trials', color='.7', lw=3.)
        plt.plot(x, np.mean(ss_resp, axis=0), label='non-canceled stop trials', color='k', lw=3.)

        plt.axhline(y=self.SC_thr, color='k', lw=3., linestyle='-.')
        #plt.axvline(x=thr_cross, color='k')
        #plt.axvline(x=np.mean(self.SSD['intact'])+np.mean(self.SSRT['intact']), color='k')
        #plt.axvline(x=np.mean(self.SSD['intact']), color='k')
        plt.axvline(x=SSD*self.ms, color='k', lw=3.)
        plt.axvline(x=(SSD + np.mean(self.SSRT[tag]))*self.ms, color='k', linestyle='--', lw=2.)

        plt.ylim(0,1)
        plt.xlabel('Time from Target Onset (ms)')
        plt.ylabel('%s activity' % name)
        #plt.title('SC activity during inhibited and not-inhibited stop trials: %s'%tag)
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

    def analyze_act_post(self, tag=None, nucleus=None):
        cycle = 0
        if nucleus is None:
            nucleus = 'STN'

        if nucleus == 'STN':
            nuc_tag = 'STN_acts_avg'
        elif nucleus == 'SC':
            nuc_tag = 'Thalam_unit_corr'
            cycle = 20
        elif nucleus == 'IFG':
            nuc_tag = 'IFG_acts_avg'

        if tag is None:
            tag = 'intact'
	wind = (0, 150)
	ss_pre_none = self.extract_cycles(
	    tag,
	    ((self.data['trl'][tag]['SS_presented'] == 0) &
             (self.data['trl'][tag]['prev_trial_code'] <= 1) &
	     (self.data['trl'][tag]['inhibited'] == 0)),
            nuc_tag,
	    #center='SSD',
            cycle=cycle,
	    wind=wind)

	ss_pre_ss = self.extract_cycles(
	    tag,
	    ((self.data['trl'][tag]['SS_presented'] == 0) &
             (self.data['trl'][tag]['prev_trial_code'] > 1) &
	     (self.data['trl'][tag]['inhibited'] == 0)),
	    nuc_tag,
	    #center='SSD',
            cycle=cycle,
	    wind=wind)

	x = np.linspace(-wind[0]+cycle,wind[1]+cycle,np.sum(wind)+1) * self.ms
        plt.plot(x, np.mean(ss_pre_none, axis=0), label='previous Go trial', color='k', lw=3)
        plt.plot(x, np.mean(ss_pre_ss, axis=0), label='previous Stop trial', color='.7', lw=3)
#	plt.plot(x, np.mean(ss_Go, axis=0))
	plt.xlabel('Time from stimulus onset (ms)')
	plt.ylabel('%s activity'%nucleus)
	#plt.title('%s activity following Go and Stop trials: %s'%(nucleus, tag))
	plt.legend(loc='best', frameon=False)


@pools.register_group(['stopsignal', 'cycle', 'post'])
class StopSignal_cycle_post(StopSignal_cycle):
    def __init__(self, **kwargs):
        super(StopSignal_cycle_post, self).__init__(**kwargs)

    def analyze(self):
        self.new_fig()
        self.analyze_act_post(nucleus='SC')
        self.save_plot('SC_post_ss')

        self.new_fig()
        self.analyze_act_post(nucleus='STN')
        self.save_plot('STN_post_ss')

        self.new_fig()
        self.analyze_act_post(nucleus='IFG')
        self.save_plot('IFG_post_ss')

@pools.register_group(['stopsignal2', 'cycle', 'post'])
class StopSignal_cycle_post2(StopSignal_cycle):
    def __init__(self, **kwargs):
        super(StopSignal_cycle_post2, self).__init__(**kwargs)


        for flag_id in range(len(self.flags)):
            self.flags[flag_id]['thalam_inhib'] = 1.
            #self.flags[flag_id]['proj'] = self.prefix+self.proj + '_sc.proj'

@pools.register_group(['stopsignal', 'cycle', 'sc'])
class StopSignal_cycle_SC(StopSignal_cycle):
    def __init__(self, **kwargs):
        super(StopSignal_cycle_SC, self).__init__(**kwargs)

        self.flags = []
        self.tags = []
        # for a_thr in np.linspace(0.5,.85,5):
        #     for b_inc_dt in np.linspace(0,.3,10):
        #         self.flags.append(copy(self.flag))
        #         self.tags.append('%.3f:%.3f'%(a_thr, b_inc_dt))
        #         self.flags[-1]['tag'] = '_' + self.tags[-1]
        #         self.flags[-1]['a_thr'] = a_thr
        #         self.flags[-1]['b_inc_dt'] = b_inc_dt

        a_thr = 0.588
        b_inc_dt = 0.033
        # self.flags.append(copy(self.flag))
        # self.tags.append('%.3f:%.3f'%(a_thr, b_inc_dt))
        # self.flags[-1]['tag'] = '_' + self.tags[-1]
        # self.flags[-1]['a_thr'] = a_thr
        # self.flags[-1]['b_inc_dt'] = b_inc_dt

        for g_bar_a in np.linspace(3,8,10):
            self.flags.append(copy(self.flag))
            self.tags.append('%.3f'%(g_bar_a))
            self.flags[-1]['tag'] = '_' + self.tags[-1]
            self.flags[-1]['a_thr'] = a_thr
            self.flags[-1]['b_inc_dt'] = b_inc_dt
            self.flags[-1]['g_bar_a'] = g_bar_a

    def analyze(self):
        for tag in self.tags:
            self.new_fig()
            try:
                self.analyze_act_stim_onset(tag=tag)
            except ValueError:
                continue
            self.save_plot('sc_%s'%tag)

class StopSignalDDMBase(StopSignalBase):
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

        intact = kwargs.pop('intact', False)

        super(StopSignalDDMBase, self).__init__(intact=intact, decay_ifg=1, num_trials=2000, **kwargs)
        if not kwargs.has_key('depends'):
            self.depends = ['a', 't', 'v', 'ssrt']
        else:
            self.depends = kwargs['depends']

        self.condition = 'none'

    def set_flags_condition(self, condition, start, stop, samples, tag=None):
        self.x = np.linspace(start, stop, samples)
        self.condition = condition
        for i in self.x:
            if type(condition) is list:
                for c in condition:
                    self.flag[c] = i
            else:
                self.flag[condition] = i
            if tag is None:
                tag_name = '%s_%.4f' % (condition, i)
            else:
                tag_name = '%s_%.4f' % (tag, i)
            self.tags.append(tag_name)
            self.flag['tag'] = '_' + tag_name
            self.flags.append(copy(self.flag))

    def preprocess_data(self, cutoff=0):
	# Construct data structure for ddm fits.
	# Response 0 -> prosaccade
	# Response 1 -> antisaccade

	self.stimulus = {}
	self.response = {}
	self.rt = {}
        self.subj_idx = {}
        self.tags_array = {}

        dfs = []
        for tag in self.tags:
            df = pd.DataFrame(self.data[tag])
            df['dependent'] = tag
            dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)
        data = data.rename(columns={'SS_presented' : 'ss_presented',
                                    'minus_cycles' : 'rt',
                                    'error'        : 'response',
                                    'batch'        : 'subj_idx',
                                    'SSD'          : 'ssd'
        })


        data['rt'] *= self.ms/1000.
        data['ssd'] *= self.ms/1000.

        self.hddm_data = data

    def analyze(self):
        #self.plot_RT_histo()#(cutoff=50)
        #self.save_plot('RT_histo')

        self.plot_RT_dist_SSD()

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


    def fit_and_analyze_ddm(self, retry=False):
        #from multiprocessing import Pool

        self.hddm_models = {}

        experiments = [(self.hddm_data, {param: 'dependent'}) for param in self.depends]

        stats = map(emergent.fit_hddm_stop, experiments)

        self.stats_dict = {}

        for param, stat in zip(self.depends, stats):
            self.stats_dict[param] = stat

        self.new_fig()

        #plt.subplot(211)
        logps = [self.stats_dict[param]['logp'] for param in self.depends]
        print "Logps:"
        for param, logp in zip(self.depends, logps):
            print "& %s &  %d\\" %(param, round(logp))
        print "End"

        plt.plot(logps, lw=self.lw)
        plt.ylabel('logp')
        plt.xticks(np.arange(5), ['drift', 'thresh', 'ter', 'SSRT'])
        #if self.hddm_models.has_key('none'):
        #    plt.axhline(self.hddm_models['none'].logp)
        plt.title('HDDM model fits for different varying parameters')

        self.save_plot('model_probs')


    def plot_hddm_fit(self, range_=(-1., 1.), bins=150., plot_fit=True):
        import hddm
        import copy
        x = np.linspace(range_[0], range_[1], 200)

        # Plot parameters
        for i,test_param in enumerate(self.depends):
            #print test_param
            plt.figure()
            for j, dep_val in enumerate(self.x):
                param_vals = {}

                tag = "%s('%s_%.4f',)" %(test_param, self.condition, dep_val)
                dep_tag = '%s_%.4f'%(self.condition, dep_val)

                plt.subplot(3,3,j+1)

                data = self.hddm_data[(self.hddm_data['dependent']==dep_tag) & (self.hddm_data.inhibited == False) & (self.hddm_data.ss_presented == False)]
                data = hddm.utils.flip_errors(data)

                data_ss = self.hddm_data[(self.hddm_data['dependent']==dep_tag) & (self.hddm_data.inhibited == False) & (self.hddm_data.ss_presented == True)]
                data_ss = hddm.utils.flip_errors(data_ss)

                # Plot histogram
                hist = hddm.utils.histogram(data['rt'], bins=bins, range=range_,
                                            density=True)[0]
                plt.plot(np.linspace(range_[0], range_[1], bins), hist)

                # Plot histogram of stop-errors
                hist_ss = hddm.utils.histogram(data_ss['rt'], bins=bins, range=range_,
                                               density=True)[0]
                plt.plot(np.linspace(range_[0], range_[1], bins), hist_ss)

                # Plot fit
                if plot_fit:
                    fitted_params = copy.deepcopy(self.stats_dict[test_param])
                    for param in self.depends:
                        if param == test_param:
                            param_vals[param] = fitted_params[tag]['mean']
                        else:
                            param_name = param# +'_group'
                            param_vals[param] = fitted_params[param_name]['mean']

                    #fit = hddm.likelihoods.wfpt_switch.pdf(x, param_vals['vpp'], param_vals['vcc'], param_vals['Vcc'], param_vals['a'], .5, param_vals['t'], param_vals['tcc'], param_vals['T'])
                    #plt.plot(x, fit)

                plt.title(tag)

            if not plot_fit:
                # Bail, no need to replot the same thing
                self.save_plot('hddm_fit')
                return

            self.save_plot('hddm_fit_%s'%test_param)

    def plot_param_influences(self):
        # Plot parameters
        for i,test_param in enumerate(self.depends):
            y = []
            yerr = []
            for x in self.x:
                tag = "%s('%s_%.4f',)" %(test_param, self.condition, x)
                y.append(self.stats_dict[test_param][tag]['mean'])
                yerr.append(self.stats_dict[test_param][tag]['standard deviation'])
                #yerr.append(self.hddm_models[test_param].params_est_std[tag])
            plt.subplot(2,2,i+1)
            plt.errorbar(self.x, y=y, yerr=yerr, label=test_param, lw=self.lw, color='.8')
            plt.plot(self.x, y, 'o', color='.8')
            plt.ylabel(test_param)
        self.save_plot('param_influences')

@pools.register_group(['stopsignal', 'DDM', 'STN'])
class StopSignalDDMSTN(StopSignalDDMBase):
    def __init__(self, start=0.0, stop=.75, samples=5, **kwargs):
        super(self.__class__, self).__init__(SS_prob=.85, test_ssd_mode=True, **kwargs)
        self.set_flags_condition('STN_lesion', start, stop, samples)

@pools.register_group(['stopsignal', 'long'])
class StopSignalLong(StopSignalDDMBase):
    def __init__(self, **kwargs):
        super(StopSignalLong, self).__init__(intact=True, SS_prob=.85, **kwargs)
        self.flags[-1]['test_SSD_mode'] = True

    def analyze(self):
        self.plot_RT_dist_SSD()
        self.hddm_data['rt'] *= 3
        self.fit_and_analyze_ddm()

    def fit_and_analyze_ddm(self):
        #from multiprocessing import Pool
        from hddm.sandbox.model_stopddm import StopDDM

        model = StopDDM(self.hddm_data.to_records(), is_group_model=False)

        model.map(runs=2)
        model.sample(25000, burn=20000)
        model.print_stats()
        model.plot_posteriors()
