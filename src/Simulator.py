# -*- coding:utf-8 -*-
"""
Created on Mar 13, 2012

@author: alex
"""
from __future__ import division
import numpy as np
from numpy import zeros, ones, array, r_, float64, matrix, bmat, Inf, ceil, arange
#from scipy.linalg import lu_solve, inv, lu_factor
from numpy import zeros_like
from copy import deepcopy
from numpy import sin, sqrt
from numpy.random import randn, rand
from matplotlib.pyplot import savefig
ABCD_LABEL_SIZE = 30 
RESULTS_DIR = '/home/alex/Workspaces/Python/OU_hitting_time/Results/'
SIMPLE_RESULTS_DIR = '/home/alex/Workspaces/Python/OU_hitting_time/Results/Simple'
FIGS_DIR = '/home/alex/Workspaces/Latex/LIFEstimation/Figs/Trajectories'
import os

for D in [RESULTS_DIR, FIGS_DIR, SIMPLE_RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

class OUSinusoidalSimulator():
    def __init__(self, alpha, beta, gamma, theta):  
        self._path_params = OUSinusoidalParams( alpha, beta, gamma, theta)
        self._v_thresh = 1.
        
    def simulate(self, Tf, dt = None):
        #Set default dt:        
        if (None == dt):
            dt = 1.0 / self._path_params._theta / 32.0 #there are four phases to a sin cycle and conservatively say 8pts per cycle;
        
        #Set the (fixed) integration  times:
        ts = arange(0., Tf, dt);
#        num_steps = ceil(Tf / dt);
        #PReallocate space for the solution and the spike times:
        spike_ts = [];
        vs = zeros_like(ts);
        
        Bs = randn(len(ts));
        sqrt_dt = sqrt(dt);
        
        alpha, beta, gamma, theta, phi = self._path_params.getParams();

        #THE MAIN INTEGRATION LOOP:
        for t, idx in zip(ts[1:], xrange(1, len(ts))):            
            v_prev = vs[idx-1];
            
            dv = (alpha - v_prev + gamma*sin(theta*t))*dt + beta*Bs[idx]*sqrt_dt;
             
            v_new = vs[idx-1] + dv
            
            if v_new >= self._v_thresh:
            #You can try a more sophisticated zeroing in on the exact spike time, \e.g. linear interpolate based on the how (v_new - v_old) > (v_thresh - v_old)
                spike_ts.append(t)
                v_new = .0;
            
            vs[idx] = v_new;
        
        #Return:
        path_params = deepcopy(self._path_params)
        simulatedPath = Path(ts, vs, spike_ts, path_params)
        
        return simulatedPath;

class OUResetSinusoidalSimulator():
    def __init__(self, alpha, beta, gamma, theta, phi):  
        self._path_params = OUSinusoidalParams( alpha, beta, gamma, theta,phi)
        self._v_thresh = 1.
        
    def simulate(self, Tf, dt = None):
        #Set default dt:        
        if (None == dt):
            dt = 1.0 / self._path_params._theta / 32.0 #there are four phases to a sin cycle and conservatively say 8pts per cycle;
        
        #Set the (fixed) integration  times:
        ts = arange(0., Tf, dt);
#        num_steps = ceil(Tf / dt);
        #PReallocate space for the solution and the spike times:
        spike_ts = [];
        vs = zeros_like(ts);
        
        Bs = randn(len(ts));
        sqrt_dt = sqrt(dt);
        
        alpha, beta, gamma, theta, phi = self._path_params.getParams();

        t_last = .0
        
        #THE MAIN INTEGRATION LOOP:
        for t, idx in zip(ts[1:], xrange(1, len(ts))):            
            v_prev = vs[idx-1];
            
            dv = (alpha - v_prev + gamma*sin(theta*(t -t_last + phi)))*dt + beta*Bs[idx]*sqrt_dt;
             
            v_new = vs[idx-1] + dv
            
            if v_new >= self._v_thresh:
            #You can try a more sophisticated zeroing in on the exact spike time, \e.g. linear interpolate based on the how (v_new - v_old) > (v_thresh - v_old)
                spike_ts.append(t)
                v_new = .0;
                t_last = t;
            
            vs[idx] = v_new;
        
        #Return:
        path_params = deepcopy(self._path_params)
        simulatedPath = Path(ts, vs, spike_ts, path_params)
        
        return simulatedPath;


class OUBinnedSinusoidalSimulator():
    def __init__(self, alpha, beta, gamma, theta, phi, dphi):
        self._path_params = OUSinusoidalParams( alpha, beta, gamma, theta, phi)

        self._v_thresh = 1.
        self._dphi = dphi;
        
    def simulate(self, Tf, dt = None):
        #Set default dt:        
        if (None == dt):
            dt = 1.0 / self._path_params._theta / 32.0 #there are four phases to a sin cycle and conservatively say 8pts per cycle;
        
        #Get the path parameters:
        alpha, beta, gamma, theta, phi = self._path_params.getParams();

        #Set the (fixed) integration  times:
        ts = arange(0., Tf, dt);
#        num_steps = ceil(Tf / dt);
        
        #Preallocate space for the solution and the spike times and spike phases:
        spike_ts = [];
        vs = zeros_like(ts);
        spike_phases = [phi];
        
        Bs = randn(len(ts));

        sqrt_dt = sqrt(dt);

        #Time of last spike:        
        t_last =.0;
        
        #set the deviation from phi_m, start in the middle!:
        dphi = 0.;
        
        #THE MAIN INTEGRATION LOOP:
        for t, idx in zip(ts[1:], xrange(1, len(ts))):            
            v_prev = vs[idx-1];
            
            dv = (alpha - v_prev + gamma*sin(theta*(t - t_last  + phi + dphi)))*dt + beta*Bs[idx]*sqrt_dt;
             
            v_new = vs[idx-1] + dv
            
            if v_new >= self._v_thresh:
            #You can try a more sophisticated zeroing in on the exact spike time, \e.g. linear interpolate based on the how (v_new - v_old) > (v_thresh - v_old)
                t_last = t;
                spike_ts.append(t)
                #Append the phase of the current spike:
                spike_phases.append(phi+dphi)

                v_new = .0;
                
                #Reset the deviation:
                dphi = -self._dphi/2 + self._dphi * rand()
            
            vs[idx] = v_new;
        
        #Return:
        path_params = deepcopy(self._path_params)
        
        if len(spike_phases) > len(spike_ts):
            spike_phases = spike_phases[:-1]

        simulatedPath = SinBinPath(ts, vs, spike_ts, path_params, spike_phases)
        
        return simulatedPath;


class OUSimpleSimulator():
    def __init__(self, a, b):
        self._path_params = OUSimpleParams(a,b)
        self._v_thresh = 1.0;

    def simulate(self, Tf, dt = None):
        #Set default dt:        
        if (None == dt):
            dt = .01
        
        #Set the (fixed) integration  times:
        ts = arange(0., Tf, dt);
        num_steps = ceil(Tf / dt);
        
        #PReallocate space for the solution and the spike times:
        spike_ts = [];
        vs = zeros_like(ts);
        
        Bs = randn(len(ts));
        sqrt_dt = sqrt(dt);
        
        alpha, beta = self._path_params.getParams();

        #THE MAIN INTEGRATION LOOP:
        for t, idx in zip(ts[1:], xrange(1, len(ts))):            
            v_prev = vs[idx-1];
            
            dv = (alpha - v_prev) * dt + beta*Bs[idx]*sqrt_dt;
             
            v_new = vs[idx-1] + dv
            
            if v_new >= self._v_thresh:
            #You can try a more sophisticated zeroing in on the exact spike time, \e.g. linear interpolate based on the how (v_new - v_old) > (v_thresh - v_old)
            #Brownian bridge etc.
                spike_ts.append(t)
                v_new = .0;
            
            vs[idx] = v_new;
        
        #Return:
        path_params = deepcopy(self._path_params)
        simulatedPath = Path(ts, vs, spike_ts, path_params)
        
        return simulatedPath;


########################
class Path():
    def __init__(self, ts, vs, spike_ts, params):
        self._vs = vs;
        self._ts  = ts;
        self._spike_ts = spike_ts;
        
#        TODO: diff spike_ts to obtain Is here???
    
        self._params = params;
        
    def save(self, file_name):
#        path_data = {'path' : self}
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @staticmethod
    def load(file_name):
        import cPickle
        load_file = open(file_name, 'r')
        path = cPickle.load(load_file)        
        return path

class SinBinPath(Path):
    def __init__(self, ts, vs, spike_ts, params, spike_phases):
        assert(len(spike_phases) == len(spike_ts))
        
        Path.__init__(self, ts, vs, spike_ts, params)
        self._spike_phases = spike_phases;
        
    
#    def printStats(self):
#        print 'Path params = (a, b,g, \\th) = (%g, %g, %g, %g)' %(P._params._alpha,
#                                                                    P._params._beta,
#                                                                    P._params._gamma,
#                                                                    P._params._theta) 
#    
#        Is = r_[(P._spike_ts[0], diff(array(P._spike_ts)))];
#        
#        N = len(Is)
#        assert((Is>0).all())
#        assert((diff(sort(Is))>=0.).all())
#        
#        print 'spike count = ', N
#        
#        print 'least interval = ', min(Is)
#        print 'max interval = ', max(Is)
#        
#        print 'max transtion = ', max(diff(sort(Is)))
#        print 'avg transtion = ', mean(diff(sort(Is)))
#        print 'median transtion = ', median(diff(sort(Is)))
#    
        

########################
class OUSinusoidalParams():
    def __init__(self, alpha, beta, gamma, theta=2.0, phi = .0):
            self._alpha = alpha;
            self._beta = beta;
            self._gamma = gamma;
            self._theta  = theta;
            self._phi  = phi;
            
    def getParams(self):
        return self._alpha, self._beta, self._gamma, self._theta, self._phi
    
class OUSimpleParams():
    def __init__(self, alpha, beta):
            self._alpha = alpha;
            self._beta = beta;
            
    def getParams(self):
        return self._alpha, self._beta;
           
def simulateSimple():
    a = 1.5; b = .3;
    S = OUSimpleSimulator(a,b);
    
    Tf = 6000.0;
    
    P =  S.simulate(Tf);
    
    figure()
    
    subplot(211)
    plot (P._ts, P._vs)
    ylabel('$v(t)$', fontsize = 24)
    
    if (0<len(P._spike_ts)):
        hold(True)
        stem(P._spike_ts, 1.25*ones_like(P._spike_ts), 'r');
        ylim( (.0, 1.3) );
        xlim( (P._ts[0], P._ts[-1]))
        xlabel('$t$', fontsize = 24)
        
    
    subplot(212)
    Is =  diff(array(P._spike_ts));
    stem(P._spike_ts[1:],Is)    
    ylim( (.0, max(Is)) );
    xlim( (P._ts[0], P._ts[-1]))
    xlabel('$t_n$', fontsize = 24)
    ylabel('$I_n$', fontsize = 24);
    
    filename = os.path.join(RESULTS_DIR, 'simple_spike_train_T=6K.path')   
    
    print 'Saving to ' , filename
    
    P.save(filename)
    
#        
#def spike_example_SuperT():
#    mu = 1.5; tau = 1.0; A = 1000.0; omega = 2.0; sigma = 1.0; v_thresh = 1.0;
#    
#    S = OUSinusoidalSimulator(mu, tau, A, omega, sigma, v_thresh)
#    
#    Tf = 60.0 #secs
#
#    P = S.simulate(Tf);
#    
#    figure()
#    
#    subplot(211)
#    plot (P._ts[::4], P._vs[::4])
#    ylabel('$v(t)$', fontsize = 24)
#    
#    if (0<len(P._spike_ts)):
#        subplot(212)
#        stem(P._spike_ts, ones_like(P._spike_ts));
#        ylim( (.0, 1.25) );
#        xlim( (P._ts[0], P._ts[-1]))
#        xlabel('$t$', fontsize = 24)
#        
#    filename = os.path.join(FIGS_DIR, 'spike_example_superT.png')
#    savefig(filename)
#    
    
    
        
def sinusoidal_spike_train(Tf = 60., save_path=False, path_tag = '',
                                     save_fig=False, fig_tag  = '',
                                     params = [1.5,.3, 1.0, 2.0],
                                     dt = .001):
    #Parametrization:
#    alpha = 1.5; beta = .3; gamma = 1.0; theta = 2.0
    alpha,beta,gamma,theta = params[0],params[1], params[2], params[3];
    print 'Parameterizing with: ', alpha, beta, gamma, theta 

    # The actual work:  
    S = OUSinusoidalSimulator(alpha, beta, gamma, theta);
    P = S.simulate(Tf, dt);
    
    #Post-processing:
    N = len(P._spike_ts)
    assert (0<N)
    print 'Spike count = ', N
    
    if save_fig or '' != fig_tag:
        
        mpl.rcParams['figure.subplot.left'] = .15
        mpl.rcParams['figure.subplot.right'] =.975
        mpl.rcParams['figure.subplot.bottom'] = .1
        mpl.rcParams['figure.subplot.hspace'] = .5

        figure()
#        subplot(211)
        spike_height = 1.5
        hold(True)
        plot (P._ts, P._vs)
        title(r'$\alpha,\beta,\gamma = (%.3g,%.3g,%.3g) $' %(alpha,beta,gamma), fontsize = 32)
        ylabel('$X_t$', fontsize = 24)
        vlines(P._spike_ts, .0, spike_height*ones_like(P._spike_ts), linewidth=2);
        hlines(0, 0, P._ts[-1], linestyles='dashed',  linewidth=2);
        hlines(1.0, 0, P._ts[-1], 'r', linestyles='dashed',  linewidth=2);
        
        ylim( (amin(P._vs), spike_height) );
        xlim( (P._ts[0], P._ts[-1]))
        xlabel('$t$', fontsize = 24)
        plt.tick_params(labelsize = 16)
        
#        subplot(212)
#        sins = gamma* sin(theta*(P._ts))
#        plot (P._ts, sins)
#        title(r'Input sinusoidal current', fontsize = 32)
#        ylabel('$\gamma * \sin(t)$', fontsize = 24)
#        xlim( (P._ts[0], P._ts[-1]))
#        xlabel('$t$', fontsize = 24)
        
        get_current_fig_manager().window.showMaximized() 
        filename = os.path.join(FIGS_DIR, 'path_T=%d_%s.png'%(Tf, fig_tag))
        print 'Saving fig to: ', filename
        savefig(filename) 
        
#        figure()
#        subplot(211)
#        Is =  diff(array(P._spike_ts));
#        stem(P._spike_ts[1:],Is)    
#        ylim( (.0, max(Is)) );
#        xlim( (P._ts[0], P._ts[-1]))
#        title(r'Observed $I_n$', fontsize = 24)
#       
#        xlabel('$t_n$', fontsize = 24)
#        ylabel('$I_n$', fontsize = 24);
#        tick_locs = P._spike_ts[1::4]
#        tick_lbls = ['%.2g'%x for x in P._spike_ts[1::4]]
#        plt.xticks(tick_locs, tick_lbls)
#        plt.tick_params(labelsize = 16)
#        
#        subplot(212)
#        sins = sin(theta*(P._ts))
#        plot (P._ts, sins)
#        title(r'Input sinusoidal current: $\gamma = ?$', fontsize = 24)
#        ylabel('$\sin(t)$', fontsize = 24)
#        xlim( (P._ts[0], P._ts[-1]))
#        xlabel('$t$', fontsize = 24)
#        
#        get_current_fig_manager().window.showMaximized() 
#        filename = os.path.join(FIGS_DIR, 'ISI_T=%d_%s.png'%(Tf, fig_tag))
#        print 'Saving fig to: ', filename
#        savefig(filename)         

    if save_path:
        filename = os.path.join(RESULTS_DIR, 'sinusoidal_spike_train_T=%d_%s.path'%(Tf, path_tag))  
        print 'Saving path to ' , filename    
        P.save(filename)
           
        
def sinusoidal_spike_train_single_phi(Tf = 60., phi_normalized = .0, save_path=False):  
    alpha = 1.5; beta = .3; gamma = 1.0; theta = 2.0
    S = OUResetSinusoidalSimulator(alpha, beta, gamma, theta, 2*pi / theta * phi_normalized);
  
    P = S.simulate(Tf);
    
    figure()
    
    subplot(211)
    plot (P._ts, P._vs)
    ylabel('$v(t)$', fontsize = 24)
    
    print 'Interval count = ', len(P._spike_ts)
    
    if (0<len(P._spike_ts)):
        hold(True)
        stem(P._spike_ts, 1.25*ones_like(P._spike_ts), 'r');
        ylim( (.0, 1.3) );
        xlim( (P._ts[0], P._ts[-1]))
        xlabel('$t$', fontsize = 24)
    
    title('phi_normalized = %g'%phi_normalized)
    subplot(212)
    Is =  diff(array(P._spike_ts));
    stem(P._spike_ts[1:],Is)    
    ylim( (.0, max(Is)) );
    xlim( (P._ts[0], P._ts[-1]))
    xlabel('$t_n$', fontsize = 24)
    ylabel('$I_n$', fontsize = 24);
    title('')
    
    if save_path:
        filename = os.path.join(RESULTS_DIR, 'single_phi_train_phi=%d_T=%d.path'%(10*phi_normalized,Tf))
        print 'Saving to ' , filename    
        P.save(filename)


def tstep_driver(Tf = 60., phi_normalized = .4, save_path=False):  
    alpha = 1.5; beta = .3; gamma = 1.0; theta = 2.0
    S = OUResetSinusoidalSimulator(alpha, beta, gamma, theta, 2*pi / theta * phi_normalized);
    
    dt_base =     1.0 / theta / 32.0 #there are four phases to a sin cycle and conservatively say 8pts per cycle;
        
    for dt_exp in [1,2]: 
        dt = dt_base * (10.**(-dt_exp));
    
        P = S.simulate(Tf, dt);
    
        N = len(P._spike_ts);         print 'Interval count = ', N 

        figure()
        
        Is =  diff(array(P._spike_ts));
        stem(P._spike_ts[1:],Is);    
        xlim( (P._ts[0], P._ts[-1]));    ylim( (.0, max(Is)) );           
        xlabel('$t_n$', fontsize = 24);  ylabel('$I_n$', fontsize = 24);
        title('dt = %g' %dt)
        
        if save_path:
            filename = os.path.join(RESULTS_DIR, 'single_phi_train_phi=%d_dt_exp=%d_T=%d.path'%(10*phi_normalized,dt_exp,Tf))
            print 'Saving to ' , filename    
            P.save(filename)

        
def sinusoidal_spike_train_single_bin(Tf = 60., phi_normalized = .0, dphi_normalized = .01, save_path=False):
    assert(phi_normalized >= .0 and phi_normalized <=.9)
    assert(dphi_normalized >=.0 and dphi_normalized <=.99)
    
    alpha = 1.5; beta = .3; gamma = 1.0; theta = 2.0
    
    phi = phi_normalized * 2* pi / theta;
    dphi = dphi_normalized *2* pi / theta;
    
    S = OUBinnedSinusoidalSimulator(alpha, beta, gamma, theta, phi, dphi);
  
    P = S.simulate(Tf);
    
    figure()
    
    subplot(311)
    plot (P._ts, P._vs)
    ylabel('$v(t)$', fontsize = 24)
    
    if (0<len(P._spike_ts)):
        hold(True)
        stem(P._spike_ts, 1.25*ones_like(P._spike_ts), 'r');
        ylim( (.0, 1.3) );
        xlim( (P._ts[0], P._ts[-1]))
        xlabel('$t$', fontsize = 24)
    
    subplot(312)
    Is =  diff(array(P._spike_ts));
    stem(P._spike_ts[1:],Is)    
    ylim( (.0, max(Is)) );
    xlim( (P._ts[0], P._ts[-1]))
    xlabel('$t_n$', fontsize = 24)
    ylabel('$I_n$', fontsize = 24);
    
    subplot(313); hold (True)
    plot(arange(0, len(P._spike_phases)), P._spike_phases)
    title ('spike_phases')
    hlines([phi-dphi/2., phi, phi + dphi/2.], 0, len(P._spike_phases), colors = 'r', linestyles ='dashed')
    ylim((phi-1.25*dphi, phi+1.25*dphi))
    
    print 'Interval count = ', len(P._spike_phases)
    
    if save_path:
        filename = os.path.join(RESULTS_DIR, 'bin_train_phi_m=%d_dphi=%d_T=%g.path'%(10*phi_normalized, 100*dphi_normalized, Tf))
        print 'Saving to ' , filename    
        P.save(filename)

def simpleBatch():
    a = .8; b = 1.
    for idx in range(1, 65):
        
        S = OUSimpleSimulator(a,b);
    
        Tf = 100.0;
    
        P =  S.simulate(Tf, dt = 1e-4);
    
        filename = os.path.join(SIMPLE_RESULTS_DIR, 'a=0.8_b=1.0_T=100_' + str(idx) +'.path')   
    
        print 'Saving to ' , filename
    
        P.save(filename)
        
        print 'Interval count = ', len(P._spike_ts)
        
        
def batchSimulator():
#    for idx in range(1,9):
#    for idx in range(9,17):
#        sinusoidal_spike_train(5000.0, save_path=True, path_tag = 'crit_%d'%idx, 
#                           params = [.55, .5, .55, 2.0])
#
#        sinusoidal_spike_train(5000.0, save_path=True, path_tag = 'superT_%d'%idx, 
#                           params = [1.5, .3, 1.0, 2.0])
#
#        sinusoidal_spike_train(20000.0, save_path=True, path_tag = 'subT_%d'%idx, 
#                           params = [.4, .3, .4, 2.0])
#
#        sinusoidal_spike_train(5000.0, save_path=True, path_tag = 'superSin_%d'%idx, 
#                           params = [.1, .3, 2.0, 2.0])
    for idx in range(1,17):
#        sinusoidal_spike_train(5000.0, save_path=True, path_tag = 'crit_%d'%idx, 
#                           params = [.55, .5, .55, 2.0])
#
#        sinusoidal_spike_train(5000.0, save_path=True, path_tag = 'superT_%d'%idx, 
#                           params = [1.5, .3, 1.0, 2.0])

        sinusoidal_spike_train(20000.0, save_path=True, path_tag = 'subT_%d'%idx, 
                           params = [.4, .3, .4, 2.0])

#        sinusoidal_spike_train(5000.0, save_path=True, path_tag = 'superSin_%d'%idx, 
#                           params = [.1, .3, 2.0, 2.0])

def generateSDF(Is):
    N = len(Is)
    unique_Is = unique(Is)
    SDF = zeros_like(unique_Is)
    
    for (Ik, idx) in zip(unique_Is, arange(len(SDF))):
        SDF[idx] = sum(Is> Ik) / N;
        
    return SDF, unique_Is

def visualizeDistributions(file_name, fig_name):
    file_name = os.path.join(RESULTS_DIR, file_name)
    P = Path.load(file_name)
    Is = r_[(P._spike_ts[0], diff(array(P._spike_ts)))];
    
    SDF, unique_Is = generateSDF(Is)
    
    mpl.rcParams['figure.subplot.left'] = .15
    mpl.rcParams['figure.subplot.right'] =.95
    mpl.rcParams['figure.subplot.bottom'] = .15
    mpl.rcParams['figure.subplot.hspace'] = .4
    
    figure()
    ax = subplot(211)
    hist(Is, 100)
    title(r'$\alpha,\beta,\gamma = (%.3g,%.3g,%.3g) $' %(P._params._alpha, P._params._beta, P._params._gamma), fontsize = 24)
    xlabel('$I_n$', fontsize = 22);
    ylabel('$g(t)$', fontsize = 22);
    
    for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(20)
    for label in ax.yaxis.get_majorticklabels():
        label.set_fontsize(20)
    
    
    ax = subplot(212)
    plot(unique_Is, SDF, 'rx', markersize = 10)
    ylim((.0,1.))
    xlabel('$t$', fontsize = 22)
    ylabel('$1 - G(t)$', fontsize = 22)
    for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(20)
    for label in ax.yaxis.get_majorticklabels():
        label.set_fontsize(20)
            
    
    fig_name = os.path.join(FIGS_DIR, fig_name)
    print 'saving to ', fig_name
    savefig(fig_name)

def visualize_regime_paths(regimeParams, inner_titles,
                           Tf = 24., save_fig=False, fig_tag  = '',
                                     dt = .0001):
    #Parametrization:
    mpl.rcParams['figure.figsize'] = 17, 6*2
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['figure.subplot.left'] = .15
    mpl.rcParams['figure.subplot.right'] =.975
    mpl.rcParams['figure.subplot.bottom'] = .075
    mpl.rcParams['figure.subplot.hspace'] = .25
    mpl.rcParams['figure.subplot.wspace'] = .25
    label_font_size = 24
    xlabel_font_size = 40
    
    figure()
    for idx, params in enumerate(regimeParams):
        alpha,beta,gamma,theta = params[0],params[1], params[2], params[3];
        print 'Parameterizing with: ', alpha, beta, gamma, theta 
    
        # The actual work:  
        S = OUSinusoidalSimulator(alpha, beta, gamma, theta);
        P = S.simulate(Tf, dt);
        
        #Post-processing:
        N = len(P._spike_ts)
        assert (0<N)
        print 'Spike count = ', N
  

        ax = subplot(2,2,idx+1)
        spike_height = 1.5
        hold(True)
        plot (P._ts, P._vs, linewidth=.5)
#        title(r'$\alpha,\beta,\gamma = (%.3g,%.3g,%.3g) $' %(alpha,beta,gamma), fontsize = 32)
        if (0 == mod(idx,2)):
            ylabel('$X_t$', fontsize = xlabel_font_size)
        vlines(P._spike_ts, .0, spike_height*ones_like(P._spike_ts), linewidth=2);
        hlines(0, 0, P._ts[-1], linestyles='dashed',  linewidth=2);
        hlines(1.0, 0, P._ts[-1], 'r', linestyles='dashed',  linewidth=2);
        
        ylim( (amin(P._vs), spike_height) );
        xlim( (P._ts[0], P._ts[-1]))
        if (idx > 1):
            xlabel('$t$', fontsize = xlabel_font_size)
        tick_params(labelsize = label_font_size)
        
        def add_inner_title(ax, title, loc, size=None, **kwargs):
            from matplotlib.offsetbox import AnchoredText
            from matplotlib.patheffects import withStroke
            if size is None:
                size = dict(size=plt.rcParams['legend.fontsize'])
            at = AnchoredText(title, loc=loc, prop=size,
                              pad=0., borderpad=0.5,
                              frameon=False, **kwargs)
            ax.add_artist(at)
            at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
            return at
        t = add_inner_title(ax, inner_titles[idx], loc=3,
                             size=dict(size=ABCD_LABEL_SIZE))
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)
    
    if save_fig:
        get_current_fig_manager().window.showMaximized() 
        filename = os.path.join(FIGS_DIR, 'path_T=%d_%s.pdf'%(Tf, 'combined'))
        print 'Saving fig to: ', filename
        savefig(filename, dpi=(300))
 
if __name__ == '__main__':
    from pylab import *
    
#    batchSimulator()
#    simpleBatch()
    
#    simulateSimple()
    
#    for phi_norm in arange(.1, 1., .1):    
#        sinusoidal_spike_train_single_phi(3000, phi_norm, save_path=True)
#    tstep_driver(Tf=600, phi_normalized = .4, save_path=True)


#    for dpn in [.01, .05, .1, .2]:
#        sinusoidal_spike_train_single_bin(6000*dpn,
#                                          phi_normalized = .4,
#                                          dphi_normalized=dpn, save_path = True)
    theta = 1.0;
#    for params, tag in zip([[1.5-.1, .3, .1 * sqrt(1. + theta**2),     theta],
#                                [.5,     .3, .5* sqrt(1. + theta**2),      theta],
#                                [.4,     .3, .4* sqrt(1. + theta**2),      theta],
#                                [.1,     .3, (1.5-.1)*sqrt(1. + theta**2), theta]],
#                                ['superT', 'crit', 'subT','superSin']):
#        sinusoidal_spike_train(24, save_path=False, 
#                           save_fig = True, fig_tag='TrajExample_' + tag,
#                           params=params,
#                           dt = 1e-4)
#    
        
    regimeParams = [[1.5-.1, .3, .1 * sqrt(1. + theta**2),     theta],
                    [.1,     .3, (1.5-.1)*sqrt(1. + theta**2), theta],
                    [.5,     .3, .5* sqrt(1. + theta**2),      theta],
                    [.4,     .3, .4* sqrt(1. + theta**2),      theta]
                    ]
    inner_titles = {0: 'A',
                    1:'B',
                    2:'C',
                    3:'D'}
    visualize_regime_paths(regimeParams, inner_titles,
                            save_fig=True, dt = 5e-4)
    
#    sinusoidal_spike_train(10000.0, save_path=True, path_tag = 'superSin_3', 
#                           params = [.1, .3, 2.0, 2.0])
#    
#    sinusoidal_spike_train(25000.0, save_path=True, path_tag = 'subT_3', 
#                           params = [.4, .3, .4, 2.0])
#    
        
    
#    sinusoidal_spike_train(Tf=120, save_fig=True, fig_tag='SuperT', params = [1.5, .3, 1.0, 2.0])
#    sinusoidal_spike_train(Tf=120, save_fig=True, fig_tag='SuperSin',params =  [.1, .3, 2.0, 2.0])
    
#    visualizeDistributions('sinusoidal_spike_train_T=5000_superSin_14.path', 'SuperSin_Distributions')
#    visualizeDistributions('sinusoidal_spike_train_T=5000_superT_14.path', 'SuperT_Distributions')
    
    show()
        