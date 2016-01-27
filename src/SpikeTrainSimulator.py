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

RESULTS_DIR = '/home/alex/Workspaces/Python/OU_hitting_time/SpikeTrains/'
FIGS_DIR = '/home/alex/Workspaces/Latex/LIFEstimation/Figs/OU'
import os

import ext_fpc

for D in [RESULTS_DIR, FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

class OUSinusoidalSimulator():
    def __init__(self, alpha, beta, gamma, theta=1.0, tau=1.0):  
        self._path_params = OUSinusoidalParams( alpha, beta, gamma, theta)
        self._v_thresh = 1.
        
    def simulate(self, spikes_requested, dt = None):
        #Set default dt:        
        if (None == dt):
            dt = 1e-4 / self._path_params._theta
        
        #Main Sim Routine
#        spike_ts = self._pysimulate(spikes_requested, dt)
        spike_ts = self._csimulate(spikes_requested, dt)
        
        #Return:
        path_params = deepcopy(self._path_params)
        simulatedSpikeTrain = SpikeTrain(spike_ts, path_params)
        
        return simulatedSpikeTrain;
        
    def _csimulate(self, spikes_requested, dt):
#        alpha, beta, gamma, theta = ;
        abgth = array( self._path_params.getParams() );
        return ext_fpc.simulateSDE(abgth,spikes_requested,dt);
        
    def _pysimulate(self, spikes_requested, dt):
        #Set the (fixed) integration  times:
        spike_ts = [];
#        ts = arange(0., spikes_requested, dt);
#        vs = zeros_like(ts);
        
#        Bs = randn(len(ts));
        sqrt_dt = sqrt(dt);
        
        alpha, beta, gamma, theta = self._path_params.getParams();

        #THE MAIN INTEGRATION LOOP:
        v = .0
        t = .0;
        recorded_spikes =0;
        while recorded_spikes < spikes_requested:
            
            dB = randn()*sqrt_dt
            dv = (alpha - v + gamma*sin(theta*t))*dt + \
                          beta*dB

            v += dv;
            t += dt;
            if v >= self._v_thresh:
                spike_ts.append(t)
                v = .0;
                recorded_spikes += 1

        #Return:
        return spike_ts;


########################
class SpikeTrain():
    FILE_EXTENSION = '.st'
    def __init__(self, spike_ts, params):
        self._spike_ts = spike_ts;
        self._params = params; # this is an instance of OUSinusoidalParams.
    
    def getTf(self):
        return self._spike_ts[-1]
        
    def save(self, file_name):
        import cPickle
        dump_file = open(file_name + SpikeTrain.FILE_EXTENSION, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @staticmethod
    def load(file_name):
        import cPickle
        load_file = open(file_name+ SpikeTrain.FILE_EXTENSION, 'r')
        path = cPickle.load(load_file)        
        return path

########################
class OUSinusoidalParams():
    def __init__(self, alpha, beta, gamma, theta=1.0):
            self._alpha = alpha;
            self._beta = beta;
            self._gamma = gamma;
            self._theta  = theta;
            
    def getParams(self):
        return self._alpha, self._beta, self._gamma, self._theta
                
def sinusoidal_spike_train(N_spikes = 60, save_path=False, path_tag = '',
                                     save_fig=False, fig_tag  = '',
                                     params = [],
                                     dt = None,
                                     overwrite = False):
    filename = os.path.join(RESULTS_DIR, 'sinusoidal_spike_train_N=%d_%s'%(N_spikes, path_tag))  
    if save_path and False == overwrite and True  == os.path.exists(filename+'.st'):
        print filename, ' exists, returning'
        return
        
    alpha,beta,gamma,theta = params[0],params[1], params[2], params[3];
    print 'Parameterizing with: ', alpha, beta, gamma, theta 

    # The actual work:  
    S = OUSinusoidalSimulator(alpha, beta, gamma, theta);
    T = S.simulate(N_spikes, dt);
    print 'Spike count = ', N_spikes
    print 'Simulation time = ' , T.getTf()
    print T._params.getParams()
    
    #Post-processing:
    if save_fig or '' != fig_tag:
        figure()
#        subplot(211)
#        hold(True)
#        plot (P._ts, P._vs)
#        title(r'$\alpha,\beta,\gamma = (%.3g,%.3g,%.3g) $' %(alpha,beta,gamma), fontsize = 24)
#        ylabel('$v(t)$', fontsize = 24)
#        stem(P._spike_ts, 1.25*ones_like(P._spike_ts), 'r');
#        ylim( (.0, 1.3) );
#        xlim( (P._ts[0], P._ts[-1]))
#        xlabel('$t$', fontsize = 24)
        
#        subplot(212)
        Is =  diff(array(T._spike_ts));
        stem(T._spike_ts[1:],Is)    
        ylim( (.0, max(Is)) );
        xlim( (0., T.getTf()))
        xlabel('$t_n$', fontsize = 24)
        ylabel('$I_n$', fontsize = 24);
        
        filename = os.path.join(FIGS_DIR, 'sinusoidal_train_N=%d_%s.png'%(N_spikes, fig_tag))
        print 'Saving fig to: ', filename
        savefig(filename) 
        
    if save_path:   
        print 'Saving path to ' , filename    
        T.save(filename)
           
#
#def tstep_driver(Tf = 60., phi_normalized = .4, save_path=False):  
#    alpha = 1.5; beta = .3; gamma = 1.0; theta = 2.0
#    S = OUResetSinusoidalSimulator(alpha, beta, gamma, theta, 2*pi / theta * phi_normalized);
#    
#    dt_base =     1.0 / theta / 32.0 #there are four phases to a sin cycle and conservatively say 8pts per cycle;
#        
#    for dt_exp in [1,2]: 
#        dt = dt_base * (10.**(-dt_exp));
#    
#        P = S.simulate(Tf, dt);
#    
#        N = len(P._spike_ts);         print 'Interval count = ', N 
#
#        figure()
#        
#        Is =  diff(array(P._spike_ts));
#        stem(P._spike_ts[1:],Is);    
#        xlim( (P._ts[0], P._ts[-1]));    ylim( (.0, max(Is)) );           
#        xlabel('$t_n$', fontsize = 24);  ylabel('$I_n$', fontsize = 24);
#        title('dt = %g' %dt)
#        
#        if save_path:
#            filename = os.path.join(RESULTS_DIR, 'single_phi_train_phi=%d_dt_exp=%d_T=%d.path'%(10*phi_normalized,dt_exp,Tf))
#            print 'Saving to ' , filename    
#            P.save(filename)

def thetaSimulator(N_spikes = 1000, N_trains = 100, overwrite=False,
                   thetas = [1, 5, 10, 20]):
    
    tag = 'critical'
    for  idx in range(1, N_trains+1):
        for theta in thetas:
            params =  [.5,   .3, .5* sqrt(1. + theta**2),      1.0*theta]
                      
                      
            path_tag = '%s_theta=%d_%d'%(tag, theta, idx)
                    
            sinusoidal_spike_train(N_spikes, save_path=True, path_tag = path_tag, 
                                    params = params,
                                    overwrite = overwrite)
                
                
def batchSimulator(N_spikes = 100, N_trains = 100, overwrite=False, theta = 1.0):
    
    for  idx in range(1, N_trains+1):
        for params, tag in zip([[1.5-.1, .3, .1 * sqrt(1. + theta**2),     theta],
                                [.5,     .3, .5* sqrt(1. + theta**2),      theta],
                                [.4,     .3, .4* sqrt(1. + theta**2),      theta],
                                [.1,     .3, (1.5-.1)*sqrt(1. + theta**2), theta]],
                                ['superT', 'crit', 'subT','superSin']):
            
                    path_tag = '%s_%d'%(tag, idx)
                    
                    
                    sinusoidal_spike_train(N_spikes, save_path=True, path_tag = path_tag, 
                                                params = params, overwrite=True)

def generateSDF(Is):
    N = len(Is)
    unique_Is = unique(Is)
    
    ts = r_[(.0, unique_Is)]
    SDF = ones_like(ts)
    
    for (Ik, idx) in zip(unique_Is, arange(1, len(SDF))):
        SDF[idx] = sum(Is> Ik) / N;
    return SDF, unique_Is

def visualizeDistributions(file_name, fig_name):
    file_name = os.path.join(RESULTS_DIR, file_name)
    P = SpikeTrain.load(file_name)
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
    plot(r_[(.0,unique_Is)], SDF, 'rx', markersize = 10)
    ylim((.0,1.))
    xlabel('$t$', fontsize = 22)
    ylabel('$1 - G(t)$', fontsize = 22)
    for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(20)
    for label in ax.yaxis.get_majorticklabels():
        label.set_fontsize(20)
            
    
    fig_name = os.path.join(FIGS_DIR, fig_name)
    print 'saving to ', fig_name
#    savefig(fig_name)

def CvsPySimulator():
    import time
    N_spikes = 1000;
    for params, tag in zip([[1.5, .3, 1.0, 2.0], [.5, .3, 1.12, 2.0], [.4, .3, .4, 2.0],[.1, .3, 2.5, 2.0]],
                           ['superT', 'crit', 'subT','superSin']):
        alpha,beta,gamma,theta = params[0],params[1], params[2], params[3];
        print 'Parameterizing with: ', alpha, beta, gamma, theta 

        # The actual work:  
        S = OUSinusoidalSimulator(alpha, beta, gamma, theta);
        
        start = time.clock()
        pyTs         = S._pysimulate(N_spikes, dt=1e-4)
        stop = time.clock()
        print 'Py time = ', stop-start
        
        start = time.clock()
        cTs         = S._csimulate(N_spikes, dt=1e-4)
        stop = time.clock()
        print 'C time = ', stop-start
        
        figure()
        subplot(211)
        hist(diff(pyTs), normed=1)
        subplot(212)
        hist(diff(cTs), normed=1) 

def latexParamters():
    theta = 1.0
    print r'Regime Name & $\a$ & $\b$ & $\g$ \\ \hline'
    for params, tag in zip([    [1.5-.1, .3, .1 * sqrt(1. + theta**2),     theta],
                                [.5,     .3, .5* sqrt(1. + theta**2),      theta],
                                [.4,     .3, .4* sqrt(1. + theta**2),      theta],
                                [.1,     .3, (1.5-.1)*sqrt(1. + theta**2), theta]],
                        ['Supra-Threshold', 'Critical', 'Sub Threshold','Super Sinusoidal']):
        print tag + r'&%.2f&%.2f&%.2f \\' %(params[0],params[1],params[2])
        
    print r'''\end{tabular}
              \caption{Example $\abg$ parameters for the different regimes, given $\th = %.1f$}
        '''%theta
        
def rename_stst():
    import subprocess
    p = subprocess.Popen("cd " + RESULTS_DIR +"; ls *.st.st", stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    output = output.split('\n')
    
    import os
    
    for line in output:
        src = os.path.join(RESULTS_DIR, line)
        dst = os.path.join(RESULTS_DIR, line.replace('st.st', 'st'))
#        print ' from ', src , ' to ' , dst 
        os.rename(src, dst)
        
        
#    import os
    
    
        
if __name__ == '__main__':
    from pylab import *
    import time

#    latexParamters()
#    CvsPySimulator()

#    thetaSimulator(N_spikes=1000, N_trains = 100)
    
#    start = time.clock()
    batchSimulator(N_spikes=4, N_trains = 1, overwrite=True)
#    batchSimulator(N_spikes=1000, N_trains = 1, overwrite=True)
#    print time.clock() - start


#    sinusoidal_spike_train(N_spikes=1000, fig_tag = 'SuperT', 
#                           params = [1.5, .3, 1.0, 2.0], dt =1e-4)
#    sinusoidal_spike_train(N_spikes = 200, fig_tag = 'Crit', 
#                           params = [.55, .5, .55, 2.0])
#    sinusoidal_spike_train(N_spikes = 200, fig_tag = 'SubT', 
#                           params = [.4, .3, .4, 2.0])

#    sinusoidal_spike_train(N_spikes=100, fig_tag = 'SuperSin', 
#                           params = [.1, .3, 2.0, 2.0], dt =1e-4)
    
#    visualizeDistributions('sinusoidal_spike_train_N=1000_superSin_12', 'SuperSin_Distributions')
#    visualizeDistributions('sinusoidal_spike_train_T=5000_superT_14.path', 'SuperT_Distributions')
    
    show()
        