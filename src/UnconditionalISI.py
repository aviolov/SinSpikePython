# -*- coding:utf-8 -*-
"""
Created on Apr 23, 2012

@author: alex
"""


from __future__ import division

from BinnedSpikeTrain import BinnedSpikeTrain, intervalStats, loadPath,\
    generateSDF
#from Simulator import Path, OUSinusoidalParams
from SpikeTrainSimulator import SpikeTrain, OUSinusoidalParams

from numpy import linspace, float, arange, sum
from numpy import sin, sqrt, ones_like, zeros_like, where, diff, pi, log, max , sign
from numpy import zeros, ones, array, c_, r_, float64, matrix, bmat, Inf, ceil, arange, empty, interp, dot, sqrt
from scipy.sparse import spdiags, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from numpy.linalg import eig
from copy import deepcopy

from core.numeric import outer

from FPMultiPhiSolver import FPMultiPhiSolver
#from Simulator import OUSinusoidalParams
#from FPSolver import FIGS_DIR
#from InitBox import initialize_right_2std
##from DataHarvester import regime_label

#RESULTS_DIR = '/home/alex/Workspaces/Python/OU_hitting_time/Results/FP/'
FIGS_DIR = '/home/alex/Workspaces/Latex/LIFEstimation/Figs/PhiAveraged'
import os
for D in [FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time


def obtainAnalyticDistros(gs,h,dt):
    d_analytic = ravel( dot(h.transpose(), gs) )
    iD_analytic =  r_[(1., 1. - cumsum(d_analytic)*dt)] 
    
    return d_analytic, iD_analytic

def obtainRenewalDensities(phis, abg, theta, Tf):
    xmin = FPMultiPhiSolver.calculate_xmin(Tf, abg)
    dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg, xmin, factor = 5.)
    S = FPMultiPhiSolver(theta, phis,
                             dx, dt, Tf, xmin);
      
    Fs = S.solve(abg)
    ts = S._ts;
            
    iG = squeeze(Fs[:, :,-1])
    
    g = -diff(iG)/ S._dt;
    
    t_mids = (S._ts[1:]+S._ts[:-1])/2.0
    
    return g, S._ts ,t_mids

def generateTransitionKernel(gs, ts, theta, phis, Tf):
    N_phi = len(phis)
    Psi = matrix(empty([N_phi,N_phi]));
    #NOTE that we've used the transpose of the usual Markov transition matrix!
    
    for col in xrange(N_phi):
        for row in xrange(N_phi):
            phi_from = phis[col]
            phi_to   = phis[row]
            
            gs_from = gs[col, :]
            
            dphi = phi_to - phi_from;
            T = 2*pi/theta;
            
            K_min = int( max([0, ceil( dphi / T )] ))
            K_max = int( floor( (Tf - dphi ) / T));
            sample_pts = dphi + T * arange(K_min, K_max+1)
                        
            if 0 < len(sample_pts): 
                Psi[row, col] = sum(interp(sample_pts, ts, gs_from))                
            else:
                #CAN THIS HAPPEN???
                Psi[row, col] = 0.
              
    #Regularize Psi: (make it Markovian st. probability is conserved...)
    for col in xrange(N_phi):  
        Psi[:, col] =  1./sum(Psi[:, col]) * Psi[:, col] 

    return Psi

    
def obtainStationaryDistribution(Psi):        
    ls, vs = eig(Psi)
    stable_ind = find(ls == max(ls));
#    print 'l_stable = ', ls[stable_ind]
    
    h = vs[:,stable_ind]

    print 'h_stable = ', h / sum(h)
        
    return h / sum(h)

def obtainEmpiricalDistros(Is, ts):
    iD_empirical, unique_Is = generateSDF(Is)
    
    iD_empirical =  interp(ts, r_[(.0,unique_Is)],
                                r_[(1., iD_empirical)])    
    d_empirical = -diff(iD_empirical)/ (ts[1] - ts[0]);
    
    return d_empirical, iD_empirical

            
def visualiziDistribution(sim_id = 'N=1000_crit_5', save_fig_name = ''):
    file_name = 'sinusoidal_spike_train_' + sim_id  
    save_fig_name = sim_id
    
    N_phi = 32
    print 'N_phi = ', N_phi
    nominal_phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi) 
    
    binnedTrain = BinnedSpikeTrain.initFromFile(file_name, nominal_phis)    
    Is = binnedTrain._spike_Is;
    alpha,beta,gamma,theta = binnedTrain._Train._params.getParams()
        
    Tf_observed = binnedTrain.getTf()
    print 'Tf_observed = ', Tf_observed
    
    Tf = min([5*2*pi/theta, max([2* 2*pi/theta, Tf_observed]) ])
    print 'Tf_computational = ', Tf
    
    physical_phis  = binnedTrain.bins.keys();
    print 'phis = ', physical_phis
    
    gs, ts, t_mids = obtainRenewalDensities(physical_phis, [alpha, beta,gamma], theta, Tf)
    
    d_empirical, iD_empirical = obtainEmpiricalDistros(Is, ts)
    
    Psi = generateTransitionKernel(gs, t_mids, theta, physical_phis, Tf)
    h = obtainStationaryDistribution(Psi)
    
    d_analytic, iD_analytic = obtainAnalyticDistros(gs,h, ts[1]-ts[0])

#    PLOT IT:
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['figure.subplot.top'] = .9
    mpl.rcParams['figure.subplot.wspace'] = .3
    
            
    figure()
    subplot(211)
    plot(t_mids, d_analytic,'b',label='Analytic')
    plot(t_mids, d_empirical,'r+', label='Data')
    title('Densities')
    legend();
    
    subplot(212)
    plot(ts, iD_analytic, 'b', label='Analytic')
    plot(ts, iD_empirical,'r+', label='Data')
    ylim((.0, 1.05));
    ylabel('$1 - G(t)$', fontsize = 24); xlabel('$t$', fontsize = 24)
    legend();
    title('Survivor Distributions')

    get_current_fig_manager().window.showMaximized()
    
    

    if '' != save_fig_name:    
        file_name = os.path.join(FIGS_DIR, save_fig_name+'.png')
        print 'saving to ', file_name
        savefig(file_name) 



if __name__ == '__main__':
    from pylab import *
    
    visualiziDistribution()
 
    
    show()
    