# -*- coding:utf-8 -*-
"""
Created on Jun 21, 2012

@author: alex
"""


from BinnedSpikeTrain import BinnedSpikeTrain, intervalStats
#from Simulator import Path, OUSinusoidalParams
from SpikeTrainSimulator import SpikeTrain, OUSinusoidalParams

from FPMultiPhiSolver import FPMultiPhiSolver

from numpy import linspace, float, arange, sum, sort
from numpy import sin, sqrt, ones_like, zeros_like, where, diff, pi, log, max , sign, arctan, exp
from numpy import zeros, ones, array, r_, float64, matrix, bmat, Inf, ceil, arange, empty, interp, dot, sqrt, abs

from numpy.random import randn, rand
from scipy.stats.distributions import norm

import time
from InitBox import initialize5, initialize_right_2std
from AdjointEstimator import NMEstimator

def TaucharEstimator(S, binnedTrain, abg_init):
    from scipy.optimize import fmin
    
    print 'NelderMead method: '
    
    def func(abgt):
        'Solve it:'
        abg = abgt[:3]
        xmin = FPMultiPhiSolver.calculate_xmin(S.getTf(), abg)
        dt = FPMultiPhiSolver.calculate_dt(S._dx, abg, xmin)
        S.rediscretize(S._dx, dt, S.getTf(), xmin)
        
        Fs = S.solve_tau(abgt, visualize=False)[:,:,-1]
        Ss = S.transformSurvivorData(binnedTrain)
        Ls = Fs - Ss

        'Return '
        G = .5*sum(Ls*Ls)*S._dt 
        
        return G
    
    abgt_est = fmin(func, abg_init, ftol = 1e-1)
    
    return abgt_est


def estimateTau(regime = 'crit', number=11,
                     N_thresh = 64, T_thresh = 16. ):
    file_name = 'sinusoidal_spike_train_N=1000_' + regime + '_' + str(number)
    print file_name
    
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
    binnedTrain.pruneBins(None, N_thresh, T_thresh)
    print 'N_bins = ', len(binnedTrain.bins.keys())
    
    Tf = binnedTrain.getTf()
    print 'Tf = ', Tf 
    abg_init = initialize_right_2std(binnedTrain)

    
    phis = binnedTrain.bins.keys();
    theta = binnedTrain.theta

    dx = .02; dt = FPMultiPhiSolver.calculate_dt(dx, abg_init,x_min= -1.0)
                
    S = FPMultiPhiSolver(theta, phis, dx, dt,
                       Tf, X_min = -2.)
        
    abgt_init = [abg_init[0],
                 abg_init[1],
                 abg_init[2],                     
                 .5]
    
    print 'abgt_init = ', abgt_init
    
    start = time.clock()
    abgt_est = TaucharEstimator(S, binnedTrain, abgt_init)
    finish = time.clock()
    
    print 'abgt_est = ', abgt_est
    print 'compute time = ', finish-start
    
    print 'No tau comparison:'
    start = time.clock()
    abg_est = NMEstimator(S, binnedTrain, abg_init)
    finish = time.clock()
    
    print 'abg_est = ', abg_est
    print 'compute time = ', finish-start
    
    return abgt_est
    
if __name__ == '__main__':
    estimateTau('crit', 14)
    
    '''
    sinusoidal_spike_train_N=1000_superT_9
N_phi =  20
N_bins =  9
Tf =  4.35250000014
abgt_init =  [1.5665212743326333, 0.2531191533999802, 1.0414544417151581, 0.5]
NelderMead method: 
Optimization terminated successfully.
         Current function value: 0.007411
         Iterations: 66
         Function evaluations: 139
abgt_est =  [ 1.87489463  0.31415139  0.91759959  0.61929902]
compute time =  255.87
No tau comparison:
NelderMead method: 
Optimization terminated successfully.
         Current function value: 0.008861
         Iterations: 42
         Function evaluations: 104
abg_est =  [ 1.46816976  0.32715329  0.97621878  0.4826113 ]
compute time =  166.39

sinusoidal_spike_train_N=1000_superT_14
N_phi =  20
N_bins =  8
Tf =  4.3884999989
abgt_init =  [1.6139515643635614, 0.2643594500570613, 0.93894470173428735, 0.5]
NelderMead method: 
Optimization terminated successfully.
         Current function value: 0.007448
         Iterations: 46
         Function evaluations: 116
abgt_est =  [ 1.90398768  0.26906198  0.91009779  0.61072449]
compute time =  197.0
No tau comparison:
NelderMead method: 
Optimization terminated successfully.
         Current function value: 0.008160
         Iterations: 32
         Function evaluations: 78
abg_est =  [ 1.46563996  0.28105817  0.96848499]
compute time =  114.13

sinusoidal_spike_train_N=1000_crit_14
N_phi =  20
N_bins =  8
Tf =  13.3926999966
abgt_init =  [0.76468187791530928, 0.51273997672497584, 0.37479463090509157, 0.5]
NelderMead method: 
Optimization terminated successfully.
         Current function value: 0.064158
         Iterations: 54
         Function evaluations: 121
abgt_est =  [ 0.94657025  0.52626892  0.55184594  0.53410389]
compute time =  727.59
No tau comparison:
NelderMead method: 
Optimization terminated successfully.
         Current function value: 0.050356
         Iterations: 61
         Function evaluations: 123
abg_est =  [ 0.49556242  0.53317567  0.53075204]
compute time =  860.73
;;;'''