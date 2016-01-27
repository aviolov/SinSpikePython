# -*- coding:utf-8 -*-
"""
Created on Jul 24, 2012

@author: alex

:: We try to implement a hybrid method that uses both the Fokker-Plank and the Fortet models
together!!! (so far unsuccessful)
"""


from __future__ import division

from BinnedSpikeTrain import BinnedSpikeTrain, intervalStats, loadPath
from SpikeTrainSimulator import SpikeTrain, OUSinusoidalParams

from numpy import linspace, float, arange, sum, arctan, exp
from numpy import sin, sqrt, ones_like, zeros_like, where, diff, pi, log, max , sign
from numpy import zeros, ones, array, c_, r_, float64, matrix, bmat, Inf, ceil, arange, empty, interp, dot, sqrt
from scipy.sparse import spdiags, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve

from FPMultiPhiSolver import FPMultiPhiSolver

from InitBox import initialize_right_2std

#RESULTS_DIR = '/home/alex/Workspaces/Python/OU_hitting_time/Results/FP/'
#FIGS_DIR = '/home/alex/Workspaces/Latex/LIFEstimation/Figs/FP'
#import os
#for D in [RESULTS_DIR]:
#    if not os.path.exists(D):
#        os.mkdir(D)

import time
        
        
def FortetLineEstimator(binnedTrain, abg_start, dG_normalized, dp_tol):
    ''' dG is assumed pointing in the gradient direction and so gradient descent moves in 
    the direction -dG!!!'''
    MIN_BETA_GAMMA = 1e-3;
    bins = binnedTrain.bins;
    phis = bins.keys()
    N_phi = len(phis)
    
    def getMovingThreshold(a,g, phi):
        theta = binnedTrain.theta
        psi = arctan(theta)
        mvt = lambda ts: 1. - ( a*(1 - exp(-ts)) + \
                                g / sqrt(1+theta*theta) * ( sin ( theta *  ( ts + phi) - psi) \
                                                    -exp(-ts)*sin(phi*theta - psi) ))

        return mvt

    def getMax_dp(abg, dG_norm):
        dp_max = 1.0
        
        for ptv_indx in [1,2]:
            if dG_norm[ptv_indx] >.0:
                dp_max = min(dp_max,  (abg[ptv_indx] - MIN_BETA_GAMMA) / dG_norm[ptv_indx])
        
        return dp_max
        

    from scipy.stats.distributions import norm
    def loss_function(dp):
#            #SIMPLE: Penalize negative a's, we want a positive, b/c for a<0, the algorithm is different:
#            if min(abg)<.0 or max(abg) > 5.:
#                return 1e6#
            abg = abg_start - dp*dG_normalized
            error = .0;
            
            for (phi_m, phi_idx) in zip(phis, xrange(N_phi)):
                Is = bins[phi_m]['Is']
                uniqueIs = bins[phi_m]['unique_Is']
                 
                a,b,g = abg[0], abg[1], abg[2]
                movingThreshold = getMovingThreshold(a,g, phi_m)
                
                LHS_numerator = movingThreshold(uniqueIs) *sqrt(2.)
                LHS_denominator = b * sqrt(1 - exp(-2*uniqueIs))
                LHS = 1 -  norm.cdf(LHS_numerator / LHS_denominator)
                
                RHS = zeros_like(LHS)
                N  = len(Is)
                for rhs_idx in xrange(len(uniqueIs)):
                    t = uniqueIs[rhs_idx]
                    lIs = Is[Is<t]
                    taus = t - lIs;
                    
                    numerator = (movingThreshold(t) - movingThreshold(lIs)* exp(-taus)) * sqrt(2.)
                    denominator = b *  sqrt(1. - exp(-2*taus))
                    RHS[rhs_idx] = sum(1. - norm.cdf(numerator/denominator)) / N
                
#                error += sum(abs(LHS - RHS));
                error += sum((LHS - RHS)**2);
#                error += max(abs(LHS - RHS))

            return error

    from scipy.optimize import fminbound
    dp_max = getMax_dp(abg_start, dG_normalized)
    
    return fminbound(loss_function, x1 = .0, x2 =dp_max, xtol=dp_tol/2.0, maxfun =32,
                   disp=1);
    

def MixedEstimator(abg_init, binnedTrain, dp_tol = 1e-2):
    phis = binnedTrain.bins.keys();
    theta = binnedTrain.theta

    dp = dp_tol*2.0;
    abg = abg_init
    
    while dp > dp_tol:
        Tf = binnedTrain.getTf()
   
        xmin = FPMultiPhiSolver.calculate_xmin(Tf, abg)
        dx = FPMultiPhiSolver.calculate_dx(abg, xmin)
        dt = FPMultiPhiSolver.calculate_dt(dx, abg, xmin, factor = 8.)

        S = FPMultiPhiSolver(theta, phis,
                             dx, dt, Tf, xmin)

        Fs = S.solve(abg, visualize=False)
        Ss = S.transformSurvivorData(binnedTrain)
        Ls = Fs[:,:,-1] - Ss
        Nus = S.solveAdjoint(abg, Ls)
    
        dGdp = S.estimateParameterGradient(abg, Fs, Nus)

        from numpy.linalg.linalg import norm
        
        dG_normalized = dGdp/ norm(dGdp) 
        
        dp = FortetLineEstimator(binnedTrain, abg, dG_normalized, dp_tol)
        
        abg = abg - dp*dG_normalized

        print 'dG = ', dG_normalized
        print 'dp = ', dp
        print 'abg = (%.3g, %.3g, %.3g)'%(abg[0],abg[1],abg[2])
        print '-'

    return abg
        
    
def MixedDriver():
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
    
#    file_name = 'sinusoidal_spike_train_N=1000_superT_13'
    file_name = 'sinusoidal_spike_train_N=1000_subT_1'
#    file_name = 'sinusoidal_spike_train_N=1000_crit_5'

#    intervalStats(file_name)
    
    binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)

    phi_omit = None
#    phi_omit = r_[(linspace(.15, .45, 4),
#                   linspace(.55,.95, 5) )]  *2*pi/ binnedTrain.theta
    binnedTrain.pruneBins(phi_omit, N_thresh = 80, T_thresh= 16.)
    print 'N_bins = ', len(binnedTrain.bins.keys())
    
    Tf = binnedTrain.getTf() #/ 2.
    print 'Tf = ', Tf

    params = binnedTrain._Train._params
    abg_true = (params._alpha, params._beta, params._gamma)
    print 'true = ',     abg_true

    abg_init = initialize_right_2std(binnedTrain)
    print 'init = ',     abg_init

    
    start = time.clock()
    abg_est = MixedEstimator(abg_init, binnedTrain)
    finish = time.clock()
    print 'Mixed est = ', abg_est
    mixed_time = finish-start
    print 'Mixed time = ', mixed_time
    
    from AdjointEstimator import FortetEstimator 
    start = time.clock()
    abg_est = FortetEstimator(binnedTrain, abg_est)
    finish = time.clock()
    print 'Mixed+Fortet est = ', abg_est
    print 'MIxed+Fortet time = ', finish-start + mixed_time

    #Compare with straight up Fortet:
    start = time.clock()
    abg_est = FortetEstimator(binnedTrain, abg_init)
    finish = time.clock()
    print 'Fortet est = ', abg_est
    print 'Fortet time = ', finish-start
           
           
if __name__ == '__main__':
    from pylab import *
    
    MixedDriver()
    
    show()
        