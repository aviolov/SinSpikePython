'''
Created on May 16, 2012

@author: alex
'''


from BinnedSpikeTrain import BinnedSpikeTrain, intervalStats
from Simulator import Path, OUSinusoidalParams

from FPMultiPhiSolver import FPMultiPhiSolver

from numpy import linspace, float, arange, sum, sort
from numpy import sin, sqrt, ones_like, zeros_like, where, diff, pi, log, max , sign, arctan, exp
from numpy import zeros, ones, array, r_, float64, matrix, bmat, Inf, ceil, arange, empty, interp, dot, sqrt, abs

from numpy.random import randn
from scipy.stats.distributions import norm

import time
from InitBox import initialize5



def GradedDriver():
    from scipy.optimize import fmin_bfgs
    
    N_phi = 10;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    print 'GradedEstimator'
    
    for file_name in ['sinusoidal_spike_train_T=20000_subT_3.path',
                      'sinusoidal_spike_train_T=20000_subT_8.path',
                      'sinusoidal_spike_train_T=20000_subT_13.path']:
    
        print file_name
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
        binnedTrain.pruneBins(None, N_thresh = 32, T_thresh = 32.)
        abg_est = abs( initialize5(binnedTrain))
        
        print 'abg_init = ',abg_est
        
        theta = binnedTrain.theta
        
        for T_thresh, N_thresh, max_iters in zip([32/8., 32/4., 32/2., 32.],
                                          [128, 128, 64, 32],
                                          [50,50,100,None]):
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            binnedTrain.pruneBins(None, N_thresh, T_thresh)
            print 'N_bins = ', len(binnedTrain.bins.keys())
        
            Tf = binnedTrain.getTf()
            print 'Tf = ', Tf 
            dx = .02; dt = FPMultiPhiSolver.calculate_dt(dx, 4., 10.)
        
            phis = binnedTrain.bins.keys();
            
            S = FPMultiPhiSolver(theta, phis, dx, dt,
                                 Tf, X_MIN = -.5)
            
            from scipy.optimize import fmin
            def func(abg):
                'Solve it:'
                Fs = S.solve(abg, visualize=False)[:,:,-1]
                Ss = S.transformSurvivorData(binnedTrain)
                Ls = Fs - Ss
        
                'Return'
                G = .5*sum(Ls*Ls)*S._dt 
                return G
    
            abg_est = fmin(func, abg_est, ftol = 1e-2*T_thresh, maxiter=max_iters)
            
            print 'current_estimate = ', abg_est
        
        print 'final estimate = ', abg_est
        
        
def GradedNMEstimator(file_name, phi_norms, abg_est, T_max, N_thresh_final):
        for T_thresh, N_thresh, max_iters in zip(array([1./8., 1./4., 1./2., 1.])*T_max,
                                                 array([2,2,2,1])*N_thresh_final,
                                                 [50,50,100,None]):
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            binnedTrain.pruneBins(None, N_thresh, T_thresh)
            print 'N_bins = ', len(binnedTrain.bins.keys())
        
            Tf = binnedTrain.getTf()
            print 'Tf = ', Tf 
            dx = .02; dt = FPMultiPhiSolver.calculate_dt(dx, 5., 10.)
        
            phis = binnedTrain.bins.keys();
            theta = binnedTrain.theta
            
            S = FPMultiPhiSolver(theta, phis, dx, dt,
                                 Tf, X_MIN = -.5)
            from scipy.optimize import fmin
            def func(abg):
                'Solve it:'
                Fs = S.solve(abg, visualize=False)[:,:,-1]
                Ss = S.transformSurvivorData(binnedTrain)
                Ls = Fs - Ss
        
                'Return'
                G = .5*sum(Ls*Ls)*S._dt 
                return G
    
            abg_est = fmin(func, abg_est, ftol = 1e-2*T_thresh, maxiter=max_iters)
            
            print 'current_estimate = ', abg_est
        
        return abg_est
        
        
from DataHarvester import DataHarvester
from AdjointEstimator import FortetEstimator

def BatchGradedNMEstimator():
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_T='

    D = DataHarvester('GradedNMx16', 'GradedNM_SubTx16')
    N_thresh = 32
    for regime_name, T_sim, T_thresh in zip(['subT'],
                                                       [20000],
                                                       [32.]):
        regime_label = base_name + str(T_sim)+ '_' + regime_name
            
        for sample_id in xrange(4,17):
            file_name = regime_label + '_' + str(sample_id) + '.path'
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            ps = binnedTrain._Path._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            D.setRegime(regime_name,abg_true, T_sim)
            
            phi_omit = None
            binnedTrain.pruneBins(phi_omit, N_thresh = N_thresh, T_thresh=T_thresh)
            Tf = binnedTrain.getTf()
            D.addSample(sample_id, Tf, binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
        
            start = time.clock()
            abg_init = initialize5(binnedTrain)
            finish = time.clock()
            D.addEstimate(sample_id, 'Initializer', abg_init, finish-start) 
                    
            start = time.clock()
            abg_est = GradedNMEstimator(file_name, phi_norms, abg_init, T_thresh, N_thresh)
            finish = time.clock()
            D.addEstimate(sample_id, 'Graded_Nelder-Mead', abg_est, finish-start) 
            
            start = time.clock()
            abg_est = FortetEstimator(binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'Fortet', abg_est, finish-start)
        
    D.closeFile() 
   
    print 'batch time = ', (time.clock() - batch_start) / 3600.0, ' hrs'

        
if __name__ == '__main__':
    from pylab import *
    
#    GradedDriver()
    BatchGradedNMEstimator()