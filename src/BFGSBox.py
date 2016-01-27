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
from InitBox import guesstimate5pts, initialize5


class Estimator():
    def __init__(self, Solver, binnedTrain, verbose = False):
        self._abg_current = None
        self._Solver = Solver
        self._binnedTrain = binnedTrain
        self._G = None
        self._dGdp = None
        self._verbose = verbose
        
    def func(self, abg):
            if not all(self._abg_current == abg):
                self._abg_current = abg;
                self._dGdp = None
                
                self._resetSolver()
                self._computeFLG() 
                   
                if self._verbose:
                    print 'abg = ', abg
                    print self._G

            return self._G  
                
    def dfunc(self, abg):
            if not all(self._abg_current == abg): 
                self._abg_current = abg;

                self._resetSolver()
                self._computeFLG()
                self._computedG()
            
                if self._verbose:
                    print 'abg = ', abg
                    print self._dGdp
                
            elif (None == self._dGdp):
                self._computedG()
                if self._verbose:
                    print 'abg = ', abg
                    print self._dGdp
            
            return self._dGdp

    def _resetSolver(self):
        abg = self._abg_current
        max_speed = abg[0] + abs(abg[2]) - self._Solver._xs[0];
        dx = abg[1] / max_speed / 1e2
        dt = FPMultiPhiSolver.calculate_dt(dx, max_speed, 1e2)
        
        Tf = self._Solver.getTf()
        self._Solver.rediscretize(dx, dt, Tf)
        
    def _computeFLG(self):
        Ss = self._Solver.transformSurvivorData(self._binnedTrain)
        self._Fs = self._Solver.solve(self._abg_current, visualize=False)
        self._Ls = self._Fs[:,:,-1] - Ss 
        
        self._G = .5*sum(self._Ls*self._Ls)*self._Solver._dt
    
    def _computedG(self):
        Nus = self._Solver.solveAdjointUpwinded(self._abg_current, self._Ls)
        self._dGdp = self._Solver.estimateParameterGradient(self._abg_current, self._Fs, Nus)



def BFGSEstimator():
    from scipy.optimize import fmin_bfgs
    
    phis =  linspace(.05, .95, 10)
    phi_omit = None
    
    file_name = 'sinusoidal_spike_train_T=20000_subT_13.path'
    
    binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
    binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh = 32.)
    print 'N_bins = ', len(binnedTrain.bins.keys())

    Tf = binnedTrain.getTf()
    print 'Tf = ', Tf
   
    solver_phis = binnedTrain.bins.keys();
    theta = binnedTrain.theta
    x_min = -.5;
    
    S = FPMultiPhiSolver(theta, solver_phis,
                     .1, .1,
                     Tf, X_MIN = x_min)
    
    ps = binnedTrain._Path._params
    abg_true = array([ps._alpha, ps._beta, ps._gamma]);
    print 'true = ', abg_true
#    abg_init = abg_true + .5*abg_true*randn(3);
    abg_init = [ 0.37345572 , 0.32178958 , 0.31556914]
    abg_init = [ 0.37366108 , 0.32516912,  0.31569902]
    print 'init = ', abg_init
    
    lE = Estimator(S, binnedTrain)
    
#    abg_est = fmin_bfgs(lE.func, abg_init, lE.dfunc,  gtol = 1e-6*binnedTrain.getTf(), maxiter= 128, full_output = 0)
    abg_est, fopt, gopt, Bopt, func_calls, grad_calls, warnflag  = fmin_bfgs(lE.func, abg_init,
                                                     lE.dfunc,  gtol = 1e-5*binnedTrain.getTf(), maxiter=32, full_output = 1)
    
    return abg_est


def BFGSGradedEstimator():
    from scipy.optimize import fmin_bfgs
    
    print 'GradedEstimator'
    
    N_phi = 10;
    print 'N_phi = ', N_phi
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
    phi_omit = None

    for file_name in ['sinusoidal_spike_train_T=20000_subT_4.path',
                      'sinusoidal_spike_train_T=20000_subT_7.path',
                      'sinusoidal_spike_train_T=20000_subT_13.path']:
    
        print file_name
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
        binnedTrain.pruneBins(phi_omit, N_thresh = 32, T_thresh = 32.)
        abg_est = abs( initialize5(binnedTrain))
        
        print 'abg_init = ',abg_est
        
        for T_thresh, N_thresh, max_iters in zip([32/8., 32/4., 32/2., 32.],
                                                 [128, 128, 64, 32],
                                                 [32,24,16,8]):
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
            binnedTrain.pruneBins(phi_omit, N_thresh, T_thresh)
            Tf = binnedTrain.getTf()
            print 'Tf = ', Tf
            print 'N_bins = ', len(binnedTrain.bins.keys())
           
            solver_phis = binnedTrain.bins.keys();
            theta = binnedTrain.theta
            x_min = -.5;
            
            S = FPMultiPhiSolver(theta, solver_phis,
                             .1, .1,
                             Tf, X_MIN = x_min)
            
            lE = Estimator(S, binnedTrain, verbose = True)
            
        #    abg_est = fmin_bfgs(lE.func, abg_init, lE.dfunc,  gtol = 1e-6*binnedTrain.getTf(), maxiter= 128, full_output = 0)
            abg_est, fopt, gopt, Bopt, func_calls, grad_calls, warnflag  = fmin_bfgs(lE.func, abg_est,
                                                                                     lE.dfunc,  gtol = 1e-08*binnedTrain.getTf(), maxiter=max_iters, full_output = 1)
            
            print 'estimate gradient =', gopt
            print 'current_estimate = ', abg_est
        
        print 'final estimate = ', abg_est
        
                
if __name__ == '__main__':
    from pylab import *
    
    BFGSEstimator()
#    BFGSGradedEstimator()