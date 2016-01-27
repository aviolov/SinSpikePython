'''
Created on May 3, 2012

@author: alex
'''
from __future__ import division

from BinnedSpikeTrain import BinnedSpikeTrain, intervalStats
#from Simulator import Path, OUSinusoidalParams
from SpikeTrainSimulator import SpikeTrain, OUSinusoidalParams

from FPMultiPhiSolver import FPMultiPhiSolver

from numpy import abs, linspace, float, float64,arange, sum, sort, amax, ndarray, tile, iterable
from numpy import sin, sqrt, ones_like, zeros_like, where, diff, pi, log, max , sign, arctan, exp
from numpy import zeros, ones, array, r_, float64, matrix, bmat, Inf, ceil, arange, empty, interp, dot, sqrt, abs

from numpy.random import randn
from scipy.stats.distributions import norm

import time
from InitBox import guesstimate5pts, initialize5


def TNCEstimator(S, binnedTrain, abg_init):
    from scipy.optimize import fmin_tnc
    
    print 'TNC method: '
    
    lbounds = ((.01, 5.),
               (.01, 5.),
               (.0 , 5.)  );
              
    def func(abg):
        'Solve it:'
        Fs = S.solve(abg, visualize=False)
        Ss = S.transformSurvivorData(binnedTrain)
        Ls = Fs[:,:,-1] - Ss
        Nus = S.solveAdjoint(abg, Ls)

        'Return '
        G = .5*sum(Ls*Ls)*S._dt 
        dGdp = S.estimateParameterGradient(abg, Fs, Nus)
        
        return G, dGdp

    
    abg_est, nfeval, rc_code = fmin_tnc(func, abg_init, 
                                        ftol = 1e-4, maxfun = 50)
    
    return abg_est

class Estimator():
    def __init__(self, Solver, Ss):
        self._abg_current = (.0, .0, .0)
        self._Solver = Solver
        self._Ss = Ss
        
#            self._G = None
#            self._dGdp = None
        
    def func(self, abg):
            if not all(self._abg_current == abg):
                self._abg_current = abg;
                self._dGdp = None
                self._computeFLG() 
                
            return self._G  
                
    def dfunc(self, abg):
            if not all(self._abg_current == abg): 
                self._abg_current = abg;
                self._computeFLG()
                self._computedG()
            
#                print 'abg = ', abg
#                print self._dGdp
                
            elif (None == self._dGdp):
                self._computedG()
                
#                print 'abg = ', abg
#                print self._dGdp
            
            return self._dGdp

    def _computeFLG(self):
        self._Fs = self._Solver.solve(self._abg_current, visualize=False)
        self._Ls = self._Fs[:,:,-1] - self._Ss
        
        self._G = .5*sum(self._Ls*self._Ls)*self._Solver._dt
    
    def _computedG(self):
        Nus = self._Solver.solveAdjoint(self._abg_current, self._Ls)
        self._dGdp = self._Solver.estimateParameterGradient(self._abg_current, self._Fs, Nus)


def BFGSEstimator(Solver, binnedTrain, abg_init, max_iters = 16):
    from scipy.optimize import fmin_bfgs
    
    print 'BFGS Method: '
    
    Ss = Solver.transformSurvivorData(binnedTrain)
    
    lE = Estimator(Solver, Ss)
    
#    abg_est = fmin_bfgs(lE.func, abg_init, lE.dfunc,  gtol = 1e-6*binnedTrain.getTf(), maxiter= 128, full_output = 0)
    abg_est, fopt, gopt, Bopt, func_calls, grad_calls, warnflag  = fmin_bfgs(lE.func, abg_init,
                                                     lE.dfunc,  gtol = 1e-5*binnedTrain.getTf(), maxiter=max_iters, full_output = 1)
    print 'gopt = ', gopt
    return abg_est

def CGEstimator(Solver, binnedTrain, abg_init):
    from scipy.optimize import fmin_cg
    
    print 'CG Method: '
    
    Ss = Solver.transformSurvivorData(binnedTrain)
    
    lE = Estimator(Solver, Ss)
    
    abg_est = fmin_cg(lE.func, abg_init, lE.dfunc,
                        gtol = 1e-1, maxiter = 20)
    
    return abg_est

def NMEstimator(S, binnedTrain, abg_init):
    from scipy.optimize import fmin
    
    print 'FP - NelderMead: '
    
#    bins = binnedTrain.bins;
#    ts = S._ts;
    
    def func(abg):
        xmin = FPMultiPhiSolver.calculate_xmin(S.getTf(), abg)
        dt = FPMultiPhiSolver.calculate_dt(S._dx, abg, xmin)
        S.rediscretize(S._dx, dt, S.getTf(), xmin)
        
        'Solve it:'
        Fs = S.solve(abg, visualize=False)[:,:,-1]
        Ss = S.transformSurvivorData(binnedTrain)
        Ls = Fs - Ss

        'Return '
        G = .5*sum(Ls*Ls)*S._dt 
        
#        G = .0;
#        for phi, phi_idx in zip(S._phis, xrange(S._num_phis() )):
#            unique_Is = bins[phi]['unique_Is']
#            SDF = bins[phi]['SDF']
#            Ls = SDF - interp(unique_Is, ts, Fs[phi_idx,:])
#            G += sum(Ls*Ls) 
        
        return G
    
    abg_est = fmin(func, abg_init, 
                   ftol = 1e-1)
    
    return abg_est

def cNMEstimator(S, binnedTrain, abg_init):
    from scipy.optimize import fmin
    
    print 'FP-C - NelderMead method: '
    
#    bins = binnedTrain.bins;
#    ts = S._ts;
    
    def func(abg):
        xmin = FPMultiPhiSolver.calculate_xmin(S.getTf(), abg)
        dt = FPMultiPhiSolver.calculate_dt(S._dx, abg, xmin)
        S.rediscretize(S._dx, dt, S.getTf(), xmin)
        
        'Solve it:'
        Fss = S.c_solve(abg);
        Fs = Fss[:,:,-1];
        Ss = S.transformSurvivorData(binnedTrain)
        Ls = Fs - Ss
        
        'Return '
        G = .5*sum(Ls*Ls)*S._dt 
        
        print 'abg = ', abg, '; G=', G
#        G = .0;
#        for phi, phi_idx in zip(S._phis, xrange(S._num_phis() )):
#            unique_Is = bins[phi]['unique_Is']
#            SDF = bins[phi]['SDF']
#            Ls = SDF - interp(unique_Is, ts, Fs[phi_idx,:])
#            G += sum(Ls*Ls) 
        
        del Fss, Fs, Ss, Ls;
        
        return G
    
#    //TODO: What is the correct ftol for this!
    abg_est = fmin(func, abg_init, maxiter = 100,
                   ftol = 1e-1)
    
    return abg_est

def FPL2Estimator(S,binnedTrain, abg_init):
    from scipy.optimize import fmin
    
    print 'FP-C - NelderMead method (weighted): '
    
    weight_vector = [];
    for phi in S._phis:
        lbin = binnedTrain.bins[phi]
        weight_vector.append( len(lbin['Is']) )
    
    weight_vector = array(weight_vector)
    
    def func(abg):
        xmin = FPMultiPhiSolver.calculate_xmin(S.getTf(), abg)
        dt = FPMultiPhiSolver.calculate_dt(S._dx, abg, xmin)
        S.rediscretize(S._dx, dt, S.getTf(), xmin)
        
        'Solve it:'
        Fss = S.c_solve(abg)
        Fs = Fss[:,:,-1]
        Ss = S.transformSurvivorData(binnedTrain)
        Ls = Fs - Ss

        'Return '
        G = .5*sum( dot(weight_vector, Ls*Ls) )*S._dt / sum(weight_vector)
        
#        'clean up'
#        del Fss, Fs, Ss, Ls;
        
        print 'abg = ', abg, '; G=', G
        return G
    
#    //TODO: What is the correct ftol for this!
    abg_est = fmin(func, abg_init, maxiter = 100,
                         ftol = 5e-2)
    
    return abg_est


def FPSupEstimator(S,binnedTrain, abg_init, verbose = False, ftol=None):
    from scipy.optimize import fmin
    
    print 'FP-C - NelderMead, Sup Metric, weighted: '
    
    weight_vector = [];
    for phi in S._phis:
        lbin = binnedTrain.bins[phi]
        weight_vector.append( len(lbin['Is']) )
    
    weight_vector = array(weight_vector)
    
    def func(abg):
        xmin = FPMultiPhiSolver.calculate_xmin(S.getTf(), abg, S._theta)
        dt = FPMultiPhiSolver.calculate_dt(S._dx, abg, xmin)
        S.rediscretize(S._dx, dt, S.getTf(), xmin)
        
        'Solve it:'
        Fss = S.c_solve(abg)
        Fs = Fss[:,:,-1]
        Ss = S.transformSurvivorData(binnedTrain)
        lSups = amax(abs(Fs - Ss) , axis = 1)

        'Return '
        G = dot(weight_vector, lSups)
        
        if verbose:
            print 'abg = ', abg, '; G=', G
        return G
    
#    //TODO: What is the correct ftol for this!
    if None == ftol:
        ftol = 1e-3 * binnedTrain.getSpikeCount()
        print 'ftol = ', ftol
    abg_est, fopt, iter, funcalls, warnflag, allvecs = fmin(func, abg_init,
                                                            maxiter = 200, 
                                                            xtol = 1e-2,  ftol = ftol,
                                                            disp =1, full_output = True, retall = 1)
    
    return abg_est, warnflag

def COBYLAEstimator(S, binnedTrain, abg_init):
    from scipy.optimize import fmin_cobyla
    
    print 'COBYLA method: '

    constraints = [lambda p:   p[0]- .01,
                   lambda p:   8. - p[0],
                    lambda p:  p[1] - .01,
                    lambda p:  5. - p[1],
                    lambda p:  p[2] - .01,
                    lambda p:  8. - p[2]]
    
#    lbounds = ((.01, 5.),
#               (.01, 5.),
#               (.0 , 5.)  );
#              
    def func(abg):
        'Solve it:'
        Fs = S.solve(abg, visualize=False)
        Ss = S.transformSurvivorData(binnedTrain)
        Ls = Fs[:,:,-1] - Ss

        'Return '
        G = .5*sum(Ls*Ls)*S._dt 
        
        return G
    
    abg_est = fmin_cobyla(func, abg_init, constraints, disp=0);   
        
    return abg_est
    

from DataHarvester import DataHarvester

def BatchAdjointEstimator():
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_T='

    D = DataHarvester('Live2x16')
    for regime_name, T_sim, T_thresh in zip(['superT', 'subT', 'crit', 'superSin'],
                                                       [5000 , 20000, 5000, 5000],
                                                       [4., 32, 16., 16.]):
#    D = DataHarvester('Live2x2')
#    for regime_name, T_sim, T_thresh in zip(['superT', 'crit'],
#                                                       [5000 , 5000],
#                                                       [4., 16.]): 

        regime_label = base_name + str(T_sim)+ '_' + regime_name
            
        for sample_id in xrange(8,17):
            file_name = regime_label + '_' + str(sample_id) + '.path'
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            D.setRegime(regime_name,abg_true, T_sim)
            
            phi_omit = None
            binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh=T_thresh)
            Tf = binnedTrain.getTf()
            D.addSample(sample_id, Tf, binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
             
            dx = .05; dt = FPMultiPhiSolver.calculate_dt(dx, 5., 2.)
        
            phis = binnedTrain.bins.keys();
            theta = binnedTrain.theta
            
            S = FPMultiPhiSolver(theta, phis,
                                 dx, dt,
                                 Tf)
        
            start = time.clock()
            abg_init = initialize5(binnedTrain)
            finish = time.clock()
            D.addEstimate(sample_id, 'Initializer', abg_init, finish-start) 
                    
            start = time.clock()
            abg_est = NMEstimator(S, binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'Nelder-Mead', abg_est, finish-start) 
            
            start = time.clock()
            abg_est = BFGSEstimator(S, binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'BFGS', abg_est, finish-start) 
            
            start = time.clock()
            abg_est = FortetEstimator(binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'Fortet', abg_est, finish-start)
        
    D.closeFile() 
   
    print 'batch time = ', (time.clock() - batch_start) / 3600.0, ' hrs'


def BFGSItersComparison():
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_T='

    D = DataHarvester('BFGS_Iters')
    for regime_name, T_sim, T_thresh in zip(['crit', 'superSin'],
                                                       [5000, 5000],
                                                       [16., 16.]):

        regime_label = base_name + str(T_sim)+ '_' + regime_name
            
        for sample_id in xrange(1,4):
            file_name = regime_label + '_' + str(sample_id) + '.path'
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            D.setRegime(regime_name,abg_true, T_sim)
            
            phi_omit = None
            binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh=T_thresh)
            Tf = binnedTrain.getTf()
            D.addSample(sample_id, Tf, binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
             
            dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, 5., 2.)
        
            phis = binnedTrain.bins.keys();
            theta = binnedTrain.theta
            
            S = FPMultiPhiSolver(theta, phis,
                                 dx, dt,
                                 Tf, X_MIN = -2.0)
        
            start = time.clock()
            abg_init = initialize5(binnedTrain)
            finish = time.clock()
            D.addEstimate(sample_id, 'Initializer', abg_init, finish-start) 
                    
            start = time.clock()
            abg_est = BFGSEstimator(S, binnedTrain, abg_init, max_iters = 8)
            finish = time.clock()
            D.addEstimate(sample_id, 'BFGS_8', abg_est, finish-start) 
            
            start = time.clock()
            abg_est = BFGSEstimator(S, binnedTrain, abg_est,max_iters = 8)
            finish = time.clock()
            D.addEstimate(sample_id, 'BFGS_16', abg_est, finish-start)
            
            start = time.clock()
            abg_est = BFGSEstimator(S, binnedTrain, abg_est,max_iters = 8)
            finish = time.clock()
            D.addEstimate(sample_id, 'BFGS_24', abg_est, finish-start)
            
    D.closeFile() 
   
    print 'batch time = ', (time.clock() - batch_start) / 3600.0, ' hrs'


def NelderMeadSubTEstimator():
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_T='

    D = DataHarvester('SubT_NMx16_refined_sim_dt')
    for regime_name, T_sim, T_thresh in zip(['subT'],
                                           [20000],
                                           [32.]):

        regime_label = base_name + str(T_sim)+ '_' + regime_name
            
        for sample_id in xrange(1,17):
            file_name = regime_label + '_' + str(sample_id) + '.path'
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            D.setRegime(regime_name,abg_true, T_sim)
            
            phi_omit = None
            binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh=T_thresh)
            Tf = binnedTrain.getTf()
            D.addSample(sample_id, Tf, binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
             
            dx = .04; dt = FPMultiPhiSolver.calculate_dt(dx, 4., 2.)
        
            phis = binnedTrain.bins.keys();
            theta = binnedTrain.theta
            
            S = FPMultiPhiSolver(theta, phis,
                                 dx, dt,
                                 Tf, X_MIN = -.5)
        
            start = time.clock()
            abg_init = initialize5(binnedTrain)
            finish = time.clock()
            D.addEstimate(sample_id, 'Initializer', abg_init, finish-start) 

            abg_init = abs(abg_init)            
            start = time.clock()
            abg_est = NMEstimator(S, binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'Nelder-Mead', abg_est, finish-start) 
        
    D.closeFile() 
   
    print 'batch time = ', (time.clock() - batch_start) / 3600.0, ' hrs'


def AdjointEstimator():
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
    
    file_name = 'sinusoidal_spike_train_N=1000_crit_1'
    print file_name
    
    binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
    
    phi_omit = None
    binnedTrain.pruneBins(phi_omit, N_thresh = 100, T_thresh = 10.0)
    print 'N_bins = ', len(binnedTrain.bins.keys())
    
    Tf = binnedTrain.getTf()
    print 'Tf = ', Tf
        
    phis = binnedTrain.bins.keys();
    theta = binnedTrain.theta
    
    
    ps = binnedTrain._Train._params
    abg_true = array((ps._alpha, ps._beta, ps._gamma))
    print 'abg_true = ', abg_true
    
    abg = abg_true
    xmin = FPMultiPhiSolver.calculate_xmin(Tf, abg)
    dx = FPMultiPhiSolver.calculate_dx(abg, xmin)
    dt = FPMultiPhiSolver.calculate_dt(dx, abg, xmin, factor = 8.)
    print 'xmin, dx, dt = ', xmin, dx, dt
    S = FPMultiPhiSolver(theta, phis,
                     dx, dt, Tf, xmin)

    abg_init = initialize5(binnedTrain)
    print 'abg_init = ', abg_init
        
#    start = time.clock()
#    abg_est = TNCEstimator(S, binnedTrain, abg_init)
#    print 'Est. time = ', time.clock() - start
#    print 'abg_est = ', abg_est
    
#    start = time.clock()
#    abg_est = NMEstimator(S, binnedTrain, abg_init)
#    print 'Est. time = ', time.clock() - start
#    print 'abg_est = ', abg_est
#
#    start = time.clock()
#    abg_est = COBYLAEstimator(S, binnedTrain, abg_init)
#    print 'Est. time = ', time.clock() - start
#    print 'abg_est = ', abg_est

#    start = time.clock()
#    abg_est = CGEstimator(S, binnedTrain, abg_init)
#    print 'Est. time = ', time.clock() - start
#    print 'abg_est = ', abg_est

    start = time.clock()
    abg_est = BFGSEstimator(S, binnedTrain, abg_init)
    print 'Est. time = ', time.clock() - start
    print 'abg_est = ', abg_est

#    start = time.clock()
#    abg_est = FortetEstimator(binnedTrain, abg_init)
#    print 'Est. time = ', time.clock() - start
#    print 'abg_est = ', abg_est
    
    
def MultiTrainEstimator():
    phis =  linspace(.05, .95, 10)
    for file_name in ['sinusoidal_spike_train_T=1000.path',
                      'sinusoidal_spike_train_T=6000.path',
                      'sinusoidal_spike_train_T=10000_superSin.path',
                      'sinusoidal_spike_train_T=10000_crit.path']:

        print '#'*64
        
        print file_name
    
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
        
        phi_omit = None
        binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh = 10.)
        print 'N_bins = ', len(binnedTrain.bins.keys())
        
        Tf = binnedTrain.getTf()
        print 'Tf = ', Tf
            
        dx = .05; dt = FPMultiPhiSolver.calculate_dt(dx, 5., 2.)
    
        binphis = binnedTrain.bins.keys();
        theta = binnedTrain.theta
        
        S = FPMultiPhiSolver(theta, binphis,
                             dx, dt,
                             Tf)
    
        ps = binnedTrain._Train._params
        abg_true = array((ps._alpha, ps._beta, ps._gamma))
        print 'abg_true = ', abg_true
    
        abg_init = abs(abg_true + abg_true * randn(3)) ;
        print 'abg_init = ', abg_init
            
        start = time.clock()
        abg_est = TNCEstimator(S, binnedTrain, abg_init)
        print 'Est. time = ', time.clock() - start
        print 'abg_est = ', abg_est
        print 'error = ', abg_true - abg_est
        
        start = time.clock()
        abg_est = NMEstimator(S, binnedTrain, abg_init)
        print 'Est. time = ', time.clock() - start
        print 'abg_est = ', abg_est
        print 'error = ', abg_true - abg_est
    
def NMCritEstimator():
        phis =  linspace(.05, .95, 10)
        file_name = 'sinusoidal_spike_train_T=10000_crit.path'

        print '#'*64
        
        print file_name
    
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
        
        phi_omit = None
        binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh = 10.)
        print 'N_bins = ', len(binnedTrain.bins.keys())
        
        Tf = binnedTrain.getTf()
        print 'Tf = ', Tf
            
        dx = .05; dt = FPMultiPhiSolver.calculate_dt(dx, 5., 2.)
    
        binphis = binnedTrain.bins.keys();
        theta = binnedTrain.theta
        
        S = FPMultiPhiSolver(theta, binphis,
                             dx, dt,
                             Tf)
    
        ps = binnedTrain._Train._params
        abg_true = array((ps._alpha, ps._beta, ps._gamma))
        print 'abg_true = ', abg_true
    
        abg_init =  [0.804, 0.38 , 0.498]
        print 'abg_init = ', abg_init
            
        start = time.clock()
        abg_est = NMEstimator(S, binnedTrain, abg_init)
        print 'Est. time = ', time.clock() - start
        print 'abg_est = ', abg_est
        print 'error = ', abg_true - abg_est

      
def NMSuperSinEstimator():
        phis =  linspace(.05, .95, 40)
        file_name = 'sinusoidal_spike_train_T=10000_superSin.path'

        print '#'*64
        
        print file_name
    
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
        
        phi_omit = None
        binnedTrain.pruneBins(phi_omit, N_thresh = 128, T_thresh = 8.)
        print 'N_bins = ', len(binnedTrain.bins.keys())
        
        Tf = binnedTrain.getTf()
        print 'Tf = ', Tf
            
        dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, 5., 2.)
    
        binphis = binnedTrain.bins.keys();
        theta = binnedTrain.theta
        
        S = FPMultiPhiSolver(theta, binphis,
                             dx, dt,
                             Tf)
    
        ps = binnedTrain._Train._params
        abg_true = array((ps._alpha, ps._beta, ps._gamma))
        print 'abg_true = ', abg_true
    
        abg_init = [0.716 , 0.199 , 0.51]
        print 'abg_init = ', abg_init
            
        start = time.clock()
        abg_est = NMEstimator(S, binnedTrain, abg_init)
        print 'Est. time = ', time.clock() - start
        print 'abg_est = ', abg_est
        print 'error = ', abg_true - abg_est

    
if __name__ == '__main__':
    from pylab import *
    
    
    
#    MultiTrainEstimator()
#    AdjointEstimator()
#    BFGSItersComparison()
#    BatchAdjointEstimator()
    
#    NelderMeadSubTEstimator()
#    NMSuperSinEstimator()
#    NMCritEstimator()

    show()