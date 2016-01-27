'''
Created on Jul 5, 2012

@author: alex
'''

from BinnedSpikeTrain import BinnedSpikeTrain, intervalStats
from SpikeTrainSimulator import SpikeTrain, OUSinusoidalParams

from FPMultiPhiSolver import FPMultiPhiSolver

from numpy import linspace, float, arange, sum, sort, unique, amax, amin
from numpy import sin, sqrt, ones_like, zeros_like, where, diff, pi, log, max , sign, arctan, exp
from numpy import zeros, ones, array, r_, float64, matrix, bmat, Inf, ceil, arange, empty, interp, dot, sqrt, abs

from numpy.random import randn, randint
from scipy.stats.distributions import norm

import time
from InitBox import guesstimate5pts, initialize5, initialize_right_2std
from AdjointEstimator import NMEstimator, cNMEstimator,\
    FPL2Estimator, FPSupEstimator
from FortetEstimator import FortetEstimatorL2, FortetEstimatorSup
from DataHarvester import DataHarvester, DataPrinter


def ThetaBox(thetas, sample_id = 1):   
    D = DataPrinter('')
    for theta in thetas:
        N_phi = 20;
        phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

        file_name = 'sinusoidal_spike_train_N=1000_critical_theta=%d_%d'%(theta, sample_id)
        print file_name
                    
        T_thresh = 128.0;
        
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
        ps = binnedTrain._Train._params
        abg_true = array((ps._alpha, ps._beta, ps._gamma))
        print 'ps = ', ps.getParams()
        
        regime_name = 'theta=%d'%int(ps._theta)
        print regime_name
        D.setRegime(regime_name,abg_true, Tsim=-1.0)
        
        binnedTrain.pruneBins(None, N_thresh = 5, T_thresh=T_thresh)
        D.addSample(sample_id, binnedTrain.getTf(), binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
                    
        abg_init = initialize_right_2std(binnedTrain, cap_beta_gamma=True)
        D.addEstimate(sample_id, 'Initializer', abg_init,.0, warnflag = 0) 
                   
        #RELOAD ALL DATA:               
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
        binnedTrain.pruneBins(None, N_thresh = 1, T_thresh=T_thresh)
        
        #Weighted Fortet:            
        start = time.clock()
        abg_est, warnflag = FortetEstimatorSup(binnedTrain, abg_init, verbose = False)
        finish = time.clock()
        D.addEstimate(sample_id, 'Fortet', abg_est, finish-start, warnflag)
        
        #Weighted F-P:
        dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg_true, -1.0) 
        phis = binnedTrain.bins.keys();
        S = FPMultiPhiSolver(binnedTrain.theta, phis,
                             dx, dt,
                             binnedTrain.getTf(), X_min = -1.0)            
        
        start = time.clock()
        abg_est, warnflag = FPSupEstimator(S, binnedTrain, abg_init, verbose = False)
        finish = time.clock()
        D.addEstimate(sample_id, 'FP', abg_est, finish-start, warnflag)
                    
    del D  

def CustomEstimate(spike_trains):   
    D = DataPrinter('')
    for spike_train in spike_trains:
        regime_name = spike_train[0];
        sample_id = spike_train[1]
        N_spikes = spike_train[2];
        N_phi = spike_train[3];
    
        phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

        base_name = 'sinusoidal_spike_train_N=%d_'%N_spikes

        T_thresh = 128.0;
        regime_label = base_name + regime_name
            
        file_name = regime_label + '_' + str(sample_id)
        print file_name
        
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
        ps = binnedTrain._Train._params
        abg_true = array((ps._alpha, ps._beta, ps._gamma))
        
        D.setRegime(regime_name,abg_true, Tsim=-1.0)
        
        binnedTrain.pruneBins(None, N_thresh = 5, T_thresh=T_thresh)
        D.addSample(sample_id, binnedTrain.getTf(), binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
                    
        abg_init = initialize_right_2std(binnedTrain)
        abg_init[1] = amax([.1, abg_init[1]])
        abg_init[2] = amax([.0, abg_init[2]])
        D.addEstimate(sample_id, 'Initializer', abg_init,.0, warnflag = 0) 
                   
        #RELOAD ALL DATA:               
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
        binnedTrain.pruneBins(None, N_thresh = 1, T_thresh=T_thresh)
        
        #Weighted Fortet:            
        start = time.clock()
        abg_est, warnflag = FortetEstimatorSup(binnedTrain, abg_init, verbose = True)
        finish = time.clock()
        D.addEstimate(sample_id, 'Fortet', abg_est, finish-start, warnflag)
        
        #Weighted F-P:
        dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg_true, -1.0) 
        phis = binnedTrain.bins.keys();
        S = FPMultiPhiSolver(binnedTrain.theta, phis,
                             dx, dt,
                             binnedTrain.getTf(), X_min = -1.0)            
        
        start = time.clock()
        abg_est, warnflag = FPSupEstimator(S, binnedTrain, abg_init, verbose = True)
        finish = time.clock()
        D.addEstimate(sample_id, 'FP', abg_est, finish-start, warnflag)
                    
    del D  

def TestEstimate(N_spikes = 100, N_trains=5, N_phi=8):
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_N=%d_'%N_spikes
    
    D = DataPrinter('')

    for regime_name, T_thresh in zip(['superT'],
                                     [64]):
        regime_label = base_name + regime_name
            
        for sample_id in xrange(1,N_trains +1):
            file_name = regime_label + '_' + str(sample_id)
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            D.setRegime(regime_name,abg_true, Tsim=-1.0)
            
            binnedTrain.pruneBins(None, N_thresh = 5, T_thresh=T_thresh)
            D.addSample(sample_id, binnedTrain.getTf(), binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
                        
            abg_init = initialize_right_2std(binnedTrain)
            abg_init[1] = amax([.1, abg_init[1]])
            abg_init[2] = amax([.0, abg_init[2]])
            D.addEstimate(sample_id, 'Initializer', abg_init,.0, warnflag = 0) 
                       
            #RELOAD ALL DATA:               
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            binnedTrain.pruneBins(None, N_thresh = 1, T_thresh=T_thresh)
            
            #Weighted Fortet:            
            start = time.clock()
            abg_est, warnflag = FortetEstimatorSup(binnedTrain, abg_init, verbose = True)
            finish = time.clock()
            D.addEstimate(sample_id, 'Fortet', abg_est, finish-start, warnflag)
            
            #Weighted F-P:
            dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg_true, -1.0) 
            phis = binnedTrain.bins.keys();
            S = FPMultiPhiSolver(binnedTrain.theta, phis,
                                 dx, dt,
                                 binnedTrain.getTf(), X_min = -1.0)            
            
            start = time.clock()
            abg_est, warnflag = FPSupEstimator(S, binnedTrain, abg_init, verbose = True)
            finish = time.clock()
            D.addEstimate(sample_id, 'FP', abg_est, finish-start, warnflag)
                    
    del D  
    print 'batch time = ', (time.clock() - batch_start) / 3600.0, ' hrs'


def FortetVsWeightedFortet():
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_N=1000_'

    D = DataHarvester('FvsWF_4x16')
    for regime_name, T_thresh in zip(['superT', 'subT', 'crit', 'superSin'],
                                                       [6., 64, 32., 32.]):
        regime_label = base_name + regime_name
            
        for sample_id in xrange(1,17):
            file_name = regime_label + '_' + str(sample_id)
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            D.setRegime(regime_name,abg_true, Tsim=-1.0)
            
            phi_omit = None
            binnedTrain.pruneBins(phi_omit, N_thresh = 10, T_thresh=T_thresh)
            Tf = binnedTrain.getTf()
            D.addSample(sample_id, Tf, binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
             
            start = time.clock()
            abg_init = initialize_right_2std(binnedTrain)
            finish = time.clock()
            D.addEstimate(sample_id, 'Initializer', abg_init, finish-start) 
            
            start = time.clock()
            abg_est = FortetEstimator(binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'Fortet10', abg_est, finish-start)
            
            start = time.clock()
            abg_est = WeightedFortetEstimator(binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'WeghtedFortet', abg_est, finish-start)
            
            binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh=T_thresh)
            start = time.clock()
            abg_est = FortetEstimator(binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'Fortet64', abg_est, finish-start)
            
        
    D.closeFile() 
   
    print 'batch time = ', (time.clock() - batch_start) / 3600.0, ' hrs'


def WeightedFortet_N1():
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_N=1000_'

    D = DataHarvester('FvsWF_4x16')
    for regime_name, T_thresh in zip(['superT', 'subT', 'crit', 'superSin'],
                                     [32., 64, 32., 32.]):
        regime_label = base_name + regime_name
            
        for sample_id in xrange(1,17):
            file_name = regime_label + '_' + str(sample_id)
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            D.setRegime(regime_name,abg_true, Tsim=-1.0)
            
            phi_omit = None
            binnedTrain.pruneBins(phi_omit, N_thresh = 10, T_thresh=T_thresh)
            D.addSample(sample_id, binnedTrain.getTf(), binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
                
            abg_init = initialize_right_2std(binnedTrain)
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            
            start = time.clock()
            abg_est = FortetEstimator(binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'Fortet', abg_est, finish-start)
            
            start = time.clock()
            abg_est = WeightedFortetEstimator(binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'QuadFortet', abg_est, finish-start)
        
    D.closeFile() 
   
    print 'batch time = ', (time.clock() - batch_start) / 3600.0, ' hrs'



def Fortet_SupVsL2(N_spikes = 1000, N_trains = 16):
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_N=%d_'%N_spikes

    D = DataHarvester('Fortet_SupVsL2_4x%d'%N_trains)
#    for regime_name, T_thresh in zip(['subT', 'crit', 'superSin', 'superT'],
#                                     [64., 64, 32., 32.]):
    for regime_name, T_thresh in zip(['crit', 'superSin', 'superT'],
                                     [64, 32., 32.]):
        regime_label = base_name + regime_name
        
            
        for sample_id in xrange(1,N_trains+1):
            file_name = regime_label + '_' + str(sample_id)
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            D.setRegime(regime_name,abg_true, Tsim=-1.0)
            
            phi_omit = None
            binnedTrain.pruneBins(phi_omit, N_thresh = 10, T_thresh=T_thresh)
            D.addSample(sample_id, binnedTrain.getTf(), binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
                
            abg_init = initialize_right_2std(binnedTrain)
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            
            start = time.clock()
            abg_est = FortetEstimatorL2(binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'FortetL2', abg_est, finish-start)
            print abg_est, ' | %.2f'%(finish-start)
            
            start = time.clock()
            abg_est = FortetEstimatorSup(binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'FortetSup', abg_est, finish-start)
            print abg_est, ' | %.2f'%(finish-start)
        
    D.closeFile() 
   
    print 'batch time = ', (time.clock() - batch_start) / 3600.0, ' hrs'


def WeightedFP_N1(N_spikes = 100, N_trains=1):
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_N=%d_'%N_spikes

    D = DataHarvester('FPvsWFP_4x%d_N=%d'%(N_trains,N_spikes))
    for regime_name, T_thresh in zip(['subT','superT', 'crit', 'superSin'],
                                     [32.,32., 32., 32.]):
        regime_label = base_name + regime_name
            
        for sample_id in xrange(1,N_trains +1):
            file_name = regime_label + '_' + str(sample_id)
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            D.setRegime(regime_name,abg_true, Tsim=-1.0)
            
            #### N_thresh = 10
            binnedTrain.pruneBins(None, N_thresh = 10, T_thresh=T_thresh)
            D.addSample(sample_id, binnedTrain.getTf(), binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
                        
            abg_init = initialize_right_2std(binnedTrain)
            abg_init[1] = amax([.1, abg_init[1]])
            abg_init[2] = amax([.0, abg_init[2]])
            D.addEstimate(sample_id, 'init_N10', abg_init,.0) 
                       
            dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg_true, -1.0)            
            phis = binnedTrain.bins.keys();
            theta = binnedTrain.theta            
            
            S = FPMultiPhiSolver(theta, phis,
                                 dx, dt,
                                 binnedTrain.getTf(), X_min = -1.0)            
            start = time.clock()
            abg_est = cNMEstimator(S, binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'FP_N10', abg_est, finish-start) 
            del S;
            
            ##### N_thresh = 1
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            binnedTrain.pruneBins(None, N_thresh = 1, T_thresh=T_thresh)
            phis = binnedTrain.bins.keys();
            S = FPMultiPhiSolver(theta, phis,
                                 dx, dt,
                                 binnedTrain.getTf(), X_min = -1.0)            
            
            start = time.clock()
            abg_est = WeightedFPEstimator(S,binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'WFP_N1', abg_est, finish-start)
            del S;
                    
    D.closeFile() 
   
    print 'batch time = ', (time.clock() - batch_start) / 3600.0, ' hrs'


def FP_L2_vs_Sup(N_spikes = 1000, N_trains=20):
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_N=%d_'%N_spikes

    D = DataHarvester('FPvsWFP_4x%d_N=%d'%(N_trains,N_spikes))
    for regime_name, T_thresh in zip(['subT','superT', 'crit', 'superSin'],
                                     4*[64.]):
        regime_label = base_name + regime_name
            
        for sample_id in xrange(1,N_trains +1):
            file_name = regime_label + '_' + str(sample_id)
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            D.setRegime(regime_name,abg_true, Tsim=-1.0)
            
            #### N_thresh = 10
            binnedTrain.pruneBins(None, N_thresh = 10, T_thresh=T_thresh)
            D.addSample(sample_id, binnedTrain.getTf(), binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
                        
            abg_init = initialize_right_2std(binnedTrain)
            abg_init[1] = amax([.1, abg_init[1]])
            abg_init[2] = amax([.0, abg_init[2]])
            D.addEstimate(sample_id, 'init_N10', abg_init,.0) 


            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
                       
            dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg_true, -1.0)            
            theta = binnedTrain.theta            
            binnedTrain.pruneBins(None, N_thresh = 1, T_thresh=T_thresh)

            phis = binnedTrain.bins.keys();

            S = FPMultiPhiSolver(theta, phis,
                                 dx, dt,
                                 binnedTrain.getTf(), X_min = -1.0)            
            
            start = time.clock()
            abg_est = FPL2Estimator(S,binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'FP_L2', abg_est, finish-start)
            
            start = time.clock()
            abg_est = FPSupEstimator(S,binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'FP_Sup', abg_est, finish-start)
                    
    D.closeFile() 
   
    print 'batch time = ', (time.clock() - batch_start) / 3600.0, ' hrs'


def FinalEstimate(N_spikes = 100, N_trains=1, N_phi=20):
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_N=%d_'%N_spikes

    D = DataHarvester('FinalEstimate_4x%d_N=%d'%(N_trains,N_spikes),
                      overwrite=True)
    for regime_name, T_thresh in zip(['subT','superT', 'crit', 'superSin'],
                                     [128., 64., 64., 64.]):
        regime_label = base_name + regime_name
            
        for sample_id in xrange(1,N_trains +1):
            file_name = regime_label + '_' + str(sample_id)
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
        
            D.setRegime(regime_name,abg_true, Tsim=-1.0)
            
            binnedTrain.pruneBins(None, N_thresh = 5, T_thresh=T_thresh)
            D.addSample(sample_id, binnedTrain.getTf(), binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
                        
            abg_init = initialize_right_2std(binnedTrain)
            abg_init[1] = amax([.1, abg_init[1]])
            abg_init[2] = amax([.0, abg_init[2]])
            D.addEstimate(sample_id, 'Initializer', abg_init,.0, warnflag = 0) 
                       
            #RELOAD ALL DATA:               
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            binnedTrain.pruneBins(None, N_thresh = 1, T_thresh=T_thresh)
            
            #Weighted Fortet:            
            start = time.clock()
            abg_est, warnflag = FortetEstimatorSup(binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'Fortet', abg_est, finish-start, warnflag)
            
            #Weighted F-P:
            dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg_true, -1.0) 
            phis = binnedTrain.bins.keys();
            S = FPMultiPhiSolver(binnedTrain.theta, phis,
                                 dx, dt,
                                 binnedTrain.getTf(), X_min = -1.0)            
            
            start = time.clock()
            abg_est, warnflag = FPSupEstimator(S, binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'FP', abg_est, finish-start, warnflag)
                    
    D.closeFile() 
   
    print 'batch time = ', (time.clock() - batch_start) / 3600.0, ' hrs'


def ThetaEstimate(N_spikes = 1000, N_trains=100, N_phi=20, 
                  thetas = [1, 5, 10, 20]):
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    batch_start = time.clock()
    base_name = 'sinusoidal_spike_train_N=%d_critical_theta='%N_spikes

    T_thresh = 64.
    
    D = DataHarvester('ThetaEstimate_%dx%d_N=%d'%(len(thetas),N_trains,N_spikes))
    for sample_id in xrange(1,N_trains +1):
        for theta in thetas:    
            regime_name = 'theta%d'%theta
            regime_label = base_name + '%d'%theta            
            file_name = regime_label + '_%d'%sample_id
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            D.setRegime(regime_name,abg_true, Tsim=-1.0)
            
            binnedTrain.pruneBins(None, N_thresh = 5, T_thresh=T_thresh)
            D.addSample(sample_id, binnedTrain.getTf(), binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
                        
            abg_init = initialize_right_2std(binnedTrain)
            abg_init[1] = amax([.1, abg_init[1]])
            abg_init[2] = amax([.0, abg_init[2]])
            D.addEstimate(sample_id, 'Initializer', abg_init,.0, warnflag = 0) 
                       
            #RELOAD ALL DATA:               
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            binnedTrain.pruneBins(None, N_thresh = 1, T_thresh=T_thresh)
            
            #Weighted Fortet:            
            start = time.clock()
            abg_est, warnflag = FortetEstimatorSup(binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'Fortet', abg_est, finish-start, warnflag)
            
            #Weighted F-P:
            dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg_true, -1.0) 
            phis = binnedTrain.bins.keys();
            S = FPMultiPhiSolver(binnedTrain.theta, phis,
                                 dx, dt,
                                 binnedTrain.getTf(), X_min = -1.0)            
            
            start = time.clock()
            abg_est, warnflag = FPSupEstimator(S, binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'FP', abg_est, finish-start, warnflag)
                    
    D.closeFile() 
   
    print 'batch time = ', (time.clock() - batch_start) / 3600.0, ' hrs'


def CvsPyEstimate():
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_N=1000_'

    D = DataHarvester('CvsPY_2x4')
    for regime_name, T_thresh in zip(['subT', 'superSin'],
                                                       [32, 16.]):
        regime_label = base_name + regime_name
            
        for sample_id in xrange(1,4):
            file_name = regime_label + '_' + str(sample_id)
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            D.setRegime(regime_name,abg_true, Tsim=-1.0)
            
            phi_omit = None
            binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh=T_thresh)
            Tf = binnedTrain.getTf()
            D.addSample(sample_id, Tf, binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
             
            start = time.clock()
            abg_init = initialize_right_2std(binnedTrain)
            finish = time.clock()
            D.addEstimate(sample_id, 'Initializer', abg_init, finish-start) 
             
            dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg_true, -1.0)
            
            phis = binnedTrain.bins.keys();
            theta = binnedTrain.theta
            
            S = FPMultiPhiSolver(theta, phis,
                                 dx, dt,
                                 Tf, X_min = -1.0)

            start = time.clock()
            abg_est = cNMEstimator(S, binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'FP-C', abg_est, finish-start)
               
            start = time.clock()
            abg_est = NMEstimator(S, binnedTrain, abg_init)
            finish = time.clock()
            D.addEstimate(sample_id, 'FP-PY', abg_est, finish-start) 
            
        
    D.closeFile()

def estimateSubT(N_spikes=100, sample_id =4,  T_thresh = 32.):
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    base_name = 'sinusoidal_spike_train_N=%d_'%N_spikes
    regime_name = 'subT'
    regime_label = base_name + regime_name
    file_name = regime_label + '_' + str(sample_id)
    print file_name
            
    binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
    ps = binnedTrain._Train._params
    print array((ps._alpha, ps._beta, ps._gamma))
    
    #### N_thresh = 10
    binnedTrain.pruneBins(None, N_thresh = 10, T_thresh=T_thresh)
                
    abg_init = initialize_right_2std(binnedTrain)
    print abg_init
    abg_init[1] = amax([.1, abg_init[1]])
    abg_init[2] = amax([.0, abg_init[2]])
            
    dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg_init, -1.0)            
    print dx, dt
    theta = binnedTrain.theta            
      
    ##### N_thresh = 1
    binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
    binnedTrain.pruneBins(None, N_thresh = 1, T_thresh=T_thresh)
    phis = binnedTrain.bins.keys();
    S = FPMultiPhiSolver(theta, phis,
                         dx, dt,
                         binnedTrain.getTf(), X_min = -1.0)            
    abg_est = WeightedFPEstimator(S,binnedTrain, abg_init)
    print abg_est
    
    
if __name__ == '__main__':
    
#    TestEstimate(N_spikes=100, N_trains = 1, N_phi=8)
#    TestEstimate(N_spikes=1000, N_trains = 2, N_phi=8)
#    CustomEstimate([('superT', 54, 100, 8),
#                    ('superSin', 80, 100,8 )] )
    CustomEstimate([('critical_theta=20', 32, 1000, 20)
                    ] )

#    ThetaBox(thetas = [5,10,20], sample_id=30)
#    ThetaEstimate(N_trains=100)

#    FinalEstimate(N_spikes=100, N_trains = 100, N_phi=8)
#    FinalEstimate(N_spikes=1000, N_trains = 100,N_phi=20)
    
#    TestEstimate()
#    BatchEstimate()
#    CvsPyEstimate()
#    FortetVsWeightedFortet()
#    WeightedFortet_N1()
#    WeightedFP_N1()

#    Fortet_SupVsL2(N_spikes=1000, N_trains = 20);

#    FP_L2_vs_Sup(N_spikes= 1000,N_trains = 20)
    
#    import cProfile
#    cProfile.run('estimateSubT()')
#    estimateSubT()