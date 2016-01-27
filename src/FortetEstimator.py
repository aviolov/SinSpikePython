# -*- coding:utf-8 -*-
"""
Created on May 28, 2012

@author: alex
"""
from __future__ import division

from BinnedSpikeTrain import BinnedSpikeTrain, intervalStats, loadPath
#from Simulator import Path, OUSinusoidalParams,
from SpikeTrainSimulator import SpikeTrain, OUSinusoidalParams

from numpy import iterable, tile
from numpy import linspace, float, arange, sum, amax
from numpy import sin, sqrt, ones_like, zeros_like, where, diff, pi, log, max , sign, arctan, exp
from numpy import zeros, ones, array, c_, r_, float64, matrix, bmat, Inf, ceil, arange, empty, interp, dot, sqrt
from scipy.sparse import spdiags, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from copy import deepcopy
from DataHarvester import DataAnalyzer
from scipy.stats.distributions import norm

import ext_fpc

#RESULTS_DIR = '/home/alex/Workspaces/Python/OU_hitting_time/Results/Fortet/'
FIGS_DIR = '/home/alex/Workspaces/Latex/LIFEstimation/Figs/Fortet'
import os
for D in [FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time
    
def FortetEstimator(binnedTrain, abg_init):
    print 'Fortet Method: '
    
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
         
    def loss_function(abg):
#            #SIMPLE: Penalize negative a's, we want a positive, b/c for a<0, the algorithm is different:
#            if min(abg)<.0 or max(abg) > 5.:
#                return 1e6#
            error = .0;
            for (phi_m, phi_idx) in zip(phis, xrange(N_phi)):
                Is = bins[phi_m]['Is']
                uniqueIs = bins[phi_m]['unique_Is']
                 
                a,b,g = abg[0], abg[1], abg[2]
                movingThreshold = getMovingThreshold(a,g, phi_m)
                
                LHS_numerator = movingThreshold(uniqueIs[1:]) *sqrt(2.)
                LHS_denominator = b * sqrt(1 - exp(-2*uniqueIs[1:]))
                LHS = 1 -  norm.cdf(LHS_numerator / LHS_denominator)
                
                RHS = zeros_like(LHS)
                N  = len(Is)
                for rhs_idx in xrange(1,len(uniqueIs)):
                    t = uniqueIs[rhs_idx]
                    lIs = Is[Is<t]
                    taus = t - lIs;
                    
                    numerator = (movingThreshold(t) - movingThreshold(lIs)* exp(-taus)) * sqrt(2.)
                    denominator = b *  sqrt(1. - exp(-2*taus))
                    RHS[rhs_idx-1] = sum(1. - norm.cdf(numerator/denominator)) / N
                
#                error += sum(abs(LHS - RHS));
                error += sum((LHS - RHS)**2);
#                error += max(abs(LHS - RHS))

            return error

    from scipy.optimize import fmin
    abg = fmin(loss_function, abg_init, xtol=1e-4, ftol = 1e-5, disp=1);
    
    return abg
        

def getMovingThreshold(a,g, phi, theta):
    psi = arctan(theta)
    mvt = lambda ts: 1. - ( a*(1 - exp(-ts)) + \
                            g / sqrt(1+theta*theta) * ( sin ( theta *  ( ts + phi) - psi) \
                                                -exp(-ts)*sin(phi*theta - psi) ))

    return mvt

def FortetEstimatorL2(binnedTrain, abg_init):
    print 'Weighted Fortet Method L2 metric: '
    
    bins = binnedTrain.bins;
    phis = bins.keys()
    N_phi = len(phis)
    
     
    def loss_function_manualquad(abg ):
        error = .0;
        for (phi_m, phi_idx) in zip(phis, xrange(N_phi)):
            Is = bins[phi_m]['Is']
            unique_Is = bins[phi_m]['unique_Is']
            N_Is  = len(Is);
            Tf = amax(Is);
             
            a,b,g = abg[0], abg[1], abg[2]
            movingThreshold = getMovingThreshold(a,g, phi_m, binnedTrain.theta)
            
            def LHS(ts):
                LHS_numerator = movingThreshold(ts) *sqrt(2.)
                LHS_denominator = b * sqrt(1 - exp(-2*ts))
                return 1 -  norm.cdf(LHS_numerator / LHS_denominator);
              
            def RHS(ts):
                if False == iterable(ts):
                    ts =  array([ts])
                lIs = tile(Is,  len(ts) ).reshape((len(ts), len(Is))).transpose()
                lts = tile(ts, (len(Is),1 ) )
                mask = lIs < lts
                taus = (lts - lIs); #*mask
                #NOTE BELOW WE use abs(taus) since for non-positive taus we will mask away anyway:
                numerator = (movingThreshold(lts) - movingThreshold(lIs)* exp(-abs(taus))) * sqrt(2.)
                denominator = b *  sqrt(1. - exp(-2*abs(taus)))
                
                rhs = sum( (1. - norm.cdf(numerator/denominator))*mask, axis=0) / N_Is
                return rhs
            
#            dt = 1e-2;
#            ts_manual = arange(1e-8, Tf+1e-2, dt )
            dt = 1.0
            ts_manual = linspace(1e-8, Tf+1e-2, 500 )
            integrand = LHS(ts_manual) - RHS(ts_manual)
            val  = dot(integrand, integrand)*dt;
            
            weight = len(Is)
            lerror = val* weight;
            error += lerror
           
        return error
   
    from scipy.optimize import fmin
    abg = fmin(loss_function_manualquad, abg_init, xtol=1e-4, ftol = 1e-2, disp=1);
    
    return abg
    

def FortetEstimatorSup(binnedTrain, abg_init, verbose = False, ftol = None):
    print 'Weighted Fortet : Normalized Sup metric - C version'
    
    NUM_TIME_PTS = 1000;
        
    bins = binnedTrain.bins;
    phis = bins.keys()
    N_phi = len(phis)
    theta = binnedTrain.theta;   
    
    def loss_function(abg):
        error = .0;
        for phi_m in phis:
            Is = bins[phi_m]['Is']
            N_Is  = len(Is);
            Tf = amax(Is);
            ts = linspace(1e-8, Tf+1e-2, NUM_TIME_PTS)
            #Call C to compute the lhs - rhs
            abgthphi = r_[abg, theta, phi_m]
            difference = ext_fpc.FortetError(abgthphi, ts, Is)
            raw_error  = amax(difference);
            weighted_error = N_Is * raw_error;                    
            error += weighted_error
            if verbose:
                print 'abg = ', abg, '; error =', error
                                    
        return error
    
    from scipy.optimize import fmin
    if None == ftol:
        ftol = 1e-3 * binnedTrain.getSpikeCount()
        print 'ftol = ', ftol
    abg, fopt, iter, funcalls, warnflag, allvecs = fmin(loss_function, abg_init,
                                                        maxiter = 200, 
                                                        xtol = 1e-2, ftol = ftol,
                                                        disp=1, full_output=True, retall = 1);
    return abg, warnflag

def Harness(sample_id=13, regime_name='superSin', N_spikes = 1000, visualize=False):
    from scipy.stats.distributions import norm
    N_phi = 20;
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
    base_name = 'sinusoidal_spike_train_N=%d_'%N_spikes

    regime_label = base_name + regime_name
#    T_thresh = 128.;     
    file_name = regime_label + '_' + str(sample_id)
    print file_name
        
    binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
    
#    print 'Warning: pruning bins'
#    binnedTrain.pruneBins(None, N_thresh = 100)
    
    bins = binnedTrain.bins;
    phis = bins.keys()
    N_phi = len(phis)
    
    alpha,  beta, gamma, theta = binnedTrain._Train._params.getParams() 

    def loss_function_simple(abg, visualize, fig_tag = ''):
        error = .0;
        if visualize:
            figure()
                        
        for (phi_m, phi_idx) in zip(phis, xrange(N_phi)):
            Is = bins[phi_m]['Is']
            uniqueIs = bins[phi_m]['unique_Is']
             
            a,b,g = abg[0], abg[1], abg[2]
            movingThreshold = getMovingThreshold(a,g, phi_m,binnedTrain.theta)
            
            LHS_numerator = movingThreshold(uniqueIs[1:]) *sqrt(2.)
            LHS_denominator = b * sqrt(1 - exp(-2*uniqueIs[1:]))
            LHS = 1 -  norm.cdf(LHS_numerator / LHS_denominator)
            
            RHS = zeros_like(LHS)
            N  = len(Is)
            for rhs_idx in xrange(1,len(uniqueIs)):
                t = uniqueIs[rhs_idx]
                lIs = Is[Is<t]
                taus = t - lIs;
                
                numerator = (movingThreshold(t) - movingThreshold(lIs)* exp(-taus)) * sqrt(2.)
                denominator = b *  sqrt(1. - exp(-2*taus))
                RHS[rhs_idx-1] = sum(1. - norm.cdf(numerator/denominator)) / N
            
            weight = len(Is)
            lerror = dot((LHS - RHS)**2 , diff(uniqueIs)) * weight;
            error += lerror
        
            if visualize:
                    subplot(ceil(len(phis)/2),2, phi_idx+1);hold(True)
                    ts = uniqueIs[1:]; 
                    plot(ts, LHS, 'b');
                    plot(ts, RHS, 'rx');
#                    annotate('$\phi$ = %.2g'%(phi_m), ((min(ts), max(LHS)/2.)), ) 
                    annotate('lerror = %.3g'%lerror,((min(ts), max(LHS)/2.)), ) 
        if visualize:
            subplot(ceil(len(phis)/2),2, 1);
            title(fig_tag)          
        return error
    
    def loss_function_nonvectorized(abg, visualize=False):
        error = .0;
        if visualize:
            figure();  
        for (phi_m, phi_idx) in zip(phis, xrange(N_phi)):
#        for phi_m in phis:
            Is = bins[phi_m]['Is']
            N  = len(Is);
            Tf = amax(Is);
             
            a,b,g = abg[0], abg[1], abg[2]
            movingThreshold = getMovingThreshold(a,g, phi_m)
            
            def LHS(ts):
                LHS_numerator = movingThreshold(ts) *sqrt(2.)
                LHS_denominator = b * sqrt(1 - exp(-2*ts))
                return 1 -  norm.cdf(LHS_numerator / LHS_denominator);
            
#            def RHS(t):
#                lIs = Is[Is<t];
#                taus = t - lIs;
#                numerator = (movingThreshold(t) - movingThreshold(lIs)* exp(-taus)) * sqrt(2.)
#                denominator = b *  sqrt(1. - exp(-2*taus))
#                return sum(1. - norm.cdf(numerator/denominator)) / N
              
            def RHS(ts):
                if False == iterable(ts):
                    ts =  [ts] 
                rhs = empty_like(ts)
                for t, t_indx in zip(ts, xrange(size(ts))):
                    lIs = Is[Is<t];
                    taus = t - lIs;
                    numerator = (movingThreshold(t) - movingThreshold(lIs)* exp(-taus)) * sqrt(2.)
                    denominator = b *  sqrt(1. - exp(-2*taus))
                    rhs[t_indx] = sum(1. - norm.cdf(numerator/denominator)) / N
                return rhs
            
            integrand = lambda t: (LHS(t) - RHS(t)) **2
            from scipy.integrate import quad, quadrature, fixed_quad

#            quadrature, quad_error = quad(integrand, a= 1e-8, b=Tf+1e-8, limit = 50)
            quadrature, quad_error = quadrature(integrand, a= 1e-8, b=Tf+1e-8,
                                                tol=5e-03, rtol=1.49e-04,
                                                maxiter = 64,
                                                vec_func = True)
#            val , err_msg = fixed_quad( integrand, a= 1e-8, b=Tf+1e-8,
#                                                n = 12)
#            print 'quadrature = ',quadrature
#            print 'val = ',val
#            print 'difference = ', quadrature - val
            
            weight = len(Is)
            #VISUALIZE FOR NOW:
            if visualize:
                subplot(ceil(len(phis)/2),2, phi_idx+1);hold(True)
                ts = linspace(1e-8,  Tf+1e-8, 100) ; 
                lhs = empty_like(ts); rhs = empty_like(ts); 
                for t, t_indx in zip(ts,
                                      xrange(len(ts))):
                    lhs[t_indx] = LHS(t);
                    rhs[t_indx] = RHS(t);
                plot(ts, lhs, 'b');
                plot(ts, rhs, 'rx'); 
            
            error +=  quadrature* weight;
        return error
    
    def loss_function_quadGaussian(abg, visualize=False, fig_tag = ''):
        error = .0;
        if visualize:
            figure();  
        for (phi_m, phi_idx) in zip(phis, xrange(N_phi)):
#        for phi_m in phis:
            Is = bins[phi_m]['Is']
            unique_Is = bins[phi_m]['unique_Is']
            N_Is  = len(Is);
            Tf = amax(Is);
             
            a,b,g = abg[0], abg[1], abg[2]
            movingThreshold = getMovingThreshold(a,g, phi_m)
            
            def LHS(ts):
                LHS_numerator = movingThreshold(ts) *sqrt(2.)
                LHS_denominator = b * sqrt(1 - exp(-2*ts))
                return 1 -  norm.cdf(LHS_numerator / LHS_denominator);
              
            def RHS(ts):
                if False == iterable(ts):
                    ts =  array([ts])
#                rhs = empty_like(ts)
                
#                Is.reshape((len(Is),1)) 
                lIs = tile(Is,  len(ts) ).reshape((len(ts), len(Is))).transpose()
                lts = tile(ts, (len(Is),1 ) )
                mask = lIs < lts
                taus = (lts - lIs); #*mask
                #NOTE BELOW WE use abs(taus) since for non-positive taus we will mask away anyway:
                numerator = (movingThreshold(lts) - movingThreshold(lIs)* exp(-abs(taus))) * sqrt(2.)
                denominator = b *  sqrt(1. - exp(-2*abs(taus)))
                
                rhs = sum( (1. - norm.cdf(numerator/denominator))*mask, axis=0) / N_Is
                return rhs
            
            integrand = lambda t: (LHS(t) - RHS(t)) **2
            from scipy.integrate import quad, quadrature, fixed_quad

#            valcheck, quad_error = quad(integrand, a= 1e-8, b=Tf+1e-8, limit = 64)
            
            val, quad_error = quadrature(integrand, a= 1e-8, b=Tf+1e-8,
                                                tol=5e-03, rtol=1.49e-04,
                                                maxiter = 64,
                                                vec_func = True)
            
            weight = len(Is)
            lerror = val* weight;
            error += lerror

            if visualize:
                subplot(ceil(len(phis)/2),2, phi_idx+1); hold(True)
#                ts = linspace(1e-8,  Tf+1e-8, 100) ; 
                ts = unique_Is[1:];
                lhs  = LHS(ts);
                rhs  = RHS(ts);
                plot(ts, lhs, 'b');
                plot(ts, rhs, 'rx'); 
                annotate('lerror = %.3g'%lerror,((min(ts), max(lhs)/2.)), ) 
        
        if visualize:
            subplot(ceil(len(phis)/2),2,1);
            title(fig_tag)
        return error
    
    def loss_function_L1(abg, visualize=False, fig_tag = ''):
        error = .0;
        if visualize:
            figure();  
        for (phi_m, phi_idx) in zip(phis, xrange(N_phi)):
#        for phi_m in phis:
            Is = bins[phi_m]['Is']
            unique_Is = bins[phi_m]['unique_Is']
            N_Is  = len(Is);
            Tf = amax(Is);
             
            a,b,g = abg[0], abg[1], abg[2]
            movingThreshold = getMovingThreshold(a,g, phi_m, theta)
            
            def LHS(ts):
                LHS_numerator = movingThreshold(ts) *sqrt(2.)
                LHS_denominator = b * sqrt(1 - exp(-2*ts))
                return 1 -  norm.cdf(LHS_numerator / LHS_denominator);
              
            def RHS(ts):
                if False == iterable(ts):
                    ts =  array([ts])
#                rhs = empty_like(ts)
                
#                Is.reshape((len(Is),1)) 
                lIs = tile(Is,  len(ts) ).reshape((len(ts), len(Is))).transpose()
                lts = tile(ts, (len(Is),1 ) )
                mask = lIs < lts
                taus = (lts - lIs); #*mask
                #NOTE BELOW WE use abs(taus) since for non-positive taus we will mask away anyway:
                numerator = (movingThreshold(lts) - movingThreshold(lIs)* exp(-abs(taus))) * sqrt(2.)
                denominator = b *  sqrt(1. - exp(-2*abs(taus)))
                
                rhs = sum( (1. - norm.cdf(numerator/denominator))*mask, axis=0) / N_Is
                return rhs
            
            integrand = lambda t: abs(LHS(t) - RHS(t))
            from scipy.integrate import quad, quadrature, fixed_quad

            print unique_Is
            val, quad_error = quad(integrand, a= 1e-8, b=Tf+1., limit = 64,
                                   points = sort(unique_Is) )
            
#            val, quad_error = quadrature(integrand, a= 1e-8, b=Tf+1e-8,
#                                                tol=5e-03, rtol=1.49e-04,
#                                                maxiter = 64,
#                                                vec_func = True)
#            val , err_msg = fixed_quad( integrand, a= 1e-8, b=Tf+1e-8,

            weight = len(Is)
            lerror = val* weight;
            error += lerror

            if visualize:
                subplot(ceil(len(phis)/2),2, phi_idx+1); hold(True)
#                ts = linspace(1e-8,  Tf+1e-8, 100) ; 
                ts = unique_Is[1:];
                lhs  = LHS(ts);
                rhs  = RHS(ts);
                plot(ts, lhs, 'b');
                plot(ts, rhs, 'rx'); 
                annotate('lerror = %.3g'%lerror,((min(ts), max(lhs)/2.)), ) 
        
        if visualize:
            subplot(ceil(len(phis)/2),2,1); title(fig_tag)
        return error
    
    
    def loss_function_manualquad(abg, visualize=False, fig_tag = ''):
        error = .0;
        if visualize:
            figure();  
        for (phi_m, phi_idx) in zip(phis, xrange(N_phi)):
            Is = bins[phi_m]['Is']
            unique_Is = bins[phi_m]['unique_Is']
            N_Is  = len(Is);
            Tf = amax(Is);
             
            a,b,g = abg[0], abg[1], abg[2]
            movingThreshold = getMovingThreshold(a,g, phi_m, binnedTrain.theta)
            
            def LHS(ts):
                LHS_numerator = movingThreshold(ts) *sqrt(2.)
                LHS_denominator = b * sqrt(1 - exp(-2*ts))
                return 1 -  norm.cdf(LHS_numerator / LHS_denominator);
              
            def RHS(ts):
                if False == iterable(ts):
                    ts =  array([ts])
                lIs = tile(Is,  len(ts) ).reshape((len(ts), len(Is))).transpose()
                lts = tile(ts, (len(Is),1 ) )
                mask = lIs < lts
                taus = (lts - lIs); #*mask
                #NOTE BELOW WE use abs(taus) since for non-positive taus we will mask away anyway:
                numerator = (movingThreshold(lts) - movingThreshold(lIs)* exp(-abs(taus))) * sqrt(2.)
                denominator = b *  sqrt(1. - exp(-2*abs(taus)))
                
                rhs = sum( (1. - norm.cdf(numerator/denominator))*mask, axis=0) / N_Is
                return rhs
            
            integrand = lambda t: (LHS(t) - RHS(t)) **2
            dt = 1e-2;
            ts_manual = arange(1e-8, Tf+1e-2, dt )
            integrand = LHS(ts_manual) - RHS(ts_manual)
            val  = dot(integrand, integrand)*dt;
            
            weight = len(Is)
            lerror = val* weight;
            error += lerror

            if visualize:
                subplot(ceil(len(phis)/2),2, phi_idx+1); hold(True)
#                ts = linspace(1e-8,  Tf+1e-8, 100) ; 
                ts = unique_Is[1:];
                lhs  = LHS(ts);
                rhs  = RHS(ts);
                plot(ts, lhs, 'b');
                plot(ts, rhs, 'rx'); 
                annotate('lerror = %.3g'%lerror,((min(ts), max(lhs)/2.)), ) 
        
        if visualize:
            subplot(ceil(len(phis)/2),2,1);
            title(fig_tag)
        return error

    def loss_function_supnormalized(abg, visualize=False, fig_tag = ''):
        error = .0;
        if visualize:
            figure();  
        for (phi_m, phi_idx) in zip(phis, xrange(N_phi)):
            Is = bins[phi_m]['Is']
            unique_Is = bins[phi_m]['unique_Is']
            N_Is  = len(Is);
            Tf = amax(Is);
             
            a,b,g = abg[0], abg[1], abg[2]
            movingThreshold = getMovingThreshold(a,g, phi_m, binnedTrain.theta)
            
            def LHS(ts):
                LHS_numerator = movingThreshold(ts) *sqrt(2.)
                LHS_denominator = b * sqrt(1 - exp(-2*ts))
                return 1 -  norm.cdf(LHS_numerator / LHS_denominator);
              
            def RHS(ts):
                if False == iterable(ts):
                    ts =  array([ts])
                lIs = tile(Is,  len(ts) ).reshape((len(ts), len(Is))).transpose()
                lts = tile(ts, (len(Is),1 ) )
                mask = lIs < lts
                taus = (lts - lIs); #*mask
                #NOTE BELOW WE use abs(taus) since for non-positive taus we will mask away anyway:
                numerator = (movingThreshold(lts) - movingThreshold(lIs)* exp(-abs(taus))) * sqrt(2.)
                denominator = b *  sqrt(1. - exp(-2*abs(taus)))
                
                rhs = sum( (1. - norm.cdf(numerator/denominator))*mask, axis=0) / N_Is
                return rhs
            
            dt = 1e-3;
            ts_manual = arange(1e-8, Tf+1e-2, dt)
            lhs = LHS(ts_manual)
            difference = abs(lhs - RHS(ts_manual))/amax(lhs)
            val  = amax(difference);
            
            weight = len(Is)
            lerror = val* weight;
            error += lerror

            if visualize:
                subplot(ceil(len(phis)/2),2, phi_idx+1); hold(True)
#                ts = linspace(1e-8,  Tf+1e-8, 100) ; 
                ts = unique_Is[1:];
                lhs  = LHS(ts);
                rhs  = RHS(ts);
                plot(ts, lhs, 'b');
                plot(ts, rhs, 'rx'); 
                annotate('lerror = %.3g'%lerror,((min(ts), max(lhs)/2.)), ) 
        
        if visualize:
            subplot(ceil(len(phis)/2),2,1);
            title(fig_tag)
            
        return error
    def loss_function_sup(abg, visualize=False, fig_tag = ''):
        error = .0;
        if visualize:
            figure();  
        for (phi_m, phi_idx) in zip(phis, xrange(N_phi)):
            Is = bins[phi_m]['Is']
            unique_Is = bins[phi_m]['unique_Is']
            N_Is  = len(Is);
            Tf = amax(Is);
             
            a,b,g = abg[0], abg[1], abg[2]
            movingThreshold = getMovingThreshold(a,g, phi_m, binnedTrain.theta)
            
            def LHS(ts):
                LHS_numerator = movingThreshold(ts) *sqrt(2.)
                LHS_denominator = b * sqrt(1 - exp(-2*ts))
                return 1 -  norm.cdf(LHS_numerator / LHS_denominator);
              
            def RHS(ts):
                if False == iterable(ts):
                    ts =  array([ts])
                lIs = tile(Is,  len(ts) ).reshape((len(ts), len(Is))).transpose()
                lts = tile(ts, (len(Is),1 ) )
                mask = lIs < lts
                taus = (lts - lIs); #*mask
                #NOTE BELOW WE use abs(taus) since for non-positive taus we will mask away anyway:
                numerator = (movingThreshold(lts) - movingThreshold(lIs)* exp(-abs(taus))) * sqrt(2.)
                denominator = b *  sqrt(1. - exp(-2*abs(taus)))
                
                rhs = sum( (1. - norm.cdf(numerator/denominator))*mask, axis=0) / N_Is
                return rhs
            
            dt = 1e-3;
            ts_manual = arange(1e-8, Tf+1e-2, dt)
            lhs = LHS(ts_manual)
            difference = abs(lhs - RHS(ts_manual)) 
            val  = amax(difference);
            
            weight = len(Is)
            lerror = val* weight;
            error += lerror

            if visualize:
                subplot(ceil(len(phis)/2),2, phi_idx+1); hold(True)
#                ts = linspace(1e-8,  Tf+1e-8, 100) ; 
                ts = unique_Is[1:];
                lhs  = LHS(ts);
                rhs  = RHS(ts);
                plot(ts, lhs, 'b');
                plot(ts, rhs, 'rx'); 
                annotate('lerror = %.3g'%lerror,((min(ts), max(lhs)/2.)), ) 
        
        if visualize:
            subplot(ceil(len(phis)/2),2,1);
            title(fig_tag)
            
        return error
    
    #EXPERIMENT:
   
#    Analyzer = DataAnalyzer()
    
    def outlinept():
        pass
    
    
    
    analyzer = DataAnalyzer('FvsWF_4x16');

    
    
    abg_true = analyzer.getTrueParamValues(regime_name)
    loss_function_L1(abg_true, visualize=True)
    return 

    quad_estimated = analyzer.getEstimates(sample_id, regime_name, 'QuadFortet')[0]
    simple_estimated = analyzer.getEstimates(sample_id, regime_name, 'Fortet')[0]

    for abg, tag, L in zip(3*[abg_true, quad_estimated],
                           ['sup_true_params'      , 'sup_estimated_params',
                            'supnormailzed_true_params', 'supnormailzed_estimated_params',
                            'manualquad_true_params' , 'manualquad_estimated_params'],
                           2*[loss_function_sup]+
                           2*[loss_function_supnormalized] +
                           2*[loss_function_manualquad]):
        start = time.clock()
        loss =  L(abg,visualize, fig_tag = regime + '_' + tag);
        end = time.clock()  
        print tag, ':%.2f,%.2f,%.2f:' %(abg[0],abg[1],abg[2]), 'error = %.4f'%loss , ' | compute time = ', end - start  
        
#        filename = os.path.join(FIGS_DIR, tag + '_'+ regime_name + str(sample_id) + '.png')
#        print 'saving to ', filename
#        get_current_fig_manager().window.showMaximized()
#        savefig(filename)



     
def VisualizeSinusoidallyDominating():
        N_phi = 20;
        print 'N_phi = ', N_phi
        
        phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
        phi_omit = None
        
        file_name = 'sinusoidal_spike_train_T=5000_superSin_11.path'
        
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
        binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh=16.)
        print 'N_bins = ', len(binnedTrain.bins.keys())
    
        Tf = binnedTrain.getTf()
        print 'Tf = ', Tf
       
        solver_phis = binnedTrain.bins.keys();
        theta = binnedTrain.theta;
    
        ps = binnedTrain._Path._params
        abg_true = array((ps._alpha, ps._beta, ps._gamma))

        abg = abg_true
        visualizeData_vs_Fortet(abg, binnedTrain, theta,
                                title_tag =  'TRUE: ',
                                save_fig_name='SuperSin_true_params')
        get_current_fig_manager().window.showMaximized()
       
        abg_est =  [.494,.140,1.11]
        abg = abg_est
        visualizeData_vs_Fortet(abg, binnedTrain, theta,
                                 title_tag =  'F-P: ',
                            save_fig_name='SuperSin_NM_est')
        get_current_fig_manager().window.showMaximized()
       
       
        abg_est =  [.541,.181, .983]
        abg = abg_est
        visualizeData_vs_Fortet(abg, binnedTrain,theta,
                                title_tag = 'Fortet: ',
                                save_fig_name='SuperSin_Fortet_est')
        get_current_fig_manager().window.showMaximized()
#    

def FortetHarness():
    
    errors = []
    
    for N_spikes, N_phi in zip( [100,1000],
                                [8,20]):
        phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
        base_name = 'sinusoidal_spike_train_N=%d_'%N_spikes


    for regime  in ['superT', 'crit', 'subT','superSin']:
        regime_label = base_name + regime

        for sample_id in [1,23, 36, 77, 99]:
            file_name = regime_label + '_' + str(sample_id)
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
        
            bins = binnedTrain.bins;
            phis = bins.keys()
            N_phi = len(phis)
            
            alpha,  beta, gamma, theta = binnedTrain._Train._params.getParams() 
            
            from scipy.stats.distributions import norm
            def loss_function_supnormalized(abg):
                error = .0;
                for (phi_m, phi_idx) in zip(phis, xrange(N_phi)):
                    Is = bins[phi_m]['Is']
                    N_Is  = len(Is);
                    Tf = amax(Is);
                     
                    a,b,g = abg[0], abg[1], abg[2]
                    movingThreshold = getMovingThreshold(a,g, phi_m,binnedTrain.theta)
                    
                    def LHS(ts):
                        LHS_numerator = movingThreshold(ts) *sqrt(2.)
                        LHS_denominator = b * sqrt(1 - exp(-2*ts))
                        return 1 -  norm.cdf(LHS_numerator / LHS_denominator);
                      
                    def RHS(ts):
                        if False == iterable(ts):
                            ts =  array([ts])
                        lIs = tile(Is,  len(ts) ).reshape((len(ts), len(Is))).transpose()
                        lts = tile(ts, (len(Is),1 ) )
                        mask = lIs < lts
                        taus = (lts - lIs); #*mask
                        #NOTE BELOW WE use abs(taus) since for non-positive taus we will mask away anyway:
                        numerator = (movingThreshold(lts) - movingThreshold(lIs)* exp(-abs(taus))) * sqrt(2.)
                        denominator = b *  sqrt(1. - exp(-2*abs(taus)))
                        
                        rhs = sum( (1. - norm.cdf(numerator/denominator))*mask, axis=0) / N_Is
                        return rhs
                    
                    ts_manual = linspace(1e-8, Tf+1e-2, 500)
                    lhs = LHS(ts_manual)
                    
                    rhs = RHS(ts_manual)
                    
                    difference = abs(lhs - rhs)/amax(lhs)
                    
                    val  = amax(difference);
                    
                    weight = len(Is)
                    lerror = val* weight;                    
                    
                    error += lerror                    
                return error
            
            def closs_function(abg):
                error = .0;
                for (phi_m, phi_idx) in zip(phis, xrange(N_phi)):
                    Is = bins[phi_m]['Is']
                    N_Is  = len(Is);
                    Tf = amax(Is);
                    ts = linspace(1e-8, Tf+1e-2, 500)
                    #Call C to compute the lhs - rhs
                    abgthphi = r_[abg, theta, phi_m]
                    difference = ext_fpc.FortetError(abgthphi, ts, Is)
                    raw_error  = amax(difference);
                    weighted_error = N_Is * raw_error;                    
                    error += weighted_error                    
                return error
            
            abg = array([alpha,beta,gamma])
            start = time.clock()
            py_error = loss_function_supnormalized(abg)
            print 'pytime = ', time.clock() - start
            start = time.clock()
            c_error = closs_function(abg) 
            print 'ctime = ', time.clock() - start
            
            errors.append([py_error,
                           c_error])
            
    pys = array(errors)[:,0]
    cs = array(errors)[:,1]
    figure(); hold(True)
    plot(pys, cs, '*'); plot(pys, pys, '-')
    

def CLossFHarness():
    
    errors = []
    
    for N_spikes, N_phi in zip( [100,1000],
                                [8,20]):
        phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
        base_name = 'sinusoidal_spike_train_N=%d_'%N_spikes


    for regime  in ['superT', 'crit', 'subT','superSin']:
        regime_label = base_name + regime

#        for sample_id in [1,23, 36, 77, 99]:
        for sample_id in [43]:
            file_name = regime_label + '_' + str(sample_id)
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
        
            bins = binnedTrain.bins;
            phis = bins.keys()
            N_phi = len(phis)
                        
            def closs_function(abg):
                error = .0;
                for (phi_m, phi_idx) in zip(phis, xrange(N_phi)):
                    Is = bins[phi_m]['Is']
                    N_Is  = len(Is);
                    Tf = amax(Is);
                    ts = linspace(1e-8, Tf+1e-2, 500)
                    #Call C to compute the lhs - rhs
                    abgthphi = r_[abg, theta, phi_m]
                    difference = ext_fpc.FortetError(abgthphi, ts, Is)
                    raw_error  = amax(difference);
                    weighted_error = N_Is * raw_error;                    
                    error += weighted_error                    
                return error
            
            alpha,  beta, gamma, theta = binnedTrain._Train._params.getParams() 
            start = time.clock()
            c_error = closs_function([alpha, beta, gamma]) 
            print 'ctime = ', time.clock() - start
              

if __name__ == '__main__':
    from pylab import *
    
#    VisualizeSinusoidallyDominating()
#    for regime  in ['superT', 'crit', 'subT','superSin']:
#    for regime  in ['superSin']:
#        sample_id = 1+randint(16)
#        sample_id = 1
#        Harness(sample_id, regime_name = regime,visualize=True)
    import cProfile
#    cProfile.run('FortetHarness()',None,'time')
#    cProfile.run('CLossFHarness()',None,'time')

#    FortetHarness()

        

    show()