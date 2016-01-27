'''
Created on May 4, 2012

@author: alex
'''

from BinnedSpikeTrain import BinnedSpikeTrain, intervalStats
#from Simulator import Path, OUSinusoidalParams
from SpikeTrainSimulator import SpikeTrain, OUSinusoidalParams


from numpy import linspace, float, arange, sum, max, min, sort
from numpy import sin, sqrt, ones_like, zeros_like, where, diff, pi, log, max , sign, cos
from numpy import zeros, ones, array, c_, r_, float64, Inf, ceil, arange, empty, interp, dot, sqrt, amax

from scipy import mat, matrix
from scipy.linalg import solve, lstsq
from scipy.optimize import nnls

from numpy import int0

def getKnots(ts, Ss, knot_count = 2):
    
    dts = diff(ts)
    
    dS = diff(Ss) / dts
    mean_dts = (dts[:-1] + dts[1:])/2.
    ddS =  diff(dS) / mean_dts 

    ddS = abs(ddS)
    sorted_ddS = sort(ddS)
    indxs = empty(knot_count, dtype = int0)
    for idx in xrange(0, knot_count):
        indxs[idx] = where( ddS == sorted_ddS[-1-idx] )[0][0]

#    indxs = empty(2, dtype = int0)
#    
#    indxs[0] = where(ddS == min(ddS))[0][0]
#    indxs[1] = where(ddS == max(ddS))[0][0] 
    
    return sort(ts[indxs])

    
    
#def initializeABG(ts, Ss, theta, phi):
#    
#    
#    tls = r_[(.0, tknots)]
#    trs = r_[(tknots, max(ts)+.1)]
#    
##    F_est = empty_like(Ss)
##    for tl, tr in zip(tls, trs):    
##        idxs = nonzero((ts >= tl)
##                        * (ts < tr) )
##        lts = ts[idxs] - tl
##        lS = Ss[idxs]
##        
##        M = mat( array([ len(lts) , sum(lts),
##                           sum(lts), sum(lts*lts) ]).reshape((2,2))  )
##        
##        v = array( [sum(lS),
##                    sum( lS *lts)])
##        
##        ab = solve(M,v) 
##        F_est[idxs] = ab[0] + ab[1]*lts
#
#    F_est = ones_like(Ss)
# 
#    t1 = tknots[0];
#    t3 = tknots[1];
#    t2 = (t1 + t3) /2.
#    
#    gc = lambda ti: (cos(theta*phi) - cos(theta*(ti+phi)))/(theta)
#    
#    M = mat( array([t1, sqrt(t1),gc(t1),
#                    t2, 0,      gc(t2),
#                    t3, -sqrt(t3), gc(t3)]).reshape((3,3))  )
#    v= ones(3) + .5*array([t1, t2, t3]);
#    a,b,g = solve(M,v)
#  
#    
##    M = mat( array([t1, sqrt(t1),
##                    t3, -sqrt(t3)]).reshape((2,2))  )
##    v= ones(2) + .5*array([t1, t3]);   
##    a,b = solve(M,v)
##    g =-1.0
##    
#    return [a,b,g], F_est, tknots

def guesstimate2pts(abg, binnedTrain, title_tag = ''):
#        print 'guesstimating based on 2 pts!!!'
        figure()
        bins = binnedTrain.bins
        phis = bins.keys()
        
        Tf = binnedTrain.getTf()

        N_phi = len(phis)
        N_pts = 2
        if(N_phi*N_pts < 3):
            raise Exception('Too few pts or phi_m\'s - underdetermined system' )
        
        M = matrix(zeros((N_pts*N_phi,3)))  
        v = empty(N_pts*N_phi)
        theta = binnedTrain.theta;
        
        for (phi_m, err_idx) in zip( sort(phis),
                                     xrange(N_phi)):
            
            unique_Is = bins[phi_m]['unique_Is']
            SDF =       bins[phi_m]['SDF']
            
            tknots = [unique_Is[SDF > (1 - .16)][-1],
                      unique_Is[SDF > .16][-1] ];
            t1 = tknots[0];
            t2 = tknots[1];
            
            ax = subplot(len(phis),1, err_idx + 1)
            plot(unique_Is, SDF, 'rx')
            hold (True)
            plot(tknots, interp(tknots, unique_Is, SDF) , 'k*', markersize = 18)
            
            gc = lambda ti: (cos(theta*phi_m) - cos(theta*(ti+phi_m)))/(theta)
    
            m_indxs = arange(N_pts*err_idx,N_pts*err_idx+N_pts)
            M[m_indxs,:] = array([t1, sqrt(t1),gc(t1),
                                  t2, -sqrt(t2), gc(t2)]).reshape((N_pts,3))  
            v[m_indxs]= ones(N_pts) + .5*array([t1, t2]);
            
            #DECORATE:
            if (0 == err_idx):
                title(title_tag + ' abg = (%.2g,%.2g,%.2g)' %(abg[0], abg[1], abg[2]) )
            if len(bins.keys()) != err_idx+1:
                setp(ax.get_xticklabels(), visible=False)
            else:
                xlabel('$t$', fontsize = 18)

            ylim((.0, 1.05))
            xlim((.0,  Tf))

            annotate('$\phi_{norm}$ = %.2g'%(phi_m / 2 / pi * binnedTrain.theta), (.1, .5), ) 
            
        abg_guess, res, rank, ss = lstsq(M, v)
        
#        if min(abg_guess) < .0:
#            print 'LS guess < 0 - trying optimize.nnls'
#            abg_guess, res = nnls(M,v)
            
#        print [binnedTrain._Path._params._alpha, binnedTrain._Path._params._beta, binnedTrain._Path._params._gamma] 
        return  abg_guess
        

def guesstimate3pts(abg, binnedTrain, title_tag = ''):
#        print 'guesstimating based on 3 pts!!!'
        figure()
        bins = binnedTrain.bins
        phis = bins.keys()
        
        Tf = binnedTrain.getTf()

        N_phi = len(phis)
        
        M = matrix(zeros((3*N_phi,3)))  
        v = empty(3*N_phi)
        theta = binnedTrain.theta;

        
        for (phi_m, err_idx) in zip( sort(phis),
                                     xrange(N_phi)):
            
            unique_Is = bins[phi_m]['unique_Is']
            SDF =       bins[phi_m]['SDF']
            
            tknots = [unique_Is[SDF > (1 - .16)][-1],
                      unique_Is[SDF > .5][-1],
                      unique_Is[SDF > .16][-1] ];
            t1 = tknots[0];
            t2 = tknots[1];
            t3 = tknots[2];
            
            ax = subplot(len(phis),1, err_idx + 1)
            plot(unique_Is, SDF, 'rx')
            hold (True)
            plot(tknots, interp(tknots, unique_Is, SDF) , 'k*', markersize = 18)
            
            gc = lambda ti: (cos(theta*phi_m) - cos(theta*(ti+phi_m)))/(theta)
    
            m_indxs = arange(3*err_idx,3*err_idx+3)
            M[m_indxs,:] = array([t1, sqrt(t1),gc(t1),
                                                t2, 0,      gc(t2),
                                                t3, -sqrt(t3), gc(t3)]).reshape((3,3))  
            v[m_indxs]= ones(3) + .5*array([t1, t2, t3]);
            
            #DECORATE:
            if (0 == err_idx):
                title(title_tag + ' abg = (%.2g,%.2g,%.2g)' %(abg[0], abg[1], abg[2]) )
            if len(bins.keys()) != err_idx+1:
                setp(ax.get_xticklabels(), visible=False)
            else:
                xlabel('$t$', fontsize = 18)

            ylim((.0, 1.05))
            xlim((.0,  Tf))

            annotate('$\phi_{norm}$ = %.2g'%(phi_m / 2 / pi * binnedTrain.theta), (.1, .5), ) 
            
        abg_guess, res, rank, ss = lstsq(M, v)
        
#        if min(abg_guess) < .0:
#            print 'LS guess < 0 - trying optimize.nnls'
#            abg_guess, res = nnls(M,v)
            
#        print [binnedTrain._Path._params._alpha, binnedTrain._Path._params._beta, binnedTrain._Path._params._gamma] 
        return abg_guess
        

def guesstimate5pts(abg, binnedTrain, title_tag = ''):
#        print 'guesstimating based on 5 pts:'
        figure()
        bins = binnedTrain.bins
        phis = bins.keys()
        
        Tf = binnedTrain.getTf()

        N_phi = len(phis)
        N_pts = 5
        if(N_phi*N_pts < 3):
            raise Exception('Too few pts or phi_m\'s - underdetermined system' )
        
        M = matrix(zeros((N_pts*N_phi,3)))  
        v = empty(N_pts*N_phi)
        theta = binnedTrain.theta;
        
        for (phi_m, err_idx) in zip( sort(phis),
                                     xrange(N_phi)):
            
            unique_Is = bins[phi_m]['unique_Is']
            SDF =       bins[phi_m]['SDF']
            
            tknots = [unique_Is[SDF < (1 - .022)][0],
                      unique_Is[SDF < (1 - .158)][0],
                      unique_Is[SDF > .500][-1],
                      unique_Is[SDF > .158][-1],
                      unique_Is[SDF > .022][-1] ];
            t1 = tknots[0];
            t2 = tknots[1];
            t_mid = tknots[2];
            t4 = tknots[3];
            t5 = tknots[4];
            
            ax = subplot(len(phis),1, err_idx + 1)
            plot(unique_Is, SDF, 'rx')
            hold (True)
            plot(tknots, interp(tknots, unique_Is, SDF) , 'k*', markersize = 18)
            
            gc = lambda ti: (cos(theta*phi_m) - cos(theta*(ti+phi_m)))/(theta)
    
            m_indxs = arange(N_pts*err_idx,N_pts*err_idx+N_pts)
            M[m_indxs,:] = array([t1, 2.*sqrt(t1), gc(t1),
                                  t2, sqrt(t2), gc(t2),
                                  t_mid, .0, gc(t_mid),
                                  t4, -sqrt(t4),   gc(t4),
                                  t5, -2.*sqrt(t5), gc(t5)]).reshape((N_pts,3))  
            v[m_indxs]= ones(N_pts) + .5*array([t1, t2, t_mid, t4, t5]);
            
            #DECORATE:
            if (0 == err_idx):
                title(title_tag + ' abg = (%.2g,%.2g,%.2g)' %(abg[0], abg[1], abg[2]) )
            if len(bins.keys()) != err_idx+1:
                setp(ax.get_xticklabels(), visible=False)
            else:
                xlabel('$t$', fontsize = 18)

            ylim((.0, 1.05))
            xlim((.0,  Tf))

            annotate('$\phi_{norm}$ = %.2g'%(phi_m / 2 / pi * binnedTrain.theta), (.1, .5), ) 
            
        abg_guess, res, rank, ss = lstsq(M, v)
        
#        if min(abg_guess) < .0:
#            print 'unconstrained LS guess < 0 - trying optimize.nnls'
#            abg_guess, res = nnls(M,v)
            
#        print [binnedTrain._Path._params._alpha, binnedTrain._Path._params._beta, binnedTrain._Path._params._gamma] 
#        print abg_guess
        return abg_guess


def initialize_right_1std(binnedTrain):
    bins = binnedTrain.bins
    phis = bins.keys()
    
    Tf = binnedTrain.getTf()

    N_phi = len(phis)
    N_pts = 1
    if(N_phi*N_pts < 3):
        raise Exception('Too few pts or phi_m\'s - underdetermined system' )
    
    M = matrix(zeros((N_pts*N_phi,3)))  
    v = empty(N_pts*N_phi)
    theta = binnedTrain.theta;
    
    for (phi_m, err_idx) in zip( sort(phis),
                                 xrange(N_phi)):
        
        unique_Is = bins[phi_m]['unique_Is']
        SDF =       bins[phi_m]['SDF']
        
        t2 = unique_Is[SDF < (1 - .158)][0]
        
        gc = lambda ti: (cos(theta*phi_m) - cos(theta*(ti+phi_m)))/(theta)

        m_indxs = arange(N_pts*err_idx,N_pts*err_idx+N_pts)
        M[m_indxs,:] = array([t2, sqrt(t2), gc(t2)])  
        v[m_indxs]   = ones(N_pts) + .5*array([t2]);
                    
    abg_guess, res, rank, ss = lstsq(M, v)
    
    return abg_guess



def initialize_right_2std(binnedTrain, cap_beta_gamma = False):
    bins = binnedTrain.bins
    phis = bins.keys()
    
    Tf = binnedTrain.getTf()

    N_phi = len(phis)
    N_pts = 2
    if(N_phi*N_pts < 3):
        raise Exception('Too few pts or phi_m\'s - underdetermined system' )
    
    M = matrix(zeros((N_pts*N_phi,3)))  
    v = empty(N_pts*N_phi)
    theta = binnedTrain.theta;
    
    for (phi_m, err_idx) in zip( sort(phis),
                                 xrange(N_phi)):
        
        unique_Is = bins[phi_m]['unique_Is']
        SDF =       bins[phi_m]['SDF']
        t1 = unique_Is[SDF < (1 - .022)][0]
        t2 = unique_Is[SDF < (1 - .158)][0]
        
        gc = lambda ti: (cos(theta*phi_m) - cos(theta*(ti+phi_m)))/(theta)

        m_indxs = arange(N_pts*err_idx,N_pts*err_idx+N_pts)
        M[m_indxs,:] = array([t1, 2.*sqrt(t1), gc(t1),
                              t2, sqrt(t2), gc(t2)]).reshape((N_pts,3))  
        v[m_indxs]   = ones(N_pts) + .5*array([t1, t2]);
                    
    abg_guess, res, rank, ss = lstsq(M, v)
    
    if cap_beta_gamma:
        abg_guess[1] = amax([.1, abg_guess[1]])
        abg_guess[2] = amax([.0, abg_guess[2]])
        
    return abg_guess


def initialize2_tau(binnedTrain):
    bins = binnedTrain.bins
    phis = bins.keys()
    
    Tf = binnedTrain.getTf()

    N_phi = len(phis)
    N_pts = 2
    if(N_phi*N_pts < 3):
        raise Exception('Too few pts or phi_m\'s - underdetermined system' )
    
    M = matrix(zeros((N_pts*N_phi,4)))  
    v = empty(N_pts*N_phi)
    theta = binnedTrain.theta;
    
    for (phi_m, err_idx) in zip( sort(phis),
                                 xrange(N_phi)):
        
        unique_Is = bins[phi_m]['unique_Is']
        SDF =       bins[phi_m]['SDF']
        tknots = [unique_Is[SDF < (1 - .022)][0],
                      unique_Is[SDF < (1 - .158)][0],
                      unique_Is[SDF > .500][-1],
                      unique_Is[SDF > .158][-1],
                      unique_Is[SDF > .022][-1] ];
        t1 = tknots[0];
        t2 = tknots[1];
       
        gc = lambda ti: (cos(theta*phi_m) - cos(theta*(ti+phi_m)))/(theta)

        m_indxs = arange(N_pts*err_idx,N_pts*err_idx+N_pts)
        M[m_indxs,:] = array([t1, 2.*sqrt(t1), gc(t1), -.5*t1,
                              t2, sqrt(t2), gc(t2), -.45*t2]).reshape((N_pts,4))  
        v[m_indxs]   = ones(N_pts,float_);
                    
    abgt_guess, res, rank, ss = lstsq(M, v)
    
    abgt_guess = r_[(abgt_guess[:3], 1./abgt_guess[3])]
    return abgt_guess



def initialize5_tau(binnedTrain):
    bins = binnedTrain.bins
    phis = bins.keys()
    
    Tf = binnedTrain.getTf()

    N_phi = len(phis)
    N_pts = 5
    if(N_phi*N_pts < 3):
        raise Exception('Too few pts or phi_m\'s - underdetermined system' )
    
    M = matrix(zeros((N_pts*N_phi,4)))  
    v = empty(N_pts*N_phi)
    theta = binnedTrain.theta;
    
    for (phi_m, err_idx) in zip( sort(phis),
                                 xrange(N_phi)):
        
        unique_Is = bins[phi_m]['unique_Is']
        SDF =       bins[phi_m]['SDF']
        tknots = [unique_Is[SDF < (1 - .022)][0],
                      unique_Is[SDF < (1 - .158)][0],
                      unique_Is[SDF > .500][-1],
                      unique_Is[SDF > .158][-1],
                      unique_Is[SDF > .022][-1] ];
        t1 = tknots[0];
        t2 = tknots[1];
        t_mid = tknots[2];
        t4 = tknots[3];
        t5 = tknots[4];
        
        gc = lambda ti: (cos(theta*phi_m) - cos(theta*(ti+phi_m)))/(theta)

        m_indxs = arange(N_pts*err_idx,N_pts*err_idx+N_pts)
        M[m_indxs,:] = array([t1, 2.*sqrt(t1), gc(t1),  -.5*t1,
                              t2, sqrt(t2), gc(t2),     -.45*t2,
                              t_mid, .0, gc(t_mid),     -.4*t_mid,
                              t4, -sqrt(t4),   gc(t4),  -.35*t4,
                              t5, -2.*sqrt(t5), gc(t5), -.3*t5]).reshape((N_pts,4))  
        v[m_indxs]   = ones(N_pts,float_);
                    
    abgt_guess, res, rank, ss = lstsq(M, v)
    
    abgt_guess = r_[(abgt_guess[:3], 1./abgt_guess[3])]
    return abgt_guess


def initialize5(binnedTrain):
        bins = binnedTrain.bins
        phis = bins.keys()
        
        Tf = binnedTrain.getTf()

        N_phi = len(phis)
        N_pts = 5
        if(N_phi*N_pts < 3):
            raise Exception('Too few pts or phi_m\'s - underdetermined system' )
        
        M = matrix(zeros((N_pts*N_phi,3)))  
        v = empty(N_pts*N_phi)
        theta = binnedTrain.theta;
        
        for (phi_m, err_idx) in zip( sort(phis),
                                     xrange(N_phi)):
            
            unique_Is = bins[phi_m]['unique_Is']
            SDF =       bins[phi_m]['SDF']
            
            tknots = [unique_Is[SDF < (1 - .022)][0],
                      unique_Is[SDF < (1 - .158)][0],
                      unique_Is[SDF > .500][-1],
                      unique_Is[SDF > .158][-1],
                      unique_Is[SDF > .022][-1] ];
            t1 = tknots[0];
            t2 = tknots[1];
            t_mid = tknots[2];
            t4 = tknots[3];
            t5 = tknots[4];
            
            gc = lambda ti: (cos(theta*phi_m) - cos(theta*(ti+phi_m)))/(ti*theta)
    
            m_indxs = arange(N_pts*err_idx,N_pts*err_idx+N_pts)
            M[m_indxs,:] = array([t1, 2.*sqrt(t1), gc(t1),
                                  t2, sqrt(t2), gc(t2),
                                  t_mid, .0, gc(t_mid),
                                  t4, -sqrt(t4),   gc(t4),
                                  t5, -2.*sqrt(t5), gc(t5)]).reshape((N_pts,3))  
            v[m_indxs]= ones(N_pts) + .5*array([t1, t2, t_mid, t4, t5]);
            
                        
        abg_guess, res, rank, ss = lstsq(M, v)
        
        return abg_guess
        

def regularCase():
    N_phi = 20;
#    print 'N_phi = ', N_phi
    
    dphi = 1/(2.*N_phi)
    phis =  linspace(dphi, 1. - dphi, N_phi)

#    file_name = 'sinusoidal_spike_train_T=1000.path'

#    file_name = 'sinusoidal_spike_train_T=6000_2.path'
#    file_name = 'sinusoidal_spike_train_T=6000_3.path'
#    file_name = 'sinusoidal_spike_train_T=6000_crit.path'

#    file_name = 'sinusoidal_spike_train_T=10000_crit.path'
#    file_name = 'sinusoidal_spike_train_T=10000_crit_2.path'
#    file_name = 'sinusoidal_spike_train_T=10000_crit_3.path'

#    file_name = 'sinusoidal_spike_train_T=10000_superSin.path'
#    file_name = 'sinusoidal_spike_train_T=10000_superSin_2.path'
#    file_name = 'sinusoidal_spike_train_T=10000_superSin_3.path'

#    file_name = 'sinusoidal_spike_train_T=25000_subT.path'
#    file_name = 'sinusoidal_spike_train_T=25000_subT_2.path'
#    file_name = 'sinusoidal_spike_train_T=25000_subT_3.path'
    
#    for file_name in ['sinusoidal_spike_train_T=10000_crit.path',
#                      'sinusoidal_spike_train_T=10000_crit_2.path' ,
#                      'sinusoidal_spike_train_T=10000_crit_3.path']:
##        print 'critical case:'
#    intervalStats(file_name)

#    for s_id, file_name in zip( xrange(3),
#                      ['sinusoidal_spike_train_T=10000_superSin.path',
#                        'sinusoidal_spike_train_T=10000_superSin_2.path',
#                        'sinusoidal_spike_train_T=10000_superSin_3.path']):

    for s_id, file_name in zip(xrange(3), ['sinusoidal_spike_train_T=25000_subT.path',
                                           'sinusoidal_spike_train_T=25000_subT_2.path',
                                           'sinusoidal_spike_train_T=25000_subT_3.path']):

#    for s_id, file_name in zip( xrange(3),
#                      ['sinusoidal_spike_train_T=6000.path',
#                       'sinusoidal_spike_train_T=6000_2.path',
#                       'sinusoidal_spike_train_T=6000_3.path']):
    
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
    
        phi_omit = None
    #    phi_omit = r_[(linspace(.15, .45, 4),linspace(.55,.95,5) )]  *2*pi/ binnedTrain.theta
        binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh=None)       

        ps = binnedTrain._Path._params
        abg_true = array((ps._alpha, ps._beta, ps._gamma))
        
        if 0 == s_id:
            print ' & $\\a$ & $\\b$ & $\\g$ \\\\'
            print '\hline'
            print 'true: & %.3g & %.3g & %.3g \\\\ ' %(binnedTrain._Path._params._alpha, binnedTrain._Path._params._beta, binnedTrain._Path._params._gamma)
        
        print '\hline '

        abg_2 = guesstimate2pts(abg_true, binnedTrain)
        abg_3 = guesstimate3pts(abg_true, binnedTrain)
        abg_5 = guesstimate5pts(abg_true, binnedTrain)
        
        for abg, N_pts in zip([abg_2, abg_3, abg_5],
                               [2,3,5]):
                print '$N_p = %d'%N_pts + '$ & %.3g & %.3g & %.3g \\\\' %(abg[0],abg[1], abg[2]);
            

from DataHarvester import DataHarvester
import time

def BatchInit5_vs_InitRightstd():
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)

    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_N=1000_'

    D = DataHarvester('InitComparison4x100_N100')
    for regime_name, T_thresh in zip(['superT', 'subT', 'crit', 'superSin'],
                                                       [4., 32, 16., 16.]):
        regime_label = base_name + regime_name
            
        for sample_id in xrange(1,17):
            file_name = regime_label + '_' + str(sample_id)
            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            D.setRegime(regime_name,abg_true, -.1)
            
            phi_omit = None
            binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh=T_thresh)
            Tf = binnedTrain.getTf()
            D.addSample(sample_id, Tf, binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
        
            start = time.clock()
            abg_init = initialize5(binnedTrain)
            finish = time.clock()
            D.addEstimate(sample_id, 'Init5pts', abg_init, finish-start) 
            
            start = time.clock()
            abg_init = initialize_right_2std(binnedTrain)
            finish = time.clock()
            D.addEstimate(sample_id, 'Init_right_2std', abg_init, finish-start) 
                                
    D.closeFile() 
   
    print 'batch time = ', (time.clock() - batch_start) / 60.0, ' mins'



def InitTauSamples():
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
    base_name = 'sinusoidal_spike_train_N=1000_'

    for regime_name, T_thresh in zip(['superT', 'subT', 'crit', 'superSin'],
                                                       [4., 32, 16., 16.]):
       
        regime_label = base_name + regime_name
        sample_id = randint(1,17)
        file_name = regime_label + '_' + str(sample_id)
        print file_name
        
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
        ps = binnedTrain._Train._params
        abg_true = array((ps._alpha, ps._beta, ps._gamma))
        print abg_true
            
        phi_omit = None
        binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh=T_thresh)
        Tf = binnedTrain.getTf()
                    
        abgt_init = initialize2_tau(binnedTrain)
        print '%.2g,%.2g,%.2g,%.2g' %(abgt_init[0],
                                      abgt_init[1],abgt_init[2],abgt_init[3])
        
        abgt_init = initialize5_tau(binnedTrain)
        print '%.2g,%.2g,%.2g,%.2g' %(abgt_init[0],
                                      abgt_init[1],abgt_init[2],abgt_init[3])
        
        print " "


if __name__ == '__main__':
    from pylab import *
    
#    InitTauSamples()
#    regularCase()
    BatchInit5_vs_InitRightstd()
    
#    show()1