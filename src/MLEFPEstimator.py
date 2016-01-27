'''
Created on May 3, 2012

@author: alex
'''
from __future__ import division

from BinnedSpikeTrain import BinnedSpikeTrain, intervalStats, loadPath
from SpikeTrainSimulator import SpikeTrain, OUSinusoidalParams

from FPMultiPhiSolver import FPMultiPhiSolver

from numpy import *

from numpy.random import randn, randint

import time
from InitBox import guesstimate5pts, initialize5, initialize_right_2std
#from scipy.optimize.lbfgsb import fmin_l_bfgs_b
#from scipy.optimize.cobyla import fmin_cobyla
from scipy.optimize.slsqp import fmin_slsqp
from DataHarvester import DataPrinter, DataAnalyzer, DataHarvester
from FortetEstimator import FortetEstimatorSup


class MLEBinnedSpikeTrain():
    def __init__(self, T, N_phi, T_thresh = None):
        ''' bins the spike intervals (diff(spike_ts)) into bins corresponding to the phase angle at the beginning of each interval'''
        self._Train = T
        self._spike_ts = T._spike_ts;
        self.theta = T._params._theta;
        
        self.regenerateIntervalsPhis(T_thresh)
        
        self._dphi = self.getPeriod() / (N_phi)
        self.phi_ms =  arange(.0, self.getPeriod() , self._dphi) 
        
        self._binIt()
            
    @classmethod
    def initFromFile(cls, file_name, N_phi, T_thresh=None):
        T = loadPath(file_name)
        return MLEBinnedSpikeTrain(T, N_phi, T_thresh)
    
    def getPeriod(self):
        return 2. * pi / self.theta
    def getTf(self):
        Tf = amax(self.getIs())
        return Tf;

    def getSpikeCount(self):
        return len(self._Train._spike_ts)
    def getIntervalCount(self):
        return len(self._spike_Is)
    def getBinCount(self):
        return len(self.phi_ms)+1
    def getIs(self):
        return self._spike_Is;
    
    def _binIt(self):
        Phis = self._spike_Phis
        dphi = self._dphi 
        phi_ms = r_[self.phi_ms, self.getPeriod()];
        phi_m_indexes = r_[arange(self.getBinCount()),0]
        
        self.phi_minus_indxs = empty_like(Phis, dtype = int)
        self.phi_plus_indxs = empty_like(Phis, dtype = int)
        self.phi_minus_weights = empty_like(Phis)
        self.phi_plus_weights = empty_like(Phis)
        for interval_idx, phi_star in enumerate(Phis):
            abs_differences = abs(phi_ms - phi_star)
            phi_m_index = where(abs_differences == amin(abs_differences))[0][0]
            
            minus_idx = len(phi_ms); plus_idx = len(phi_ms)
            
            phi_m = phi_ms[phi_m_index]
            if (phi_m <= phi_star):
                minus_idx = phi_m_index                
                plus_idx =  phi_m_index+1
            else:
                minus_idx = phi_m_index-1                
                plus_idx =  phi_m_index
                
            phi_minus = phi_ms[minus_idx];
            phi_plus =  phi_ms[plus_idx];
            
            self.phi_minus_weights[interval_idx] = (phi_plus - phi_star) / dphi 
            self.phi_plus_weights[interval_idx]  = (phi_star - phi_minus)/ dphi
            
            self.phi_minus_indxs[interval_idx] = minus_idx  
            self.phi_plus_indxs[interval_idx] =  mod(plus_idx, len(self.phi_ms))

#        $TODO: 
#        if a phi_m is not in the phi_minus_indxs or phi_plus_indxs
#        we can drop it from the list of phis
    
    def getTindxs(self, ts):
        spike_t_indexes = empty_like(self._spike_Is, dtype = int)
#            TODO: this can be fasterized:
        for i_idx, I_star in enumerate(self._spike_Is):
            t_idx = where(abs(ts-I_star) == amin(abs(ts-I_star)))[0][0];
            if ts[t_idx] > I_star:
                t_idx-=1
                    
            spike_t_indexes[i_idx] = t_idx  
    
        return spike_t_indexes
         

    def regenerateIntervalsPhis(self, T_thresh):
        '''Remove Is, phis from bins st. Is  > T_thresh'''
        spike_Is = r_[(self._spike_ts[0], diff(array(self._spike_ts)))];
        spike_Phis = r_[(.0, mod(self._spike_ts[:-1], 2.*pi / self.theta))]
        
        if None == T_thresh:
            self._spike_Is = spike_Is
            self._spike_Phis = spike_Phis
        else:
            self._spike_Is   = spike_Is[spike_Is <= T_thresh]
            self._spike_Phis = spike_Phis[spike_Is <= T_thresh] 
            
    
    def visualize(self, title_tag = '', save_fig_name = ''):
        '''Pylab visualize the binning procedure '''
        dphi = self._dphi
        P = self._Train
        #Visualize time:
        mpl.rcParams['figure.subplot.left'] = .15
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .125
        mpl.rcParams['figure.subplot.top'] = .925
        mpl.rcParams['figure.subplot.wspace'] = .05
#        mpl.rcParams['figure.figsize'] = 15.5, 4.5
        
        figure()
        ax  = subplot(111)
        def relabel_major(x, pos):
            if x < 0:
                    return ''
            else:
                    return '$%.1f$' %x
        for phi in self.phi_ms:
            axvline(x = phi, color='r')

        xlim((.0, self.getPeriod()))
        xlabel(r'$\phi_n$', fontsize = 28)
        ylabel(r'$i_n$', fontsize = 28)
        tick_locs = self.phi_ms
        tick_lbls = [r'$\frac{%d \pi}{%d}$'%(int(round(len(self.phi_ms)*x/pi)),
                                             len(self.phi_ms))  for x in self.phi_ms]
        xticks(tick_locs, tick_lbls)
        tick_params(labelsize = 20)
        ax.yaxis.set_major_formatter(FuncFormatter(relabel_major))
#        def add_inner_title(ax, title, loc, size=None, **kwargs):
#            from matplotlib.offsetbox import AnchoredText
#            from matplotlib.patheffects import withStroke
#            if size is None:
#                size = dict(size=plt.rcParams['legend.fontsize'])
#            at = AnchoredText(title, loc=loc, prop=size,
#                              pad=0., borderpad=0.5,
#                              frameon=False, **kwargs)
#            ax.add_artist(at)
#            at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
#            return at
#        t = add_inner_title(ax, 'A', loc=2, size=dict(size=20))
#        t.patch.set_ec("none")
#        t.patch.set_alpha(0.5)
        colours = ['g', 'r', 'b']
        for i_idx in xrange(self.getIntervalCount()):
            phi, Is = self.getPhiInterval(i_idx)
            col = colours[randint(3)]
            plot(phi, Is, col+'o', markersize = 16);
            
            phi_minus  = self.phi_ms[self._phi_minus_indxs[i_idx]]
            phi_plus   = self.phi_ms[self._phi_plus_indxs[i_idx]]
            
            minus_size = ceil(16*self._phi_minus_weights[i_idx])+2
            plus_size  = ceil(16*self._phi_plus_weights[i_idx])+2
            plot(phi_minus, Is, col+'<', markersize = minus_size)
            plot(phi_plus,  Is, col+'>', markersize = plus_size)
                
        
    def getPhiInterval(self, index):
        return self._spike_Phis[index],  self._spike_Is[index] 

    def getRandomPhiInterval(self):
        interval_index = randint(self.getSpikeCount())
        return self.getPhiInterval(interval_index)


def getApproximatePhis(phi_star, approximate_phis, theta):
    abs_differences = abs(approximate_phis - phi_star)
    
    phi_m_index = where(abs_differences == amin(abs_differences))[0][0]
    phi_m = approximate_phis[phi_m_index]
    
    phi_minus = phi_m; 
    phi_plus = phi_m;
    if phi_m <= phi_star:
        if  (phi_m_index == len(approximate_phis)-1):
            phi_plus = approximate_phis[0] + 2.0*pi /theta
        else:
            phi_plus = approximate_phis[phi_m_index+1]
    else:
        if  phi_m_index == 0:
            phi_minus = approximate_phis[-1] - 2.0*pi /theta
        else:
            phi_minus = approximate_phis[phi_m_index-1]
                    
    return phi_m, phi_minus,  phi_plus

def gettsIndex(ts, I_star):
    tm_idx = where(abs(ts-I_star) == amin(abs(ts-I_star)))[0][0];
    tp_idx = tm_idx
    
    t_minus = ts[tm_idx]
    
    if (t_minus < I_star):
        tp_idx+=1
    else:
        tm_idx-=1
        
    return tm_idx, tp_idx
def getDeltaPhiWeights(phi_star, phi_minus, phi_plus):
    delta_phi = phi_plus - phi_minus;
    delta_phi_minus_weight  = (phi_plus - phi_star) / delta_phi 
    delta_phi_plus_weight   = (phi_star - phi_minus)/delta_phi
    if (delta_phi_minus_weight<.0 or delta_phi_plus_weight < .0):
        print phi_minus, phi_plus
    return delta_phi_minus_weight, delta_phi_plus_weight
            
    
def sandbox():
#    file_name = 'sinusoidal_spike_train_N=1000_subT_11'
#    file_name = 'sinusoidal_spike_train_N=1000_superSin_13'
    N_sub_samples = 32
    N_samples = N_sub_samples*8;

#    N_phi = 16;
    
    #results banks:
    likelihoods = empty((4, 4, N_samples))
    errors = empty((4, 3, N_samples))
    
    seed(2013)
    for regime_idx, tag, N_phi in zip(xrange(4),
                                      ['superSin', 'crit', 'superT', 'subT'],
                                      [16,16,16,16]):
        file_name = 'sinusoidal_spike_train_N=1000_%s_22'%(tag)
        print file_name
        normalized_phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, normalized_phis)
        theta = binnedTrain.theta;
        real_phis = normalized_phis * 2.0 * pi / theta;
    
        for sample_idx in xrange(N_samples):
            if 0 == mod(sample_idx, N_sub_samples):
                train_id = randint(1,101) 
                file_name = 'sinusoidal_spike_train_N=1000_%s_%d'%(tag, train_id)
                print file_name
                binnedTrain = BinnedSpikeTrain.initFromFile(file_name, normalized_phis)
                
            phi_star,I_star = binnedTrain.getRandomPhiInterval()
            print 'phi_star_normalized, I_star: %.3f, %.3f' %(phi_star/ (2*pi/theta), I_star)
            
            phi_m, phi_minus, phi_plus = getApproximatePhis(phi_star, real_phis,theta)
            delta_phi_minus_weight, delta_phi_plus_weight = getDeltaPhiWeights(phi_star, phi_minus, phi_plus)
            
            #phi_star_idx = 0; phi_m_idx = 1; etc...
            solver_phis = [phi_star, phi_m, phi_minus, phi_plus]
#            print 'solver_phis = ', solver_phis
#            print 'weights = %.3f, %.3f'%(delta_phi_minus_weight, delta_phi_plus_weight)   
            
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            
            abg = abg_true
            
            Tf = I_star + .2;
            
            dx = .025; 
            x_min = FPMultiPhiSolver.calculate_xmin(Tf, abg, theta)
            dt = FPMultiPhiSolver.calculate_dt(dx,abg, x_min, factor = 1.0)
            
            S = FPMultiPhiSolver(theta, solver_phis,
                                    dx, dt, Tf, x_min)
            S.setTf(Tf)
            
            Fs =  S.c_solve(abg)
            Fth = Fs[:,:,-1]
#            Fth_phis = S.solveFphi(abg, Fs)[:,:,-1]
            ts = S._ts;
            
            tm_idx, tp_idx = gettsIndex(ts, I_star)
            delta_t = S._dt
            delta_phi = phi_star - phi_m
            
            #various approximations to the likelihood, L
            L_star = -diff(Fth[0, [tm_idx, tp_idx]]) / (delta_t)
            L_m    = -diff(Fth[1, [tm_idx, tp_idx]]) / (delta_t)
            L_plus_minus    = -(diff(Fth[2, [tm_idx, tp_idx]])*delta_phi_minus_weight + \
                                  diff(Fth[3, [tm_idx, tp_idx]])*delta_phi_plus_weight)/ (delta_t)
#            gradphi_g = -diff(Fth_phis[1, [tm_idx, tp_idx]]) / (delta_t);
#            logL_gradphi = log(L_m) +  delta_phi * gradphi_g / L_m
#            L_gradphi = exp(logL_gradphi)  
            L_gradphi = L_plus_minus
            
            if (.0 >= L_star*L_m*L_plus_minus):
                print 'negative likelihood encountered'
#            print 'di_F: %.4f,%.4f,%.4f' %(diF_star, diF_m, diF_plus_minus)
#            print 'error: %.4f,%.4f' %(abs(diF_star - diF_m), abs(diF_star-diF_plus_minus) )
            
            likelihoods[regime_idx, :, sample_idx] = r_[L_star,
                                                         L_m,
                                                          L_plus_minus,
                                                          L_gradphi]
            errors[regime_idx, :, sample_idx] = r_[abs(L_star - L_m),
                                                    abs(L_star-L_plus_minus),
                                                    abs(L_star - L_gradphi)]
            
        figure()
        plot(errors[regime_idx, 0,:], 'b', label='F_m')
        plot(errors[regime_idx, 1,:], 'r', label='F_min + F_plus')
        legend(); title(tag, fontsize = 32)
    
    from numpy import save
    save('likelihoods',likelihoods)
    save('errors', errors)
   

def supersin_sandbox(N_phi = 32):
    N_sub_samples = 10
    N_samples = N_sub_samples*5;

    normalized_phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
    
    #results banks:
    likelihoods = empty((4, 4, N_samples))
    errors = empty((4, 3, N_samples))
    
    seed(2013)
    for regime_idx, tag in enumerate(['superSin']):
        file_name = 'sinusoidal_spike_train_N=1000_%s_22'%(tag)
        print file_name
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, normalized_phis)
        theta = binnedTrain.theta;
        real_phis = normalized_phis * 2.0 * pi / theta;
    
        for sample_idx in xrange(N_samples):
            if 0 == mod(sample_idx, N_sub_samples):
                train_id = randint(1,101) 
                file_name = 'sinusoidal_spike_train_N=1000_%s_%d'%(tag, train_id)
                print file_name
                binnedTrain = BinnedSpikeTrain.initFromFile(file_name, normalized_phis)
                
            phi_star,I_star = binnedTrain.getRandomPhiInterval()
            print 'phi_star_normalized, I_star: %.3f, %.3f' %(phi_star/ (2*pi/theta), I_star)
            
            phi_m, phi_minus, phi_plus = getApproximatePhis(phi_star, real_phis,theta)
            delta_phi_minus_weight, delta_phi_plus_weight = getDeltaPhiWeights(phi_star, phi_minus, phi_plus)
            
            #phi_star_idx = 0; phi_m_idx = 1; etc...
            solver_phis = [phi_star, phi_m, phi_minus, phi_plus]
#            print 'solver_phis = ', solver_phis
#            print 'weights = %.3f, %.3f'%(delta_phi_minus_weight, delta_phi_plus_weight)   
            
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            
            abg = abg_true
            
            Tf = I_star + .2;
            
            dx = .025; 
            x_min = FPMultiPhiSolver.calculate_xmin(Tf, abg, theta)
            dt = FPMultiPhiSolver.calculate_dt(dx,abg, x_min, factor = 1.0)
            
            S = FPMultiPhiSolver(theta, solver_phis,
                                    dx, dt, Tf, x_min)
            S.setTf(Tf)
            
            Fs =  S.c_solve(abg)
            Fth = Fs[:,:,-1]
            Fth_phis = S.solveFphi(abg, Fs)[:,:,-1]
            ts = S._ts;
            
            tm_idx, tp_idx = gettsIndex(ts, I_star)
            delta_t = S._dt
            delta_phi = phi_star - phi_m
            
            #various approximations to the likelihood, L
            L_star = -diff(Fth[0, [tm_idx, tp_idx]]) / (delta_t)
            L_m    = -diff(Fth[1, [tm_idx, tp_idx]]) / (delta_t)
            L_plus_minus    = -(diff(Fth[2, [tm_idx, tp_idx]])*delta_phi_minus_weight + \
                                  diff(Fth[3, [tm_idx, tp_idx]])*delta_phi_plus_weight)/ (delta_t)
            gradphi_g = -diff(Fth_phis[1, [tm_idx, tp_idx]]) / (delta_t);
            logL_gradphi = log(L_m) +  delta_phi * gradphi_g / L_m
            L_gradphi = exp(logL_gradphi)  
            
            'sanity check'
            approx_Fth_phi = .5 * sum(Fth[0, [tm_idx, tp_idx]] - Fth[1, [tm_idx, tp_idx]]) / (phi_star - phi_m)
            lFth_phi = .5* (sum(Fth_phis[1, [tm_idx, tp_idx]]))  
            if (approx_Fth_phi * lFth_phi < .0):
                print 'sanity inverse: approx:%.3f , adjoint_calc: %.3f'%(approx_Fth_phi,lFth_phi) 
            
            if (.0 >= L_star*L_m*L_plus_minus*L_gradphi):
                print 'negative likelihood encountered'
#            print 'di_F: %.4f,%.4f,%.4f' %(diF_star, diF_m, diF_plus_minus)
#            print 'error: %.4f,%.4f' %(abs(diF_star - diF_m), abs(diF_star-diF_plus_minus) )
            
            likelihoods[regime_idx, :, sample_idx] = r_[L_star,
                                                         L_m,
                                                          L_plus_minus,
                                                          L_gradphi]
            errors[regime_idx, :, sample_idx] = r_[abs(L_star - L_m),
                                                    abs(L_star-L_plus_minus),
                                                    abs(L_star - L_gradphi)]
            
        figure()
        plot(errors[regime_idx, 0,:], 'b', label='F_m')
        plot(errors[regime_idx, 1,:], 'r', label='F_min + F_plus')
        legend(); title(tag, fontsize = 32)
    
    from numpy import save
    save('likelihoods_supersin',likelihoods)
    save('errors_supersin', errors)


def thetas_sandbox(save_figs=False):
    N_samples = 100;
    N_phi = 64;
    normalized_phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
    
    #results banks:
#    seed(2013)
    thetas = [10, 20]
    likelihoods = empty((len(thetas), 4, N_samples))
    errors = empty((len(thetas), 3, N_samples))
    
    base_name = 'sinusoidal_spike_train_N=1000_critical_theta='
    for regime_idx, theta in enumerate(thetas):    
        sample_id = 17
        regime_name = 'theta%d'%theta
        regime_label = base_name + '%d'%theta            
        file_name = regime_label + '_%d'%sample_id
        print file_name
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, normalized_phis)
        theta = binnedTrain.theta;
        real_phis = normalized_phis * 2.0 * pi / theta;
        
        for sample_idx in xrange(N_samples): 
            phi_star,I_star = binnedTrain.getRandomPhiInterval()
            print 'phi_star_normalized, I_star: %.3f, %.3f' %(phi_star/ (2*pi/theta), I_star)
            
            phi_m, phi_minus, phi_plus = getApproximatePhis(phi_star, real_phis,theta)
            delta_phi_minus_weight, delta_phi_plus_weight = getDeltaPhiWeights(phi_star, phi_minus, phi_plus)
            
            #phi_star_idx = 0; phi_m_idx = 1; etc...
            solver_phis = [phi_star, phi_m, phi_minus, phi_plus]
#            print 'solver_phis = ', solver_phis
#            print 'weights = %.3f, %.3f'%(delta_phi_minus_weight, delta_phi_plus_weight)   
            
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            
            abg = abg_true
            
            Tf = I_star + .2;
            
            dx = .025; 
            x_min = FPMultiPhiSolver.calculate_xmin(Tf, abg, theta)
            dt = FPMultiPhiSolver.calculate_dt(dx,abg, x_min, factor = 1.0)
            
            S = FPMultiPhiSolver(theta, solver_phis,
                                    dx, dt, Tf, x_min)
            S.setTf(Tf)
            
            Fs =  S.c_solve(abg)
            Fth = Fs[:,:,-1]
#            Fth_phis = S.solveFphi(abg, Fs)[:,:,-1]
            ts = S._ts;
            
            tm_idx, tp_idx = gettsIndex(ts, I_star)
            delta_t = S._dt
            delta_phi = phi_star - phi_m
            
            #various approximations to the likelihood, L
            L_star = -diff(Fth[0, [tm_idx, tp_idx]]) / (delta_t)
            L_m    = -diff(Fth[1, [tm_idx, tp_idx]]) / (delta_t)
            L_plus_minus    = -(diff(Fth[2, [tm_idx, tp_idx]])*delta_phi_minus_weight + \
                                  diff(Fth[3, [tm_idx, tp_idx]])*delta_phi_plus_weight)/ (delta_t)
#            gradphi_g = -diff(Fth_phis[1, [tm_idx, tp_idx]]) / (delta_t);
#            logL_gradphi = log(L_m) +  delta_phi * gradphi_g / L_m
#            L_gradphi = exp(logL_gradphi)
            L_gradphi = L_plus_minus
              
            
#            print 'di_F: %.4f,%.4f,%.4f' %(diF_star, diF_m, diF_plus_minus)
#            print 'error: %.4f,%.4f' %(abs(diF_star - diF_m), abs(diF_star-diF_plus_minus) )
            
            likelihoods[regime_idx, :, sample_idx] = r_[L_star,
                                                         L_m,
                                                          L_plus_minus,
                                                           L_gradphi]
            errors[regime_idx, :, sample_idx] = r_[abs(L_star - L_m),
                                                    abs(L_star-L_plus_minus),
                                                    abs(L_star - L_gradphi)]
            
        figure()
        plot(errors[regime_idx, 0,:], 'b', label='F_m')
        plot(errors[regime_idx, 1,:], 'r', label='F_min + F_plus')
        legend(); title('theta= %.2f'%theta, fontsize = 32)
    
    from numpy import save
    save('likelihoods_thetas',likelihoods)
    save('errors_thetas', errors)
    
    
##        mpl.rcParams['figure.subplot.left'] = .05
##        mpl.rcParams['figure.subplot.right'] = .95
#    mpl.rcParams['figure.subplot.bottom'] = .125
##        mpl.rcParams['figure.subplot.top'] = .9
#    mpl.rcParams['figure.subplot.wspace'] = .2
##        mpl.rcParams['figure.subplot.hspace'] = .55
#    mpl.rcParams['figure.figsize'] = 15.5, 4.5
#    figure();     label_font_size = 16
#    ax = subplot(1,4, M_idx + 1)
#    phi_m =  phis[0]
#    phi_idx = nonzero( abs(phis - phi_m) < 1e-4)
#    lF = squeeze(Fs[phi_idx, :,-1])
#    hold (True)
#    plot(ts, lF, 'b',linewidth=2, label='Analytic'); 
#    plot(bins[phi_m]['unique_Is'], 
#             bins[phi_m]['SDF'], 'r+', markersize = 6, label='Data')
#        legend()
#        if (0 == M_idx):
#            ylabel(r'$\bar{G}(t)$', fontsize = label_font_size)
#        else:
#            setp(ax.get_yticklabels(), visible=False)
#        xlabel('$t$', fontsize = label_font_size)
#        for label in ax.xaxis.get_majorticklabels():
#            label.set_fontsize(label_font_size)
#        for label in ax.yaxis.get_majorticklabels():
#            label.set_fontsize(label_font_size)   
#
#        ylim((.0, 1.05))
#        t = add_inner_title(ax, inner_titles[M_idx], loc=3, size=dict(size=16))
#        t.patch.set_ec("none")
#        t.patch.set_alpha(0.5)
#        
##    get_current_fig_manager().window.showMaximized() 
#    if '' != save_figs:
#        file_name = os.path.join(FIGS_DIR, 'EffectOfM.pdf')
#        print 'saving to ', file_name
#        savefig(file_name, dpi=(300)) 
def  getAdaptedPhis(N_phi_per_quarter):
    phi_image = linspace(1/(2.*N_phi_per_quarter), 1. - 1/ (2.*N_phi_per_quarter), N_phi_per_quarter)
    phis = arcsin(phi_image)
    phis = r_[phis,
               (pi - phis)[::-1],
                phis + pi,
                 (2.*pi - phis)[::-1] ] / (2.0*pi)
    
    return phis
    
    
def adapted_sandbox(save_figs=False):
#    file_name = 'sinusoidal_spike_train_N=1000_subT_11'
#    file_name = 'sinusoidal_spike_train_N=1000_superSin_13'
    N_samples = 16;

    N_phi_per_quarter = 4;
    N_phi = 4*N_phi_per_quarter;
    normalized_phis =  getAdaptedPhis(N_phi_per_quarter) 
    diFs = empty((4, 3, N_samples))
    errors = empty((4, 2, N_samples))
    
    seed(2013)
    for regime_idx, tag in enumerate(['superSin', 'crit', 'superT', 'subT']):
        file_name = 'sinusoidal_spike_train_N=1000_%s_13'%tag
        binnedTrain = BinnedSpikeTrain.initFromFile(file_name, normalized_phis)
        theta = binnedTrain.theta;
        real_phis = normalized_phis * 2.0 * pi / theta;
                
        for sample_idx in xrange(N_samples): 
            phi_star,I_star = binnedTrain.getRandomPhiInterval()
            
            print 'phi_star, I_star: ', phi_star, I_star
            
            phi_m, phi_minus, phi_plus = getApproximatePhis(phi_star, real_phis,theta)
            delta_phi_minus_weight, delta_phi_plus_weight = getDeltaPhiWeights(phi_star, phi_minus, phi_plus)
            
            solver_phis = [phi_star, phi_m, phi_minus, phi_plus]
            
            ps = binnedTrain._Train._params
            abg = array((ps._alpha, ps._beta, ps._gamma))
            
            Tf = I_star + .2;
            
            dx = .025; 
            x_min = FPMultiPhiSolver.calculate_xmin(Tf, abg, theta)
            dt = FPMultiPhiSolver.calculate_dt(dx,abg, x_min, factor = 1.0)
            
            S = FPMultiPhiSolver(theta, solver_phis,
                                    dx, dt, Tf, x_min)
            S.setTf(Tf)
            
            Fth = S.c_solve(abg)[:,:,-1]
            ts = S._ts;
            
            tm_idx, tp_idx = gettsIndex(ts, I_star)
            
            diF_star = -diff(Fth[0, [tm_idx, tp_idx]]) / (S._dt)
            diF_m    = -diff(Fth[1, [tm_idx, tp_idx]]) / (S._dt)
            diF_plus_minus    = -(diff(Fth[2, [tm_idx, tp_idx]])*delta_phi_minus_weight + \
                                  diff(Fth[3, [tm_idx, tp_idx]])*delta_phi_plus_weight)/ (S._dt)
            
#            print 'di_F: %.4f,%.4f,%.4f' %(diF_star, diF_m, diF_plus_minus)
#            print 'error: %.4f,%.4f' %(abs(diF_star - diF_m), abs(diF_star-diF_plus_minus) )
            
            diFs[regime_idx, :, sample_idx] = r_[diF_star, diF_m, diF_plus_minus]
            errors[regime_idx, :, sample_idx] = r_[abs(diF_star - diF_m), abs(diF_star-diF_plus_minus)]
            
        figure()
        plot(errors[regime_idx, 0,:], 'b', label='F_m')
        plot(errors[regime_idx, 1,:], 'r', label='F_min + F_plus')
        legend(); title(tag, fontsize = 32)
    
    from numpy import save
    save('Ls_adapted',diFs)
    save('errors_adapted', errors)

def MLEEstimator(S, mleBinnedTrain, abg_init,
                 optim_mthd = 'nm'):
    
    param_brick_bounds = ((-1.,5.),
                          (.05, 5.),
                          (.0, 5.)  );
    cobyla_constraints = [lambda x: x[1] - param_brick_bounds[1][0],
                          lambda x: x[2] - param_brick_bounds[2][0]]
    
    minus_idxs = mleBinnedTrain.phi_minus_indxs
    plus_idxs  = mleBinnedTrain.phi_plus_indxs
    minus_weights = mleBinnedTrain.phi_minus_weights
    plus_weights  = mleBinnedTrain.phi_plus_weights
    
    #TODO:
              
    def func(abg):
        'rediscretize:'
        xmin = FPMultiPhiSolver.calculate_xmin(S.getTf(), abg, S._theta)
        dt = FPMultiPhiSolver.calculate_dt(S._dx, abg, xmin, factor = 2.)
#        print abg, dt, S._dx, xmin
        S.rediscretize(S._dx, dt, S.getTf(), xmin)
        
        'Solve it:'
#        print abg
#        start = time.clock()
        Fs = S.c_solve(abg)
#        solve_end = time.clock()
        spike_t_indexes = mleBinnedTrain.getTindxs(S._ts)
#        t_indxs_end = time.clock();
#        print abg, ' : solvetime = %f; indx_time = %f'%(solve_end  - start, t_indxs_end - solve_end) 
       
        'form (approximate) likelihood:'
        pdf = -diff(Fs[:,:,-1], axis = 1) / S._dt;
        likelihoods = pdf[minus_idxs, spike_t_indexes]*minus_weights + \
                      pdf[plus_idxs, spike_t_indexes]*plus_weights
        if amin(likelihoods) <= .0:
            likelihoods[likelihoods<=.0] = 1e-8
        
        'Return '
        return -sum(log(likelihoods))

#    if 'slsqp' == optim_mthd:
#        from scipy.optimize.slsqp import fmin_slsqp
#        print 'MLE error: = FP-C, sequential QP '
#        abg_slsqp, min_slsqp, its, imode, smode = fmin_slsqp(func, abg_init,
#                                                         ieqcons = cobyla_constraints,
#                                                acc = 1e-2, iter = 100,
#                                                full_output = True)
#        return abg_slsqp

    if 'bfgs' == optim_mthd:
        from scipy.optimize import fmin_l_bfgs_b
        print 'MLE error: = FP-C, l_BFGS_b '
        abg_bfgs, func_val, info_dict = fmin_l_bfgs_b(func, abg_init, None, 
                                 approx_grad=True,
                                 bounds=param_brick_bounds,
                                 m=10, factr=1e10,
                                 maxfun=200)
        return abg_bfgs

    elif 'tnc' == optim_mthd:
        from scipy.optimize import fmin_tnc
        print 'MLE error: = FP-C, truncated Newton '
        abg_tnc, nfeval, rc_code = fmin_tnc(func, abg_init,
                                          approx_grad=True,
                                          bounds = param_brick_bounds,
                                          ftol = 1e-2,
                                          maxfun=200)
        return abg_tnc

#        abg_cobyla = fmin_cobyla(func, abg_init,
#                                 cobyla_constraints,
#                                 maxfun=200)

    elif 'nm' == optim_mthd:
        print 'MLE error: = FP-C, Nelder-Mead'
        from scipy.optimize import fmin
        abg_nm, fopt, iter, funcalls, warnflag, allvecs = fmin(func, abg_init,
                                                                       maxiter = 200, 
                                                                       xtol = 1e-2, ftol = 1e-2,
                                                                       disp=0, full_output=True, retall = 1);
    
        return abg_nm
 
def MLEEstimate(N_spikes = 100, N_trains=1 ):
    N_phi_init = 8;
    
    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_N=%d_'%N_spikes

#    old_table_name = 'FinalEstimate_4x100_N=%d'%(N_spikes)
    new_table_file_name = 'MLEEstimate_4x100_N=%d'%(N_spikes)
    dHarvester = DataHarvester(new_table_file_name,                           
                               overwrite=False)
    print 'loading ', new_table_file_name
    
    for regime_name, T_thresh, N_phi in zip(['subT','superT', 'crit', 'superSin'],
                                             [128., 64., 64., 64.],
                                             4*[32]):
        regime_label = base_name + regime_name
            
        for sample_id in xrange(1,N_trains +1):
            file_name = regime_label + '_' + str(sample_id)
            print file_name
            
            #RELOAD ALL DATA:               
            mleBinnedTrain = MLEBinnedSpikeTrain.initFromFile(file_name,
                                                              N_phi)
            ps = mleBinnedTrain ._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            dHarvester.setRegime(regime_name, abg_true, Tsim=-1.0)
            
            abg_init = dHarvester.getEstimates(sample_id, 
                                              regime_name,
                                              'Initializer')
            #MLE F-P:
            dx = .05; dt = .05; 
            phis = mleBinnedTrain.phi_ms;
            S = FPMultiPhiSolver(mleBinnedTrain.theta, phis,
                                 dx, dt,
                                 mleBinnedTrain.getTf(), X_min = -1.0)            
            
            start = time.clock()
            abg_neldermead = MLEEstimator(S, mleBinnedTrain, abg_init)
            finish = time.clock()
            
            print abg_neldermead, finish-start
            dHarvester.addEstimate(sample_id, 'MLE_nm32',
                                   abg_neldermead, finish-start, 0)
                    
    dHarvester.closeFile() 
   
    print 'batch time = ', (time.clock() - batch_start) / 3600.0, ' hrs'
    
     
def MLEEstimateGradients(N_spikes = 100, N_trains=100 ):
    N_phi_init = 8;
    
    batch_start = time.clock()    
    base_name = 'sinusoidal_spike_train_N=%d_'%N_spikes

    old_table_name = 'MLEEstimate_4x100_N=%d'%(N_spikes)
    new_table_file_name = 'MLEEstimateGradOpts_4x100_N=%d'%(N_spikes)
    dHarvester = DataHarvester(old_table_name,
                                new_table_file_name,                           
                                overwrite=False)
    print 'loading ', new_table_file_name
    
    for regime_name, T_thresh, N_phi in zip(['subT','superT', 'crit', 'superSin'],
                                             [128., 64., 64., 64.],
                                             [16,16,16,16]):
        regime_label = base_name + regime_name
            
        for sample_id in xrange(1,N_trains +1):
            file_name = regime_label + '_' + str(sample_id)
            print file_name
            
            #RELOAD ALL DATA:               
            mleBinnedTrain = MLEBinnedSpikeTrain.initFromFile(file_name,
                                                              N_phi)
            ps = mleBinnedTrain ._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
            dHarvester.setRegime(regime_name, abg_true, Tsim=-1.0)
            
            abg_init = dHarvester.getEstimates(sample_id, 
                                              regime_name,
                                              'Initializer')[0]
            #MLE F-P:
            dx = .05; dt = .05; 
            phis = mleBinnedTrain.phi_ms;
            S = FPMultiPhiSolver(mleBinnedTrain.theta, phis,
                                 dx, dt,
                                 mleBinnedTrain.getTf(), X_min = -1.0)            
            
#            start = time.clock()
#            abg_slsqp = MLEEstimator(S, mleBinnedTrain, abg_init,optim_mthd='slsqp')
#            finish = time.clock()
#            print abg_slsqp, finish-start
#            dHarvester.addEstimate(sample_id, 'MLE_slsqp',
#                                   abg_slsqp, finish-start, 0)

            start = time.clock()
            abg_bfgs = MLEEstimator(S, mleBinnedTrain,
                                     abg_init,optim_mthd='bfgs')
            finish = time.clock()
            print abg_bfgs, finish-start
            dHarvester.addEstimate(sample_id, 'MLE_bfgs',
                                   abg_bfgs, finish-start, 0)
            
            
            start = time.clock()
            abg_tnc = MLEEstimator(S, mleBinnedTrain, abg_init,optim_mthd='tnc')
            finish = time.clock()
            print abg_tnc, finish-start
            dHarvester.addEstimate(sample_id, 'MLE_tnc',
                                   abg_tnc, finish-start, 0)
                    
    dHarvester.closeFile() 
   
    print 'batch time = ', (time.clock() - batch_start) / 3600.0, ' hrs'

def MLEBox(N_spikes = 1000, N_trains=5, N_phi=16):
    print 'N_phi = ', N_phi
    
    N_phi_init = 8;
    phi_norms_init =  linspace(1/(2.*N_phi_init), 1. - 1/ (2.*N_phi_init), N_phi_init)

#    base_name = 'sinusoidal_spike_train_N=%d_'%N_spikes
#    output_file = open('mlebox_output.txt', 'w')
#    for regime_name, T_thresh in zip(['subT', 'crit', 'superSin','superT'],
#                                     [128., 64., 64., 64.]):
    output_file = open('mlebox_output_thetas.txt', 'w')
    thetas = [20]
    base_name = 'sinusoidal_spike_train_N=%d_critical_theta='%N_spikes
    T_thresh = 64.
    for theta in thetas:    
        for sample_id in xrange(1,N_trains +1):
            regime_name = 'theta%d'%theta
            regime_label = base_name + '%d'%theta            
            file_name = regime_label + '_%d'%sample_id
            print file_name
                        
#        regime_label = base_name + regime_name
#            file_name = regime_label + '_' + str(sample_id)
#            print file_name
            
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms_init)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
        
            binnedTrain.pruneBins(None, N_thresh = 8, T_thresh=T_thresh)
            abg_init = initialize_right_2std(binnedTrain)
            abg_init[1] = amax([.1, abg_init[1]])
            abg_init[2] = amax([.0, abg_init[2]])

            abg_fortet, warnflag = FortetEstimatorSup(binnedTrain, abg_init)
            
            #RELOAD ALL DATA:               
            mleBinnedTrain = MLEBinnedSpikeTrain.initFromFile(file_name,
                                                              N_phi)
            
            #MLE F-P:
            dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg_true, -1.0) 
            phis = mleBinnedTrain.phi_ms;
            S = FPMultiPhiSolver(binnedTrain.theta, phis,
                                 dx, dt,
                                 binnedTrain.getTf(), X_min = -1.0)
            
            minus_idxs = mleBinnedTrain.phi_minus_indxs
            plus_idxs  = mleBinnedTrain.phi_plus_indxs
            minus_weights = mleBinnedTrain.phi_minus_weights
            plus_weights  = mleBinnedTrain.phi_plus_weights
            
            def loglikelihood(abg):     
                'rediscretize:'
                xmin = FPMultiPhiSolver.calculate_xmin(S.getTf(), abg, S._theta)
                dt = FPMultiPhiSolver.calculate_dt(S._dx, abg, xmin)
                S.rediscretize(S._dx, dt, S.getTf(), xmin)
                'Solve it:'
                Fs = S.c_solve(abg)
                spike_t_indexes = mleBinnedTrain.getTindxs(S._ts)
                'form (approximate) likelihood:'
                pdf = -diff(Fs[:,:,-1], axis = 1) / S._dt;
                likelihoods = pdf[minus_idxs, spike_t_indexes]*minus_weights +\
                              pdf[plus_idxs, spike_t_indexes]*plus_weights
#                if amin(likelihoods) <= .0:
#                    likelihoods[likelihoods<=.0] = 1e-8
                normalized_log_likelihood = sum(log(likelihoods))
                'Return '
                return -normalized_log_likelihood
            #MLE F-P:
#            abg_tnc, abg_cobyla, abg_neldermead = MLEEstimator(S,
#                                                             mleBinnedTrain, abg_init)
            abg_neldermead = MLEEstimator(S,mleBinnedTrain, abg_init)
            #OUTPUTs
            output_file.write('\n' +file_name + ':\n')
            for tag, abg in zip(['init', 'fortet', 'nelder_mead', 'true'],
                                [abg_init, abg_fortet, abg_tnc,
                                 abg_cobyla, abg_neldermead , abg_true]):
                output_file.write(tag + ':' + str(loglikelihood(abg)) + ':' + str(abg) + '\n');
#                print tag, ':', loglikelihood(abg)
                

def thetas_explorer():
    from numpy import load
    Ls = load('likelihoods_thetas.npy')
    errors = load('errors_thetas.npy')
    
    print 'log likelihood absolute error, |True - approximate|:'
    print '\t\t basic bin | weighted bin | gradphi'
    thetas = [10, 20]
    for regime_idx, theta in enumerate(thetas):    
        log_L_star      = sum(log(Ls[regime_idx,0,:]))
        log_L_m         = sum(log(Ls[regime_idx,1,:]))
        log_L_weighted  = sum(log(Ls[regime_idx,2,:]))
        log_L_gradphi   = sum(log(Ls[regime_idx,3,:]))
#        print tag, ': %.3f, %.3f, %.3f' %(log_L_star, log_L_m, log_L_weighted)
        print theta, ': %.3f, %.3f , %.3f' %(abs(log_L_star-log_L_m),
                                           abs(log_L_star-log_L_weighted),
                                           abs(log_L_star-log_L_gradphi))
    print 'log likelihood relative error, |True - approximate|/ True:'
    print '\t\t basic bin | weighted bin | gradphi'
    for regime_idx, theta in enumerate(thetas): 
        log_L_star      = sum(log(Ls[regime_idx,0,:]))
        log_L_m         = sum(log(Ls[regime_idx,1,:]))
        log_L_weighted  = sum(log(Ls[regime_idx,2,:]))
        log_L_gradphi   = sum(log(Ls[regime_idx,3,:]))
        print theta, ': %.3f, %.3f , %.3f' %(abs(log_L_star-log_L_m) / abs(log_L_star),
                                           abs(log_L_star-log_L_weighted)/ abs(log_L_star),
                                           abs(log_L_star-log_L_gradphi) / abs(log_L_star))
    
#    print 10*'-'
    

def explorer(regime_tags = ['superSin', 'crit', 'superT', 'subT'],
             L_file = 'likelihoods.npy',
             err_file = 'errors.npy'):
    from numpy import load
    Ls = load(L_file)
    errors = load(err_file)
    
    print 'log likelihood absolute error, |True - approximate|:'
    print '\t\t basic bin | weighted bin | gradphi'
    for regime_idx, tag in enumerate(regime_tags):
        log_L_star      = -sum(log(Ls[regime_idx,0,:]))
        log_L_m         = -sum(log(Ls[regime_idx,1,:]))
        log_L_weighted  = -sum(log(Ls[regime_idx,2,:]))
        log_L_gradphi   = -sum(log(Ls[regime_idx,3,:]))
#        print tag, ': %.3f, %.3f, %.3f' %(log_L_star, log_L_m, log_L_weighted)
        print tag, ': %.3f, %.3f , %.3f' %(abs(log_L_star-log_L_m),
                                           abs(log_L_star-log_L_weighted),
                                           abs(log_L_star-log_L_gradphi))
    print 'log likelihood relative error, |True - approximate|/ True:'
    print '\t\t basic bin | weighted bin | gradphi'
    for regime_idx, tag in enumerate(regime_tags):
        log_L_star      = -sum(log(Ls[regime_idx,0,:]))
        log_L_m         = -sum(log(Ls[regime_idx,1,:]))
        log_L_weighted  = -sum(log(Ls[regime_idx,2,:]))
        log_L_gradphi   = -sum(log(Ls[regime_idx,3,:]))
#        print tag, ': %.3f, %.3f, %.3f' %(log_L_star, log_L_m, log_L_weighted)
        print tag, ': %.3f, %.3f , %.3f' %(abs(log_L_star-log_L_m) / abs(log_L_star),
                                           abs(log_L_star-log_L_weighted)/ abs(log_L_star),
                                           abs(log_L_star-log_L_gradphi) / abs(log_L_star))
    
    

#    print 10*'-'
#    #########
#    diFs = load('diFs_adapted.npy')
#    errors = load('errors_adapted.npy')    
#    for regime_idx, tag in enumerate(['superSin', 'crit', 'superT', 'subT']):
#        log_L_star = sum(log(diFs[regime_idx,0,:]))
#        log_L_m = sum(log(diFs[regime_idx,1,:]))
#        log_L_weighted = sum(log(diFs[regime_idx,2,:]))
##        print tag, ': %.3f, %.3f, %.3f' %(log_L_star, log_L_m, log_L_weighted)
#        print tag, ': %.3f, %.3f ' %(abs(log_L_star-log_L_m), abs(log_L_star- log_L_weighted))
    
#    print 10*'-'
#    #########
#    diFs = load('diFs_gradphi.npy')
#    errors = load('errors_gradphi.npy')    
#    for regime_idx, tag in enumerate(['superSin', 'crit', 'superT', 'subT']):
#        log_L_star = sum(log(diFs[regime_idx,0,:]))
#        log_L_m = sum(log(diFs[regime_idx,1,:]))
#        log_L_weighted = sum(log(diFs[regime_idx,2,:]))
##        print tag, ': %.3f, %.3f, %.3f' %(log_L_star, log_L_m, log_L_weighted)
#        print tag, ': %.3f, %.3f ' %(abs(log_L_star-log_L_m), abs(log_L_star- log_L_weighted))

  
def adapted_xplorer():
    from numpy import load
    Ls = load('Ls_adapted.npy')
    errors = load('errors.npy')
    
    print 'log likelihood absolute error, |True - approximate|:'
    print '\t\t basic bin | weighted bin '
    for regime_idx, tag in enumerate(['superSin', 'crit', 'superT', 'subT']):
        log_L_star      = sum(log(Ls[regime_idx,0,:]))
        log_L_m         = sum(log(Ls[regime_idx,1,:]))
        log_L_weighted  = sum(log(Ls[regime_idx,2,:]))
        print tag, ': %.3f, %.3f' %(abs(log_L_star-log_L_m),
                                           abs(log_L_star-log_L_weighted))
    print 'log likelihood relative error, |True - approximate|/ True:'
    print '\t basic bin | weighted bin '
    for regime_idx, tag in enumerate(['superSin', 'crit', 'superT', 'subT']):
        log_L_star      = sum(log(Ls[regime_idx,0,:]))
        log_L_m         = sum(log(Ls[regime_idx,1,:]))
        log_L_weighted  = sum(log(Ls[regime_idx,2,:]))
        print tag, ': %.3f, %.3f ' %(abs(log_L_star-log_L_m) / abs(log_L_star),
                                           abs(log_L_star-log_L_weighted)/ abs(log_L_star))
    

def MLEThetas(N_spikes = 1000, N_trains=8, N_phi=16):
    print 'N_phi = ', N_phi
    N_phi_init = 16;
    phi_norms_init =  linspace(1/(2.*N_phi_init), 1. - 1/ (2.*N_phi_init), N_phi_init)

    output_file = open('mlebox_output_thetas.txt', 'w')
    thetas = [20]
    base_name = 'sinusoidal_spike_train_N=%d_critical_theta='%N_spikes
    T_thresh = 64.
    for theta in thetas:    
        for sample_id in xrange(1,N_trains +1):
            regime_label = base_name + '%d'%theta            
            file_name = regime_label + '_%d'%sample_id
            print file_name
            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms_init)
            ps = binnedTrain._Train._params
            abg_true = array((ps._alpha, ps._beta, ps._gamma))
        
            binnedTrain.pruneBins(None, N_thresh = 8, T_thresh=T_thresh)
            abg_init = initialize_right_2std(binnedTrain)
            abg_init[1] = amax([.1, abg_init[1]])
            abg_init[2] = amax([.0, abg_init[2]])

            abg_fortet, warnflag = FortetEstimatorSup(binnedTrain, abg_init)
            
            #RELOAD ALL DATA:               
            mleBinnedTrain = MLEBinnedSpikeTrain.initFromFile(file_name,
                                                              N_phi)
            
            #MLE F-P:
            dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg_true, -1.0) 
            phis = mleBinnedTrain.phi_ms;
            S = FPMultiPhiSolver(binnedTrain.theta, phis,
                                 dx, dt,
                                 binnedTrain.getTf(), X_min = -1.0)
            
            abg_mle = MLEEstimator(S, mleBinnedTrain, abg_init)
            abg_fortet_mle = MLEEstimator(S, mleBinnedTrain, abg_fortet)
                        
            minus_idxs = mleBinnedTrain.phi_minus_indxs
            plus_idxs  = mleBinnedTrain.phi_plus_indxs
            minus_weights = mleBinnedTrain.phi_minus_weights
            plus_weights  = mleBinnedTrain.phi_plus_weights
            def loglikelihood(abg):     
                'rediscretize:'
                xmin = FPMultiPhiSolver.calculate_xmin(S.getTf(), abg, S._theta)
                dt = FPMultiPhiSolver.calculate_dt(S._dx, abg, xmin)
                S.rediscretize(S._dx, dt, S.getTf(), xmin)
                'Solve it:'
                Fs = S.c_solve(abg)
                spike_t_indexes = mleBinnedTrain.getTindxs(S._ts)
                'form (approximate) likelihood:'
                pdf = -diff(Fs[:,:,-1], axis = 1) / S._dt;
                likelihoods = pdf[minus_idxs, spike_t_indexes]*minus_weights +\
                              pdf[plus_idxs, spike_t_indexes]*plus_weights
#                if amin(likelihoods) <= .0:
#                    likelihoods[likelihoods<=.0] = 1e-8
                normalized_log_likelihood = sum(log(likelihoods))
                'Return '
                if (NaN == normalized_log_likelihood or nan == normalized_log_likelihood):
                    print '!!! nan ', abg
                
                return -normalized_log_likelihood
            #OUTPUTs
            output_file.write('\n' +file_name + ':\n')
            for tag, abg in zip(['init', 'fortet', 'mle', 'fortet+mle', 'true'],
                                [abg_init, abg_fortet, 
                                 abg_mle, abg_fortet_mle, abg_true]):
                output_file.write(tag + ':' + str(loglikelihood(abg)) + ':' + str(abg) + '\n');
#                print tag, ':', loglikelihood(abg)
                
    

def visualizeBinning():
    for  tag in        ['superT', 'crit', 'subT','superSin']:
        mleBT = MLEBinnedSpikeTrain.initFromFile('sinusoidal_spike_train_N=8_%s_1'%tag, N_phi=3)
        mleBT.visualize()
            
if __name__ == '__main__':
    from pylab import *
 
#    sandbox()
#    explorer()
#    thetas_sandbox()
#    thetas_explorer()
#    adapted_sandbox()
#    gradphiF_sandbox()
    
#    supersin_sandbox()
#    explorer(regime_tags=['superSin'],
#             L_file = 'likelihoods_supersin.npy',
#             err_file = 'errors_supersin.npy')

#    adapted_xplorer()

#    MLEEstimateGradients(N_spikes=100, N_trains=100)
    MLEEstimate(N_trains=100, N_spikes=100)
#    MLEEstimate(N_trains=100, N_spikes=1000)
    
#    MLEBox()
#    MLEThetas(N_phi = 64)

#    mleBT = MLEBinnedSpikeTrain.initFromFile('sinusoidal_spike_train_N=4_superT_1', N_phi=4)
#    visualizeBinning()
    
    show()