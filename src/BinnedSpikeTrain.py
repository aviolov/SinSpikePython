# -*- coding:utf-8 -*-
"""
Created on Apr 30, 2012

@author: alex
"""
from __future__ import division

from SpikeTrainSimulator import SpikeTrain, RESULTS_DIR as PATH_DIR, OUSinusoidalParams

from numpy import unique, zeros, ones, ones_like, array, r_, float64, ceil, arange, pi, zeros_like, diff, sort, mean, median, mod, where, setdiff1d, min, max, nonzero
from numpy.random import randint

import os
from Simulator import ABCD_LABEL_SIZE
FIGS_DIR = '/home/alex/Workspaces/Latex/LIFEstimation/Figs/Bins'
import os
for D in [FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

def loadPath(file_name):
        file_name = os.path.join(PATH_DIR, file_name) 
        return SpikeTrain.load(file_name)


def generateSDF(Is):
    N = len(Is)
    
    unique_Is = unique(Is)
    ts = r_[(.0, unique_Is)]

    unique_Is = ts;
        
    SDF = ones_like(ts)
    
    for (Ik, idx) in zip(unique_Is, arange(1, len(SDF))):
        SDF[idx] = sum(Is> Ik) / N;
    return SDF, unique_Is


def intervalStats(file_name):
    T =  loadPath(file_name)

    print 'Path params = (a, b,g, \\th) = (%g, %g, %g, %g)' %(T._params._alpha,
                                                    T._params._beta,
                                                    T._params._gamma,
                                                    T._params._theta) 
    
    Is = r_[(T._spike_ts[0], diff(array(T._spike_ts)))];
    
    N = len(Is)
    assert((Is>0).all())
    assert((diff(sort(Is))>=0.).all())
    
    print 'spike count = ', N
    
    print 'least interval = ', min(Is)
    print 'max interval = ', max(Is)
    
    print 'max transtion = ', max(diff(sort(Is)))
    print 'avg transtion = ', mean(diff(sort(Is)))
    print 'median transtion = ', median(diff(sort(Is)))


class BinnedSpikeTrain():
    def __init__(self, T, phi_norms):
        ''' bins the spike intervals (diff(spike_ts)) into bins corresponding to the phase angle at the beginning of each interval'''
        self._Train = T
        self._spike_ts = T._spike_ts;
        self.theta = T._params._theta;
        self._spike_Is = r_[(self._spike_ts[0], diff(array(self._spike_ts)))];
        self._spike_Phis = r_[(.0, mod(self._spike_ts[:-1], 2.*pi / self.theta))]
        
        self.phi_ms = phi_norms * 2. * pi / self.theta 
        self._dphi = self.phi_ms[1] - self.phi_ms[0]
        
        self.bins = self._ripBins()
        '''Be very! careful - the bins.keys != phi_ms: phi_ms are all the segmented bins,
         however some might have been discarded either manually or b/c there were not eneough data pts in said bin'''
            
    @classmethod
    def initFromFile(cls, file_name, phi_norms):
        P = loadPath(file_name)
        return BinnedSpikeTrain(P, phi_norms)
    
    def getTf(self):
        Tf = .0;
        for phi in self.bins.keys():
            uniqueIs = self.bins[phi]['unique_Is']
            
            Tf = max([max(uniqueIs), Tf])
    
        return Tf;

    def getSpikeCount(self):
        return len(self._Train._spike_ts)
    def getBinCount(self):
        return len(self.bins.keys())
    
    def _ripBins(self):
        Phis = self._spike_Phis 
                
        dphi = self._dphi 
    
        bins = {}    
        for phi in self.phi_ms:
            local_bin = {}

            choose_array = where((Phis >= (phi - dphi/2.))*
                                 (Phis < (phi + dphi/2.)))
    
            lIs = self._spike_Is[choose_array]
            if len(lIs) == 0:
                continue
                
            local_bin['Is']   = lIs
            local_bin['Phis'] = Phis[choose_array]
            
            SDF, uniqueIs = generateSDF(lIs)
            
            local_bin['SDF'] = SDF
            local_bin['unique_Is'] = uniqueIs
            
    
            bins[phi] =  local_bin
            
        return bins

    def pruneBins(self, phi_discard=None, N_thresh = 0, T_thresh = None):
        '''remove some phis from the bins and return the pruned bins'''
        '''Also remove Is from bins st. Is  > T_thresh'''
        bins = self.bins
        phi_ms = bins.keys()  
        for phi_m in phi_ms:
            if (None != phi_discard and min(abs(phi_discard - phi_m))<1e-4):
                #Discard specific phi's:
                bins.pop(phi_m)
                continue
            
            lIs = bins[phi_m]['Is']
            N_m = len(lIs);
            if (N_m < N_thresh):
                #Discard phi's with too little samples
                bins.pop(phi_m)
                continue
            
            if None != T_thresh:
                unique_Is = bins[phi_m]['unique_Is']
                SDF = bins[phi_m]['SDF']
                idxs = nonzero(unique_Is <= T_thresh)

                if 0 == len(idxs[0]):
                    raise Exception('The applied T_thresh reduces this bin to no live pts...')
                
                unique_Is = unique_Is[idxs];
                SDF = SDF[idxs];
                bins[phi_m]['unique_Is'] = unique_Is
                bins[phi_m]['SDF'] = SDF
                
        return bins

    
    def visualize(self, title_tag = '', save_fig_name = '', phis = None):
        '''Pylab visualize the binning procedure '''
        bins = self.bins; 
        if None == phis:
            phis = bins.keys()
        
        dphi = self._dphi
        P = self._Train
        #Visualize time:
        mpl.rcParams['figure.subplot.left'] = .1
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .225
        mpl.rcParams['figure.subplot.top'] = .95
        mpl.rcParams['figure.subplot.wspace'] = .075
        mpl.rcParams['figure.figsize'] = 17, 6
        label_font_size = 24
        xlabel_font_size = 40
        
        figure()
        ax = subplot(121)
        def relabel_major(x, pos):
            if x < 0:
                    return ''
            else:
                    return '$%.1f$' %x
        for phi in phis:
            if phi in bins.keys():
                local_bin = bins[phi]
                
                Phis = local_bin['Phis']
                Is   = local_bin['Is']
                
                plot(Phis, Is, '.', markersize=6)
                
#            axvline(x=phi,color='b', linestyle =  '--')
            axvline(x = phi+dphi/2.0, color='r')

        xlim((.0, 2*pi))
        xlabel(r'$\phi_n$', fontsize = xlabel_font_size)
        ylabel(r'$i_n$', fontsize = xlabel_font_size)
#        title_tag = 'Raw $(i_n, \phi_n)$ for $(\\alpha,\\beta,\\gamma) = (%g, %g, %.2g)$' %(P._params._alpha,P._params._beta,P._params._gamma)
#        title(title_tag, fontsize = 20)
        tick_locs = phis
        tick_lbls = [r'$\frac{%d \pi}{%d}$'%(int(round(len(phis)*x/pi)),len(phis)) for x in phis]
#        tick_lbls = [r'$ %d \pi / %d$'%(int(round(len(phis)*x/pi)),len(phis)) for x in phis]
        xticks(tick_locs, tick_lbls)
        tick_params(labelsize = label_font_size)
        ax.yaxis.set_major_formatter(FuncFormatter(relabel_major))
#        annotate(r'A',
#                 (13* pi / 8.0 , self.getTf()), fontsize = 30)
        def add_inner_title(ax, title, loc, size=None, **kwargs):
            from matplotlib.offsetbox import AnchoredText
            from matplotlib.patheffects import withStroke
            if size is None:
                size = dict(size=plt.rcParams['legend.fontsize'])
            at = AnchoredText(title, loc=loc, prop=size,
                              pad=0., borderpad=0.5,
                              frameon=False, **kwargs)
            ax.add_artist(at)
            at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
            return at
        t = add_inner_title(ax, 'A', loc=2, size=dict(size=ABCD_LABEL_SIZE))
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)
        
        #SHOW BINS:
        ax = subplot(122)
        for phi in phis:
            if phi in bins.keys():
                local_bin = bins[phi]
                #        get_current_fig_manager().window.showMaximized() 
                Phis = local_bin['Phis']
                Is   = local_bin['Is']
            
                plot(phi*ones_like(Is), Is, '.', markersize=6)
            
#            axvline(x=phi,color='b', linestyle =  '--')
            axvline(x = phi+dphi/2.0, color='r')
       
        xlim((.0, 2*pi))
        xlabel(r'$\phi_m$', fontsize = xlabel_font_size)
        setp(ax.get_yticklabels(), visible=False)

#        ax.yaxis.set_major_formatter(FuncFormatter(relabel_major))
#        ylabel(r'$I_n$', fontsize = 28)
#        title_tag = 'Binned $(i_n, \phi_n)$ for $(\\alpha,\\beta,\\gamma) = (%g, %g, %.2g)$' %(P._params._alpha,P._params._beta,P._params._gamma)
#        title(title_tag, fontsize = 20)
                    
        xticks(tick_locs, tick_lbls)
        tick_params(labelsize = label_font_size)
#        annotate(r'B',
#                 (13* pi / 8.0 , self.getTf()), fontsize = 30)
        t = add_inner_title(ax, 'B', loc=2, size=dict(size=ABCD_LABEL_SIZE))
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)

#        get_current_fig_manager().window.showMaximized() 
        if '' != save_fig_name:
            file_name = os.path.join(FIGS_DIR, save_fig_name+'_composite.pdf')
            print 'saving to ', file_name
            savefig(file_name, dpi=(300))
        
    def getPhiInterval(self, index):
        return self._spike_Phis[index],  self._spike_Is[index] 

    def getRandomPhiInterval(self):
        interval_index = randint(self.getSpikeCount())
        return self.getPhiInterval(interval_index)
    
def repruneTest():
    N_phi = 20;
    print 'N_phi = ', N_phi
    
    phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
#    file_name = 'sinusoidal_spike_train_N=1000_subT_8'
    for regime_tag in ['superT', 'subT', 'superSin', 'crit']:
        
        for idx in [2,5,9,14]:       
            file_name = 'sinusoidal_spike_train_N=1000_' + regime_tag +'_' + str(idx)
            T = BinnedSpikeTrain.initFromFile(file_name, phis)
            print regime_tag+'_' + str(idx) 
            N_thresh = 0
            T.pruneBins(phi_discard=None, N_thresh = N_thresh, T_thresh=32.);
            bins = T.bins;
            keys = bins.keys()
            for phi in sort(keys):
                spike_count = len(bins[phi]['Is'])
                if spike_count < 2:
                    print '\t' , phi, '\t', spike_count

            
    

def visualizeBinning():
    N_phi = 8;
    print 'N_phi = ', N_phi
    phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
##    file_name = 'sinusoidal_spike_train_N=1000_subT_8'
#    file_name = 'sinusoidal_spike_train_N=1000_crit_16'
    file_name = 'sinusoidal_spike_train_N=1000_superT_8'
    binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
    print binnedTrain._Train._params.getParams()

#    intervalStats(file_name)
    
#    phi_omit = None #r_[(linspace(.25, .45, 3),linspace(.75,.95,3) )]  *2*pi/ binnedTrain.theta
#    binnedTrain.pruneBins(phi_omit, N_thresh=32)
#    print 'Tf = ', binnedTrain.getTf()
    binnedTrain.visualize(title_tag='',
                           save_fig_name = 'Example')


if __name__ == '__main__':
    from pylab import *

    visualizeBinning()
#    repruneTest()
    
    show()
    