'''
Created on May 4, 2012

@author: alex
'''

from BinnedSpikeTrain import BinnedSpikeTrain, intervalStats
from SpikeTrainSimulator import SpikeTrain, OUSinusoidalParams

from FPMultiPhiSolver import FIGS_DIR, FPMultiPhiSolver, RESULTS_DIR

from InitBox import initialize_right_2std

import os

from numpy import *
from Simulator import ABCD_LABEL_SIZE

def calculateExactSDFs():
    N_phi = 4;
    print 'N_phi = ', N_phi
    
    phis =  2*pi*array([0,.25, .5, .75])
       
    
    for regime_idx, regime_name in enumerate(['superT', 'superSin', 'crit','subT']):
    #        for regime_name in ['superT']:
        regime_label = 'sinusoidal_spike_train_N=1000_' + regime_name + '_12' 
        binnedTrain = BinnedSpikeTrain.initFromFile(regime_label, phis)
        Tf = binnedTrain.getTf()
        print 'Tf = ', Tf
        
        theta = binnedTrain.theta;
        print 'theta = ', theta
        ps = binnedTrain._Train._params
        abg_true = array((ps._alpha, ps._beta, ps._gamma))
        abg = abg_true
        xmin = FPMultiPhiSolver.calculate_xmin(Tf, abg, theta)
#        dx = .0125#        dx = .0125; ;
        dx = .0125; 
        dt = FPMultiPhiSolver.calculate_dt(dx, abg, xmin, factor = 5.)
        print 'xmin = ', xmin, ', dx, dt = ', dx, dt
        
        S = FPMultiPhiSolver(theta, phis,
                         dx, dt, Tf, xmin)
   
        S.setTf(Tf)
        Fs = S.c_solve(abg)
        ts = S._ts;
         
        
        filename= RESULTS_DIR + '/Fs_%s'%regime_name
        print 'saving Fs to ', filename
        savez(filename, ts=ts,
                         Gs=squeeze(Fs[:,:,-1]),
                         phis=phis,
                         Tf = Tf);
   
def calcG_U_Const(abg_init, ts, phis, Omega=1.):
    a = abg_init[0];
    b = abg_init[1];
    g = abg_init[2];
    
    from scipy.stats import norm
    
    Gs = ones( (len(phis), len(ts)) )
    
    for phi_k in xrange(len(phis)):
        phi = phis[phi_k]
        for tk in xrange(1,len(ts)):
            t = ts[tk];
            U_tp = t*(a - .5) - g * ( cos(Omega * ( t + phi ) ) - cos(Omega * phi) ) / Omega   
            sigma_tp = b*sqrt(t)
            Gs[phi_k,tk] = norm.cdf(1., loc = U_tp, scale = sigma_tp)
    return Gs 
            
    
                    
def calculateInitSDFs():
    N_phi = 16;
    print 'N_phi = ', N_phi
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
    
    for regime_idx, regime_name in enumerate(['superT', 'superSin', 'crit','subT']):
    #        for regime_name in ['superT']:
        filename = RESULTS_DIR + '/Fs_%s.npz'%regime_name 
        npzfile = load(filename)
        ts = npzfile['ts']
        phis = npzfile['phis']
        
        regime_label = 'sinusoidal_spike_train_N=1000_' + regime_name + '_12' 
        binnedTrain = BinnedSpikeTrain.initFromFile(regime_label, phi_norms)
        binnedTrain.pruneBins(None, N_thresh = 16, T_thresh=128.)
           
        ps = binnedTrain._Train._params
        abg = initialize_right_2std(binnedTrain)
#        abg = array((ps._alpha, ps._beta, ps._gamma))
        
        GsInit = calcG_U_Const(abg,ts,phis);
        
        filename= RESULTS_DIR + '/Gs_%s'%regime_name
        print 'saving Gs to ', filename
        savez(filename, GsInit=GsInit);
        
        
        
        
 
def visualizeInitVsExactSDF( save_figs = True ):    
    N_phi = 4;
    print 'N_phi = ', N_phi
    
#    phis =  2*pi*array([0,.25, .5, .75])
    regime_tags = {'superT':'Supra-Threshold', 'crit':'Critical',
                   'subT':'Sub-Threshold',  'superSin':'Super-Sinusoidal'}
    
    mpl.rcParams['figure.figsize'] = 17, 5*4
    mpl.rcParams['figure.subplot.left'] = .15
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.top'] = .975
    mpl.rcParams['figure.subplot.bottom'] = .05
    mpl.rcParams['figure.subplot.hspace'] = .375
    mpl.rcParams['figure.subplot.wspace'] = .3
    
    sdf_fig = figure()
    inner_titles = {0: 'A',
                    1:'B',
                    2:'C',
                    3:'D'}
    xlabel_font_size = 40
    
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
    def relabel_major(x, pos):
            if x < 0:
                    return ''
            else:
                    return '$%.1f$' %x
    
    for regime_idx, regime_name in enumerate(['superT', 'superSin', 'crit','subT']):
    #        for regime_name in ['superT']:
#        regime_label = 'sinusoidal_spike_train_N=1000_' + regime_name + '_12' 
#        binnedTrain = BinnedSpikeTrain.initFromFile(regime_label, phis)
#        Tf = binnedTrain.getTf()
#        print 'Tf = ', Tf
#        
#        theta = binnedTrain.theta;
#        print 'theta = ', theta
#        ps = binnedTrain._Train._params
#        abg_true = array((ps._alpha, ps._beta, ps._gamma))
#        abg = abg_true
#        xmin = FPMultiPhiSolver.calculate_xmin(Tf, abg, theta)
#        dx = .0125; 
#        dt = FPMultiPhiSolver.calculate_dt(dx, abg, xmin, factor = 5.)
#        print 'xmin = ', xmin, ', dx, dt = ', dx, dt
#        
#        S = FPMultiPhiSolver(theta, phis,
#                         dx, dt, Tf, xmin)
#   
#        S.setTf(Tf)
#        
#        Fs = S.c_solve(abg)
#        ts = S._ts;

        from  numpy import load
        filename  = RESULTS_DIR + '/Fs_%s.npz'%regime_name 
        npzfile = load(filename)
        ts = npzfile['ts']
        Gs = npzfile['Gs']
        phis = npzfile['phis']
        Tf = npzfile['Tf']
        
        filename= RESULTS_DIR + '/Gs_%s.npz'%regime_name
        npzfile = load(filename)
        GsInit = npzfile['GsInit']
        print GsInit.shape
        
        #SDF Fig:
        figure(sdf_fig.number)
        label_font_size = 24
        for (phi_m, err_idx) in zip( sort(phis),
                                     xrange(len(phis)) ):
            phi_idx = nonzero( abs(phis - phi_m) < 1e-4)
            lF = squeeze(Gs[phi_idx, :])
            ax = sdf_fig.add_subplot(2*len(phis),2,
                                      8*(regime_idx>1) +mod(regime_idx,2)  + 2*err_idx + 1)
            Ginit = squeeze(GsInit[phi_idx,:]);
            plot(ts, lF, 'b',
                 linewidth=3, label='Analytic');
            plot(ts, Ginit, 'r',
                 linewidth=3, label='Init. Aproxn');
            annotate(r'$ \phi_{m} =  %.2g \cdot 2\pi $' %(phi_m / 2 / pi),
                      (Tf/(1.75), .33), fontsize = label_font_size )
            ylabel(r'$\bar{G}(t)$', fontsize = xlabel_font_size)
            xlim((.0, Tf))
            legend()
    
            if len(phis) != err_idx+1:
                setp(ax.get_xticklabels(), visible=False)
            else:
                if regime_idx > 1:
                    xlabel('$t$', fontsize = xlabel_font_size)
                locs, labels = xticks()
                locs = locs[1:]
                xticks(locs)
                ax.xaxis.set_major_formatter(FuncFormatter(relabel_major))
                for label in ax.xaxis.get_majorticklabels():
                    label.set_fontsize(label_font_size)
            yticks( [0., .5,  1.0] )
            ax.xaxis.set_major_locator(MaxNLocator(6)) 
            ax.yaxis.set_major_formatter(FuncFormatter(relabel_major))
            for label in ax.yaxis.get_majorticklabels():
                label.set_fontsize(label_font_size)   
            t = add_inner_title(ax, inner_titles[regime_idx], loc=3,
                                 size=dict(size=ABCD_LABEL_SIZE))
            t.patch.set_ec("none")
            t.patch.set_alpha(0.5)
    
            ylim((.0, 1.05))
    
    if save_figs:
        file_name = os.path.join(FIGS_DIR, 'sdf_init_vs_exact.pdf')
        print 'saving to ', file_name
        sdf_fig.savefig(file_name, dpi=(300))
        

 
if __name__ == '__main__':
    from pylab import *
    from numpy import *
    
#    calculateExactSDFs()
#    calculateInitSDFs()
    
    visualizeInitVsExactSDF(True)
    

    show()