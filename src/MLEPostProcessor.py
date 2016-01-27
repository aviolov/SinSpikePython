'''
Created on May 10, 2012

@author: alex
'''

from tables import *
from DataHarvester import TABLES_DIR, DataAnalyzer
from matplotlib.gridspec import GridSpec
from numpy import array, tile
from FPMultiPhiSolver import FPMultiPhiSolver
from Simulator import ABCD_LABEL_SIZE

FIGS_DIR = '/home/alex/Workspaces/Latex/LIFEstimation/Figs/MLE'
import os
for D in [FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time

from numpy import *

regime_labels = {'subT': 'Sub-threshold',
                 'superT': 'Supra-threshold',
                 'superSin': 'Supra-sinusoidal',
                 'crit': 'Critical',
                 'theta1': r'$\\Omega=1$',
                 'theta5': r'$\\Omega=5$', 
                 'theta10': r'$\\Omega=10$', 
                 'theta20': r'$\\Omega=20$' }
labels = {'Initializer':'Initializer', 'Nelder-Mead':'Absorb. Bnd.',
           'BFGS':'F-P dtve', 'Fortet':'Fortet',
           'init_N10':'Initializer',
           'FP_N10': 'FP_N10',
           'WFP_N1': 'WFP_N1',
           'FP_L2':"L2", 'FP_Sup':'Sup',
           'FP':'Numerical FP',
           'mle_nm':'Numerical FP-MLE'
           }
plot_colours = {'Fortet':"b", 'Initializer':"g", 'BFGS':"r",
                  'Nelder-Mead':"y",
                  'init_N10':"g" ,
                  'FP_N10':"b"  ,
                  'WFP_N1': "r",
                  'FP_L2':'r', 'FP_Sup':"b" ,
                  'FP':'y'}
 
markers = {'Fortet':"d", 'Initializer':"*", 'BFGS':'h',
            'Nelder-Mead':"h",
            'init_N10':"*",
            'FP_N10':"d"  ,
            'WFP_N1':'h',
            'FP_L2':"d", 'FP_Sup':"h",
              'FP':'h'  }

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

def crossAnalyze(table_file_name,
                 fig_id = '', Method1 =  'Fortet', Method2 = 'FP',
                 regimeNames = None):
    analyzer = DataAnalyzer(table_file_name) 
        
    mpl.rcParams['figure.figsize'] = 17, 5*4
    mpl.rcParams['figure.subplot.left'] = .15
    mpl.rcParams['figure.subplot.right'] = .975
    mpl.rcParams['figure.subplot.bottom'] = .05
    mpl.rcParams['figure.subplot.top'] = .95
    mpl.rcParams['figure.subplot.wspace'] = .25
    mpl.rcParams['figure.subplot.hspace'] = .375
    
    fig = figure(); 

    if None==regimeNames:
        regimeNames =  analyzer.getAllRegimes()
    
    plot_scale = 1.05;
    regime_tags = {'superT':'Supra-Threshold', 'crit':'Critical',
                        'subT':'Sub-Threshold',  'superSin':'Super-Sinusoidal',
                    'theta1':r'$\Omega=1$',
                    'theta5':r'$\Omega=5$',
                    'theta10':r'$\Omega=10$',
                    'theta20':r'$\Omega=20$'}  
    xlabel_font_size = 32
    
    for regime_idx,regime in enumerate(regimeNames): 
        abg_true = analyzer.getTrueParamValues(regime)
        print regime, abg_true
        
        abgs = {};
        abgs[Method1] = array( analyzer.getAllEstimates(regime, Method1) )
        abgs[Method2] = array( analyzer.getAllEstimates(regime, Method2) )

        for param_idx, pname in zip(arange(3),
                                    [r'\alpha', r'\beta', r'\gamma']):
            axu_lim = amax( r_[ abgs[Method1][:, param_idx], abgs[Method2][:, param_idx]])
            axl_lim = amin( r_[ abgs[Method1][:, param_idx], abgs[Method2][:, param_idx]])
            
            plot_buffer = (plot_scale -1.)*(axu_lim - axl_lim);
            axu_lim += plot_buffer
            axl_lim -= plot_buffer
            ax = subplot(len(regimeNames),3, regime_idx*3 + param_idx+1, aspect='equal')
            plot(abgs[Method1][:, param_idx], abgs[Method2][:, param_idx],
                  linestyle='None',
                  color=  'k',   marker= 'o', markersize =  7  )
            
            if 0 == param_idx:
                text(-.4, 0.5, regime_tags[regime],
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform = ax.transAxes,
                        rotation='vertical',
                        size = xlabel_font_size)
            
#            axu_lim, axl_lim = xlim()
            hlines(abg_true[param_idx], axl_lim, axu_lim,
                   linestyles = 'dashed' )
#            axu_lim, axl_lim = ylim()
            vlines(abg_true[param_idx], axl_lim, axu_lim,
                   linestyles = 'dashed' )
            ax.xaxis.set_major_locator(MaxNLocator(6))    
            ax.yaxis.set_major_locator(MaxNLocator(6))    

            xlim((axl_lim, axu_lim));
            ylim((axl_lim, axu_lim));
            xlabel(Method1,fontsize = xlabel_font_size); 
            ylabel(Method2,fontsize = xlabel_font_size);
            title('$'+ pname + ' = %.2f$'%abg_true[param_idx], fontsize = 32)
            t = add_inner_title(ax, chr(ord('A') + 
                                        regime_idx*3 + param_idx),
                                loc=2, size=dict(size=ABCD_LABEL_SIZE))
            t.patch.set_ec("none")
            t.patch.set_alpha(0.5) 
              
    if '' != fig_id: 
        file_name = os.path.join(FIGS_DIR, fig_id  + '_cross_compare_joint.pdf')
        print 'saving to ', file_name
        savefig(file_name)


    
def crossAnalyzeJoint(table_file_name1,
                      table_file_name2,
                      fig_id = '', Method1 =  'Fortet', Method2 = 'FP',
                      regimeNames = None):
    
    
     
    analyzer = DataAnalyzer(table_file_name1)
    analyzer2 =  DataAnalyzer(table_file_name2)
        
    mpl.rcParams['figure.figsize'] = 17, 5*4
    mpl.rcParams['figure.subplot.left'] = .05
    mpl.rcParams['figure.subplot.right'] = .975
    mpl.rcParams['figure.subplot.bottom'] = .025
    mpl.rcParams['figure.subplot.top'] = .95
    mpl.rcParams['figure.subplot.wspace'] = .1
    mpl.rcParams['figure.subplot.hspace'] = .375
    fig = figure(); 

    if None==regimeNames:
        regimeNames =  analyzer.getAllRegimes()
    
    for regime_idx,regime in enumerate(regimeNames): 
        abg_true = analyzer.getTrueParamValues(regime)
        print regime, abg_true
        
        abgs = {};
        abgs[Method1] = array( analyzer.getAllEstimates(regime, Method1) )
        abgs[Method2] = array( analyzer.getAllEstimates(regime, Method2) )
        
        abgs2 = {}
        abgs2[Method1] = array( analyzer2.getAllEstimates(regime, Method1) )
        abgs2[Method2] = array( analyzer2.getAllEstimates(regime, Method2) )
         
            
        plot_scale = 1.05;
        for param_idx, pname in zip(arange(3),
                               [r'\alpha', r'\beta', r'\gamma']):
            axu_lim = amax( r_[ abgs[Method1][:, param_idx], abgs[Method2][:, param_idx]])
            axl_lim = amin( r_[ abgs[Method1][:, param_idx], abgs[Method2][:, param_idx]])
            
            plot_buffer = (plot_scale -1.)*(axu_lim - axl_lim);
            axu_lim += plot_buffer
            axl_lim -= plot_buffer
            
            ax = subplot(len(regimeNames),3, regime_idx*3 + param_idx+1, aspect='equal')
            plot(abgs[Method1][:, param_idx], abgs[Method2][:, param_idx],
                  linestyle='None',
                  color=  'k',   marker= 'x', markersize =  7,   label='$N=100$' )
            plot(abgs2[Method1][:, param_idx], abgs2[Method2][:, param_idx],
                  linestyle='None',
                  color=  'k',   marker= 'd', markersize =  7, label='$N=1000$'  )
            if param_idx == 0:
                legend(loc='lower left')
#            axu_lim, axl_lim = xlim()
            hlines(abg_true[param_idx], axl_lim, axu_lim,
                   linestyles = 'dashed' )
#            axu_lim, axl_lim = ylim()
            vlines(abg_true[param_idx], axl_lim, axu_lim,
                   linestyles = 'dashed' )
            ax.xaxis.set_major_locator(MaxNLocator(6))    
            ax.yaxis.set_major_locator(MaxNLocator(6))    

            xlim((axl_lim, axu_lim));
            ylim((axl_lim, axu_lim));
            xlabel(Method1,fontsize = 24); ylabel(Method2,fontsize = 24);
            title('$'+ pname + ' = %.2f$'%abg_true[param_idx], fontsize = 32)
            t = add_inner_title(ax, chr(ord('A') + 
                                        regime_idx*3 + param_idx),
                                loc=2, size=dict(size=20))
            t.patch.set_ec("none")
            t.patch.set_alpha(0.5)
            
#    markers = ['x', '+', '^']:
#    for marker_idx, table_file_name in  table_file_names[1:]:
#        analyzer = DataAnalyzer(table_file_name)
              
    if '' != fig_id: 
        file_name = os.path.join(FIGS_DIR, fig_id  + '_cross_compare_joint.pdf')
        print 'saving to ', file_name
        savefig(file_name)
        
def alphaVsGamma(table_file_name, fig_id='',
                  Methods=['Initializer', 'FP', 'Fortet']):
    analyzer = DataAnalyzer(table_file_name)     
   
    mpl.rcParams['figure.figsize'] = 17, 5*4
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .975
    mpl.rcParams['figure.subplot.bottom'] = .025
    mpl.rcParams['figure.subplot.top'] = .95
    mpl.rcParams['figure.subplot.wspace'] = .25
    mpl.rcParams['figure.subplot.hspace'] = .375
    fig = figure(); 

    plot_scale = 1.05;
    regimeNames  = analyzer.getAllRegimes()
    regime_tags = {'superT':'Supra-Threshold', 'crit':'Critical',
                    'subT':'Sub-Threshold',  'superSin':'Super-Sinusoidal',
                    'theta1':r'$\Omega=1$',
                    'theta5':r'$\Omega=5$',
                    'theta10':r'$\Omega=10$',
                    'theta20':r'$\Omega=20$'} 
    xlabel_font_size = 32
    
    for regime_idx,regime in enumerate(regimeNames):
        abg_true = analyzer.getTrueParamValues(regime)
        print regime, abg_true
            
        for method_idx, method in enumerate(Methods):
            abgs = array( analyzer.getAllEstimates(regime, method) )
            axu_lim = amax( r_[ abgs[:, 0], abg_true[0]])
            axl_lim = amin( r_[ abgs[:, 0], abg_true[0]])
            plot_buffer = 5*(plot_scale -1.)*(axu_lim - axl_lim);
            axu_lim += plot_buffer
            axl_lim -= plot_buffer
            
            ayu_lim = amax( r_[ abgs[:, 2], abg_true[2]])
            ayl_lim = amin( r_[ abgs[:, 2], abg_true[2]])
            plot_buffer = (plot_scale -1.)*(ayu_lim - ayl_lim);
            ayu_lim += plot_buffer
            ayl_lim -= plot_buffer
            
            ax = subplot(len(regimeNames),3,
                          regime_idx*len(Methods) + method_idx+1)
           
            plot(abgs[:, 0], abgs[:, 2],
                  linestyle='None',
                  color=  'k',   marker= 'o', markersize =  7  )
            if 0 == method_idx:
                text(-.3, 0.5, regime_tags[regime],
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform = ax.transAxes,
                        rotation='vertical',
                        size = xlabel_font_size)
            
#            axu_lim, axl_lim = xlim()
            hlines(abg_true[2], axl_lim, axu_lim,
                   linestyles = 'dashed' )
#            axu_lim, axl_lim = ylim()
            vlines(abg_true[0], ayl_lim, ayu_lim,
                   linestyles = 'dashed' )
            
            xlim((axl_lim, axu_lim));
            ylim((ayl_lim, ayu_lim));
            xlabel(r'$\alpha$',fontsize = xlabel_font_size);
            ylabel(r'$\gamma$',fontsize = xlabel_font_size);
            title(method  , fontsize = 32) #+  ' : $\\Omega= %.1f$' %sqrt((2.0*abg_true[2])**2 -1.)
            t = add_inner_title(ax, chr(ord('A') + 
                                        regime_idx*len(Methods) + method_idx),
                                loc=2, size=dict(size=ABCD_LABEL_SIZE))
            t.patch.set_ec("none")
            t.patch.set_alpha(0.5) 
        
    if '' != fig_id: 
        file_name = os.path.join(FIGS_DIR, fig_id + '_alphagamma_compare_joint.pdf')
        print 'saving to ', file_name
        savefig(file_name)



def tabulateEstimates(table_file_names,
                 methods):
    
    for table_file_name in table_file_names:
        print table_file_name
        analyzer = DataAnalyzer(table_file_name) 
        regimeNames = analyzer.getAllRegimes() 
        param_names = [r'\a',r'\b',r'\g']
        for regime in regimeNames:
            print r'\subfloat[%s]{\begin{tabular}{|c|ccc|} '%regime_labels[regime]
            abg_true = analyzer.getTrueParamValues(regime)
            print 'Parameter'
            for mthd in methods:
                print r'&', mthd
            print r'\\ \hline'               
            for idx in xrange(3):
                print  r'$%s=%.2f$' %(param_names[idx],abg_true[idx])
                for mthd in methods:
                    abg_est = array(analyzer.getAllEstimates(regime, mthd))
                    est = sort(abg_est[:, idx]) 
                    print r'& $%.2f : [%.2f, %.2f]$' %(mean(est), est[2], est[-3])
                print r'\\'
            print r' \end{tabular}}\\'               
        print 64*'#'

def compareTimes(table_names = ['FinalEstimate_4x100_N=100','FinalEstimate_4x100_N=1000']):
    regime_labels = {'subT': 'Sub-threshold',
                     'superT': 'Supra-threshold',
                     'superSin': 'Supra-sinusoidal',
                     'crit': 'Critical'}

    for table_name, N in zip(table_names,
                             [100, 1000]):
        analyzer = DataAnalyzer(table_name);
        print table_name
        for regime in analyzer.getAllRegimes():
            print regime_labels[regime]
            for mthd in [ 'Fortet', 'FP']:
                walltimes = analyzer.getWallTimes(regime, mthd)
                mu, sigma = mean(walltimes), std(walltimes)
                print '& %.2f'%mu, r'$\pm$ %.2f'%sigma
            print r'\\'
    
def flagWarnings(table_names):
    for table_name in table_names:
        analyzer = DataAnalyzer(table_name);
        print table_name
        print analyzer.getAllWarnings()

from BinnedSpikeTrain import BinnedSpikeTrain, intervalStats
from SpikeTrainSimulator import SpikeTrain, OUSinusoidalParams

def visualizeData_vs_Fortet(abg, binnedTrain, theta, title_tag = '', save_fig_name=''):
    from scipy.stats import norm
    bins = binnedTrain.bins
    phis = bins.keys()
    N_phi = len(phis)
    
    def getMovingThreshold(a,g, phi):
            psi = arctan(theta)
            mvt = lambda ts: 1. - ( a*(1 - exp(-ts)) + \
                                    g / sqrt(1+theta*theta) * ( sin ( theta *  ( ts + phi) - psi) \
                                                        -exp(-ts)*sin(phi*theta - psi) ))
            return mvt
    
    figure()
    
    a,b,g = abg[0], abg[1], abg[2]
    for (phi_m, phi_idx) in zip(phis, xrange(N_phi)):
        Is = bins[phi_m]['Is']
        uniqueIs = bins[phi_m]['unique_Is']
         
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
                        
        ax = subplot(N_phi, 1, phi_idx + 1);  hold (True)
        plot(uniqueIs, LHS, 'b', linewidth=3); 
        plot(uniqueIs, RHS, 'r+', markersize=12);
        xlim((.0, max(uniqueIs)))
        ylim((.0, max([max(LHS), max(RHS)])))
        if (0 == phi_idx):
            title(title_tag + "$\\alpha, \\beta, \gamma = (%.2g,%.2g,%.2g)$" %(abg[0], abg[1], abg[2]), fontsize = 42 )
            ylabel('$1- F(v_{th},t)$', fontsize = 24)
#        annotate('$\phi_{norm} = %.2g $'%(phi_m/2/pi*theta), (.1, max(LHS)/2.),
#                 fontsize = 24 )
#            ylabel('$\phi = %.2g \cdot 2\pi/ \\theta$'%(phi_m/2/pi*theta))
        
        if N_phi != phi_idx+1:
            setp(ax.get_xticklabels(), visible=False)
        else:
            xlabel('$t$', fontsize = 18)
#        setp(ax.get_yticklabels(), visible=False)
    
    get_current_fig_manager().window.showMaximized()    
    if '' != save_fig_name:
        filename = os.path.join(FIGS_DIR, save_fig_name + '.png')
        print 'Saving to ', filename
        savefig(filename)
      
        
def visualizeData_vs_FP(S, abgt, binnedTrain, title_tag = '', save_fig_name = ''):
        #Visualize time:
        abg = abgt
        
        mpl.rcParams['figure.subplot.left'] = .1
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .1
        mpl.rcParams['figure.subplot.top'] = .9
        
        figure()
        bins = binnedTrain.bins
        phis = bins.keys()
        Tf = binnedTrain.getTf()
        S.setTf(Tf)
        
        Fs = None
        if 3 == len(abgt):
            Fs = S.c_solve(abg)
        elif 4 == len(abgt):
            Fs = S.solve_tau(abgt)
            
        ts = S._ts;

        for (phi_m, err_idx) in zip( sort(phis),
                                     xrange(len(phis)) ):
            phi_idx = nonzero( abs(phis - phi_m) < 1e-4)
            lF = squeeze(Fs[phi_idx, :,-1])
            ax = subplot(len(phis),1, err_idx + 1)
            hold (True)
            plot(ts, lF, 'b',linewidth=3, label='Analytic'); 
            plot(bins[phi_m]['unique_Is'], 
                 bins[phi_m]['SDF'], 'r+', markersize = 10, label='Data')
            if (0 == err_idx):
                if '' != title_tag:
                    title(title_tag + ' $(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg[0], abg[1], abg[2]), fontsize = 36)
                ylabel('$1 - G(t)$', fontsize = 24)
                legend()
            if len(bins.keys()) != err_idx+1:
                setp(ax.get_xticklabels(), visible=False)
            else:
                xlabel('$t$', fontsize = 24)

            ylim((.0, 1.05))
               
#            annotate('$\phi_{norm}$ = %.2g'%(phi_m / 2 / pi * binnedTrain.theta), (.1, .5), fontsize = 24 )
        get_current_fig_manager().window.showMaximized()
        if '' != save_fig_name:
            file_name = os.path.join(FIGS_DIR, save_fig_name+'.png')
            print 'saving to ', file_name
            savefig(file_name) 

def postVisualizer():
    N_phi = 20
    print 'N_phi = ', N_phi
    phi_norms =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
    theta = 20

    base_name = 'sinusoidal_spike_train_N=1000_critical_theta=%d'%theta

    T_thresh = 64.
    
    analyzer = DataAnalyzer('ThetaEstimate_4x100_N=1000')
    sample_id = 32
    regime_label = base_name + '%d'%theta            
    file_name = base_name + '_%d'%sample_id
    print file_name

    binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phi_norms)
    binnedTrain.pruneBins(None, N_thresh = 1, T_thresh=T_thresh)
    
    regime_name = 'theta%d'%theta
    abg_true = analyzer.getTrueParamValues(regime_name);
    print abg_true
    
    
    abg_fortet = analyzer.getEstimates(sample_id, regime_name, 'Fortet')[0]
    print abg_fortet
    visualizeData_vs_Fortet(abg_fortet, binnedTrain, theta,title_tag = 'Fortet: estimates', save_fig_name='theta20_Fortet_estimates')
    visualizeData_vs_Fortet(abg_true, binnedTrain, theta,title_tag = 'Fortet: true', save_fig_name='theta20_Fortet_true')
    
    abg_fp = analyzer.getEstimates(sample_id, regime_name, 'FP')[0]
    print abg_fp
    dx = .025; dt = FPMultiPhiSolver.calculate_dt(dx, abg_true, -1.0) 
    phis = binnedTrain.bins.keys();
    S = FPMultiPhiSolver(binnedTrain.theta, phis,
                         dx, dt,
                         binnedTrain.getTf(), X_min = -1.0) 
    visualizeData_vs_FP(S, abg_fp, binnedTrain,title_tag = 'FP: estimates', save_fig_name='theta20_FP_estimates')
    visualizeData_vs_FP(S, abg_true, binnedTrain,title_tag = 'FP: true', save_fig_name='theta20_FP_true')
    


def postProcessJointBoxPlot(table_file_names, fig_id = '',
                 Methods =  ['Initializer', 'Nelder-Mead', 'Fortet'],
                 regimeNames = ['superT', 'subT', 'crit', 'superSin']):
    analyzers = [];    
    for table_file_name in table_file_names:
        analyzer = DataAnalyzer(table_file_name)  
        analyzers.append(analyzer)
    
    analyzer = analyzers[0]
    if None==regimeNames:
        regimeNames =  analyzer.getAllRegimes()
    
    for regime in regimeNames: 
        abg_true = analyzer.getTrueParamValues(regime)
        print regime, abg_true
    
    mpl.rcParams['figure.figsize'] = 17, 6
    method_labels = ['Initializer', 'Numerical FP', 'Fortet']
    xlabel_font_size = 24
    sample_size_texts = ['N=100', 'N=1000']
    
    
    for regime in regimeNames:
        
        yupper = 3*[-inf];
        ylower = 3*[inf]
        for params_idx in xrange(3):
            for mthd in Methods:
                for analyzer in analyzers:
                    ests = array(analyzer.getAllEstimates(regime, mthd))[:,params_idx]
                    ylower[params_idx] = amin([ylower[params_idx], amin(ests)])
                    yupper[params_idx] = amax([yupper[params_idx], amax(ests)])
        
        abg_true = analyzer.getTrueParamValues(regime)
        print regime, abg_true
        
        figure()
        subplots_adjust(left=0.1, right=0.975, top=0.9, bottom=0.25,
                         wspace = 0.25,  hspace = 0.01)
        
        lgs = GridSpec(2, 3)
        
        for sample_size_idx, analyzer in enumerate(analyzers):       
            abgs = {}
            for mthd in Methods:
                abgs[mthd] = array(analyzer.getAllEstimates(regime, mthd))
            
    #        method_xpos = dict.fromkeys(Methods, [1,2,3])
            
            param_tags = [r' \alpha ', r' \beta ', r' \gamma ']
                    
            for params_idx, param_tag in enumerate(param_tags):
                print params_idx, param_tag
                subplot_idx = sample_size_idx*3 + params_idx; 
                ax = subplot(lgs[subplot_idx]);  hold(True)
                
    #            for mthd in Methods:
                param_ests = [ abgs[mthd][:,params_idx] for mthd in Methods]
                param_true = abg_true[params_idx]
                    
                boxplot(param_ests, positions = [1,3,5])
                hlines(param_true, 0,5, linestyles='dashed')
                ylabel(r'$\hat{%s}$'%param_tag, fontsize = xlabel_font_size)  
                
                if (sample_size_idx ==0):
                    title(r'$%s = %.2f$'%(param_tag, param_true), fontsize = 32)
                if(sample_size_idx ==1):
                    xtickNames = setp(ax, xticklabels = method_labels)
                    setp(xtickNames, rotation=30, fontsize=xlabel_font_size)
                
                ylim((ylower[params_idx], yupper[params_idx]))
                ymin, ymax = ylim() 
                activeticks = linspace(ymin, ymax, 7)[1:-1]
                yticks(activeticks, ['%.2f'%tick for tick in activeticks])

                if 0 == params_idx:
                    text(-.25, 0.5, sample_size_texts[sample_size_idx],
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform = ax.transAxes,
                        rotation='vertical',
                        size = xlabel_font_size)
        
                tag_loc = 3;
#                if (median(param_ests[0]) > param_true):
#                    tag_loc = 3;
                xlim((.0, 5.5))
                t = add_inner_title(ax, chr(ord('A') + subplot_idx), loc=tag_loc,
                                     size=dict(size=ABCD_LABEL_SIZE))
                t.patch.set_ec("none")
                t.patch.set_alpha(0.5)
        
        file_name = os.path.join(FIGS_DIR, fig_id + regime+'_est_rel_errors_joint' +'.pdf')
        get_current_fig_manager().window.showMaximized()
        print 'saving to ', file_name
        
        savefig(file_name)


def postProcessThetaBoxPlots(table_file_name, fig_id = '',
                 Methods =  ['Initializer', 'FP', 'Fortet'],
                 regimeNames = ['theta1', 'theta5', 'theta10', 'theta20'] ):
    analyzer = DataAnalyzer(table_file_name)  
    
    if None==regimeNames:
        regimeNames =  analyzer.getAllRegimes()
    

    mpl.rcParams['figure.figsize'] = 17, 12
    figure()
    subplots_adjust(left=0.1, right=0.975, top=0.9, bottom=0.15,
                 wspace = 0.3,  hspace = 0.01)
    lgs = GridSpec(4,3)
    method_labels = ['Initializer', 'Numerical FP', 'Fortet']
    xlabel_font_size = 24
    sample_size_texts = ['N=100', 'N=1000']
    
    yupper = 3*[-inf];
    ylower = 3*[inf]
    for params_idx in xrange(3):
        for mthd in Methods:
            for regime in regimeNames:
                Omega = int(regime[5:])
                ests = array(analyzer.getAllEstimates(regime, mthd))[:,params_idx]
                
                if (2 == params_idx):
                    ests /= sqrt( 1. + Omega*Omega)
                    
                ylower[params_idx] = amin([ylower[params_idx], amin(ests)])
                yupper[params_idx] = amax([yupper[params_idx], amax(ests)])
            
    for regime_idx, regime in enumerate(regimeNames):
        abg_true = analyzer.getTrueParamValues(regime)
        Omega = int(regime[5:])
        print regime, abg_true, Omega

        abgs = {}
        param_tags = [r' \alpha ', r' \beta ', r' \gamma / \sqrt{1+ \Omega}']
        for mthd in Methods:
            abgs[mthd] = array(analyzer.getAllEstimates(regime, mthd))
            
        for params_idx, param_tag in enumerate(param_tags):
            print params_idx, param_tag
            subplot_idx = regime_idx*3 + params_idx;
             
            ax = subplot(lgs[subplot_idx]);  hold(True)
            
#            for mthd in Methods:
        
            param_ests = [ abgs[mthd][:,params_idx] for mthd in Methods]
            param_true = abg_true[params_idx]
            if params_idx == 2:
                param_true /= sqrt(1. + Omega*Omega)
                for idx in xrange(len(param_ests)):
                    param_ests[idx] /= sqrt(1. + Omega*Omega)

            boxplot(param_ests, positions = [1,3,5])
            hlines(param_true, 0,5, linestyles='dashed')
            ylabel(r'$\hat{%s}$'%param_tag, fontsize = xlabel_font_size)  
            
            if (regime_idx ==0):
                title(r'$%s = %.1f$'%(param_tag, param_true), fontsize = 32)
                if (params_idx == 2):
                    title(r'$\gamma = .5\cdot \sqrt{1+ \Omega}$', fontsize = 32)
            if(regime_idx == 3):
                xtickNames = setp(ax, xticklabels = method_labels)
                setp(xtickNames, rotation=30, fontsize=xlabel_font_size)
            
            ylim((ylower[params_idx], yupper[params_idx]))
            ymin, ymax = ylim() 
            activeticks = linspace(ymin, ymax, 7)[1:-1]
            yticks(activeticks, ['%.2f'%tick for tick in activeticks])

            if 0 == params_idx:
                text(-.25, 0.5, r'$\Omega = %s$' %regime[5:],
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform = ax.transAxes,
                    rotation='vertical',
                    size = xlabel_font_size)
    
            tag_loc = 3;
#            if (median(param_ests[0]) > param_true):
#                tag_loc = 3;
            xlim((.0,5.5 ))
            t = add_inner_title(ax, chr(ord('A') + subplot_idx), loc=tag_loc,
                                 size=dict(size=ABCD_LABEL_SIZE))
            t.patch.set_ec("none")
            t.patch.set_alpha(0.5)
        
    file_name = os.path.join(FIGS_DIR, fig_id + 'thetas_est_rel_errors' +'.pdf')
    get_current_fig_manager().window.showMaximized()
    print 'saving to ', file_name
    savefig(file_name)

def boxPlotBox():
        table_file_name =  'FinalEstimate_4x100_N=100'
        Methods =  ['Initializer', 'FP', 'Fortet']
        regime  =  'crit'  
        analyzer =  DataAnalyzer(table_file_name)

        mpl.rcParams['figure.figsize'] = 17, 6
        method_labels = ['Initializer', 'Numerical FP', 'Fortet']
        xlabel_font_size = 12

        abg_true = analyzer.getTrueParamValues(regime)
        print regime, abg_true
        
        figure()
        lgs = GridSpec(1, 3)       
        abgs = {}
        for mthd in Methods:
            abgs[mthd] = array(analyzer.getAllEstimates(regime, mthd))
        
#        method_xpos = dict.fromkeys(Methods, [1,2,3])
        
        param_tags = [r' \alpha ', r' \beta ', r' \gamma ']
                
        for params_idx, param_tag in enumerate(param_tags):
            print params_idx, param_tag
        
            ax_low = subplot(lgs[params_idx]);  hold(True)
            
#            for mthd in Methods:
            param_ests = [ abgs[mthd][:,params_idx] for mthd in Methods]
            param_true = abg_true[params_idx]
                
            boxplot(param_ests, positions = [1,3,5])
            hlines(param_true, 0,5, linestyles='dashed')
            title(r'$%s = %.2f$'%(param_tag, param_true))
            ylabel(r'$%s$'%param_tag, fontsize = xlabel_font_size)  
        
            xtickNames = setp(ax_low, xticklabels = method_labels)
            setp(xtickNames, rotation=60, fontsize=xlabel_font_size)    
#                
        subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.35,
                         wspace = 0.65,  hspace = 0.2)
#                axes(ax1); hold (True)
#                jitter = .4 *(-.5 + rand(len(diff_as)))
#                apos = 1.; bpos= 2.; gpos= 3.;
#                marker_size = 6;
#                plot(apos + jitter, capped_as, linestyle='None',
#                      color=  plot_colours[mthd], marker= markers[mthd],
#                      markersize = marker_size, label = labels[mthd])
#                plot(bpos + jitter, capped_bs, linestyle='None',
#                      color=  plot_colours[mthd], marker=markers[mthd],
#                      markersize = marker_size, label = None)
#                plot(gpos + jitter, capped_gs, linestyle='None',
#                      color=  plot_colours[mthd], marker=markers[mthd],
#                      markersize = marker_size, label = None)
#                xl = .5; xu = 3.5
#                plot([xl, xu], zeros(2), 'b', linestyle='dashed', linewidth = 5)
#                
#                xlim((xl, xu)); ylim((-ylims[regime], ylims[regime]))
#                if 0 == a_idx:
#                    ylabel('rel. error', fontsize = 20)
#    #            title('Relative errors for :$(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg_true[0], abg_true[1], abg_true[2]),
#    #                  fontsize = 22)
#    #            legend(loc = 'best')
#                
#                def relabel_x(x, pos):
#                        labels ={apos:'$\\alpha$', bpos:'$\\beta$', gpos:'$\\gamma$'}
#                        if x in [apos, bpos, gpos]:
#                            return labels[x]
#                        return ''
#                def relabel_y(x, pos):
#                    return '$%.1f$' %x
#                    
#                ax1.xaxis.set_major_formatter(FuncFormatter(relabel_x))
#                ax1.yaxis.set_major_formatter(FuncFormatter(relabel_y))
#                for label in ax1.xaxis.get_majorticklabels():
#                    label.set_fontsize(20)
#                for label in ax1.yaxis.get_majorticklabels():
#                    label.set_fontsize(20)
#
#                t = add_inner_title(ax1, chr(ord('A') + a_idx),
#                                     loc=2, size=dict(size=ABCD_LABEL_SIZE))
#                t = add_inner_title(ax2, chr(ord('A') + a_idx + len(table_file_names)),
#                                    loc=2, size=dict(size=ABCD_LABEL_SIZE))
#                t.patch.set_ec("none")
#                t.patch.set_alpha(0.5) 
#                    
#                                 
#                axes(ax2); hold(True)
#                l1_norm = abs(diff_as)+abs(diff_bs)+abs(diff_gs) 
#                #CAP IT:
#                yu = .12;
#                l1_norm = amin(c_[l1_norm, xlims[regime]*ones_like(l1_norm)], axis=1)
#                l1_offsets = {'Initializer':5*yu/8.,
#                              'FP':.0,
#                              'Fortet' :-5*yu / 8.}
#                offset = l1_offsets[mthd]
#                jitter = (yu/5.) * (-.5 + 1.*rand(diff_as.size))
#                
#                plot(l1_norm, offset + jitter,
#                     linestyle='None', color=plot_colours[mthd],
#                     marker=markers[mthd], markersize = marker_size,
#                     label = labels[mthd])
#                
#                ylim((-yu,yu)); xlim((-xlims[regime]/1.65, xlims[regime]))
#                plot(zeros(3),[-.5,.0,.5],  linestyle='dashed', linewidth = 5)
#    #            xlabel('$l^1$ relative error', fontsize = 20);
#                xlabel('sum of absolute errors', fontsize = 20);
#                
#                setp(ax2.get_yticklabels(), visible=False)
#                def relabel_major(x, pos):
#                        labels ={apos:'$\\alpha$', bpos:'$\\beta$', gpos:'$\\gamma$'}
#                        if x < 0:
#                            return ''
#                        else:
#                            return '$' +str(x) + '$'
#                ax2.xaxis.set_major_formatter(FuncFormatter(relabel_major))
#                for label in ax2.xaxis.get_majorticklabels():
#                    label.set_fontsize(24)
#                legend(loc = 'upper left')
        
        
        

if __name__ == '__main__':
    from pylab import * 
#   
#    mpl.rcParams['figure.figsize'] = 17, 7
    mpl.rcParams['figure.dpi'] = 300
#    InitsComparison('init_comparison')
#    NM_SubT()
#    NM_SubT('Live4x16')
#    CvsPY()
#    FortetPostProcessor('Fortet_SupVsL2_4x20')
#    WeightedFortet_NBox('FvsWF_4x16')

#    flagWarnings(['FinalEstimate_4x100_N=%d'%N for N in [100, 1000]])
#    rylims, rxlims = calculateYXlims('FinalEstimate_4x100_N=100',
#                                     rylim_max = 20.0, rxlim_max=20.)
#    postProcess('FinalEstimate_4x100_N=100', fig_id='FP_vs_Fortet_100x100',
#                Methods =  [ 'Initializer','FP', 'Fortet'],
#                rylims=rylims, rxlims=rxlims)
##                rylims = {'crit': 0.63427540126341975, 'superT': 0.67129194736480713, 'subT': 2.7882548006138981, 'superSin': 9.3433702834938419},
##               rxlims = {'crit': 1.4555448802684241, 'superT': 1.2055090582032593, 'subT': 4.5787803089499128, 'superSin': 10.280992690142302})
#    crossAnalyze('FinalEstimate_4x100_N=100',
#                  fig_id='FP_vs_Fortet_100x100',
#                  regimeNames = ['superT', 'superSin', 'crit', 'subT'])#,                 Method1='FP', Method2='Fortet')
##    rylims, rxlims = calculateYXlims('FinalEstimate_4x100_N=1000',
##                                     rylim_max = 20.0, rxlim_max=20.)
##    postProcess('FinalEstimate_4x100_N=1000', fig_id='FP_vs_Fortet_100x1000',
##                Methods =  [ 'Initializer','FP', 'Fortet'],
##                rylims=rylims, rxlims=rxlims)
#    crossAnalyze('FinalEstimate_4x100_N=1000',
#                  fig_id='FP_vs_Fortet_100x1000',
#                  regimeNames = ['superT', 'superSin', 'crit', 'subT'])
    
#    tabulateEstimates(['FinalEstimate_4x100_N=100',
#                      'FinalEstimate_4x100_N=1000'],
#                    methods =  [ 'Initializer','FP', 'Fortet'])
#    compareTimes()
#    postProcessJoint(['FinalEstimate_4x100_N=100',
#                       'FinalEstimate_4x100_N=1000'],
#             fig_id='FP_vs_Fortet_100x100_x1000',
#             Methods =  [ 'Initializer','FP', 'Fortet'])

########################################################################
#########THETA EXPLORATION:::
########################################################################
#    flagWarnings(['ThetaEstimate_4x100_N=1000'])
#    rylims, rxlims = calculateYXlims('ThetaEstimate_4x100_N=1000',rylim_max = 20.0, rxlim_max=20. , make_all_lims_same = True)
#    postProcessTheta('ThetaEstimate_4x100_N=1000',
#                      fig_id='thetas_100x1000',
#                      Methods =  [ 'Initializer','FP', 'Fortet'],
#                      regimeNames = ['theta1', 'theta5', 'theta10', 'theta20'])
    crossAnalyze('ThetaEstimate_4x100_N=1000', 
                 fig_id='MLE')
#    alphaVsGamma('ThetaEstimate_4x100_N=1000',
#                  fig_id='thetavariation_100x1000')

#    tabulateEstimates(['ThetaEstimate_4x100_N=1000'],
#                    methods =  [ 'Initializer','FP', 'Fortet'])
    
#    postVisualizer()


#    crossAnalyzeJoint('FinalEstimate_4x100_N=100',
#                      'FinalEstimate_4x100_N=1000',
#                  fig_id='FP_vs_Fortet_Ns_joint',
#                  regimeNames = ['superT', 'superSin', 'crit', 'subT'])
    
#    boxPlotBox()
#    postProcessJointBoxPlot(['FinalEstimate_4x100_N=100',
#                             'FinalEstimate_4x100_N=1000'],
#                            fig_id='FP_vs_Fortet_100x100_x1000',
#                            Methods =  [ 'Initializer','FP', 'Fortet'])
#    postProcessThetaBoxPlots('ThetaEstimate_4x100_N=1000',
#                             fig_id='thetas_100x1000',
#                             regimeNames = ['theta1', 'theta5', 'theta10', 'theta20'])
#  
    show() 
    