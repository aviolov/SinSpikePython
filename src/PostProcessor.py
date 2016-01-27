'''
Created on May 10, 2012

@author: alex
'''

from tables import *
from DataHarvester import TABLES_DIR, DataAnalyzer
import os
from matplotlib.gridspec import GridSpec
from numpy import array, tile
from FPMultiPhiSolver import FPMultiPhiSolver
from Simulator import ABCD_LABEL_SIZE

FIGS_DIR = '/home/alex/Workspaces/Latex/LIFEstimation/Figs/Estimates'

from numpy import sum, absolute, amin

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
           'FP':'Numerical FP'
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
        
def calculateL1Sup(abg, abg_true):
    diffs = absolute(array(abg) - array(abg_true))
    l1 = sum(diffs)
    sup = max(diffs)
    
    return l1, sup

def calculateYXlims(table_file_name,
                     Methods = ['FP', 'Fortet'],                     
                     rylim_max = 4.0, rxlim_max = 4.0,
                     make_all_lims_same=False):
#TODO: this only returns the relative lims, easy to add the abs lims:
#    ylims = {}
#    xlims = {}
    rylims = {}
    rxlims = {}
        
    analyzer = DataAnalyzer(table_file_name)   
    
    regimeNames = analyzer.getAllRegimes()
    
    def capit(rylim, rxlim):
        return min(rylim, rylim_max), min(rxlim, rxlim_max)
    
    for regime in regimeNames:
        y_per_param_lim = .0;
        x_l1_lim = .0;
        for idx, mthd in enumerate(Methods):
#            print idx, mthd

            abg = array(analyzer.getAllEstimates(regime, mthd))
            abg_true = array(analyzer.getTrueParamValues(regime))
            
            n_estimates = abg.shape[0]
            abg_true = tile(abg_true, n_estimates).reshape(n_estimates,3);
            abg_rel_errors = (abg - abg_true) / abg_true
            
            y_per_param_lim = max(y_per_param_lim, amax(abs(abg_rel_errors)))
            x_l1_lim        = max(x_l1_lim,
                                  amax(sum(abs(abg_rel_errors), axis = 1)))
            
            
        y_per_param_lim, x_l1_lim = capit(y_per_param_lim, x_l1_lim )
        
        rylims[regime] = y_per_param_lim
        rxlims[regime] = x_l1_lim
    
    
    if make_all_lims_same:
        for D in [rylims, rxlims]:
            for key in D.iterkeys():
                D[key] = amax(D.values()) 
    
    
    return rylims, rxlims
            
    

def postProcessJoint(table_file_names, fig_id = '',
                 Methods =  ['Initializer', 'Nelder-Mead', 'Fortet'],
                 regimeNames = ['superT', 'subT', 'crit', 'superSin'],
                 ylims = None, 
                 xlims = None):
    analyzers = [];    
    for table_file_name in table_file_names:
        analyzer = DataAnalyzer(table_file_name)  
        analyzers.append(analyzer)
    
    analyzer = analyzers[0]
    
    if None==regimeNames:
        regimeNames =  analyzer.getAllRegimes()
    
    if None == ylims:
        ylims, xlims = calculateYXlims(table_file_names[0], Methods)
    
    for regime in regimeNames:
        ylims[regime] *= 1.05
        xlims[regime] *= 1.05    
    
    for regime in regimeNames: 
        abg_true = analyzer.getTrueParamValues(regime)
        print regime, abg_true
    
    for regime in regimeNames:
        abg_true = analyzer.getTrueParamValues(regime)
        
        print regime, abg_true
                
        def capit(diffs, dmax):
            e = ones_like(diffs); 
            d_capped = amin(c_[dmax*e, diffs], axis=1)
            d_capped = amax(c_[-dmax*e, d_capped], axis=1)
            return d_capped
        mpl.rcParams['figure.figsize'] = 17, 6
        mpl.rcParams['figure.subplot.left'] = .175
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .125
        mpl.rcParams['figure.subplot.top'] = .95
        figure()
        
        lgs = GridSpec(2, len(table_file_names), height_ratios=[2, 1])
        
        for a_idx, analyzer in enumerate(analyzers):
            ax1 = subplot(lgs[a_idx]);
            ax2 = subplot(lgs[a_idx + len(table_file_names)]);
            hold(True)
            for mthd in Methods:
                abgs = array( analyzer.getAllEstimates(regime, mthd) )

                diff_as = (abgs [:,0] - abg_true[0])/ abg_true[0];
                diff_bs = (abgs [:,1] - abg_true[1])/ abg_true[1];
                diff_gs = (abgs [:,2] - abg_true[2])/ abg_true[2];
                
                capped_as = capit(diff_as, ylims[regime])
                capped_bs = capit(diff_bs, ylims[regime])
                capped_gs = capit(diff_gs, ylims[regime])
                
                axes(ax1); hold (True)
                jitter = .4 *(-.5 + rand(len(diff_as)))
                apos = 1.; bpos= 2.; gpos= 3.;
                marker_size = 6;
                plot(apos + jitter, capped_as, linestyle='None',
                      color=  plot_colours[mthd], marker= markers[mthd],
                      markersize = marker_size, label = labels[mthd])
                plot(bpos + jitter, capped_bs, linestyle='None',
                      color=  plot_colours[mthd], marker=markers[mthd],
                      markersize = marker_size, label = None)
                plot(gpos + jitter, capped_gs, linestyle='None',
                      color=  plot_colours[mthd], marker=markers[mthd],
                      markersize = marker_size, label = None)
                xl = .5; xu = 3.5
                plot([xl, xu], zeros(2), 'b', linestyle='dashed', linewidth = 5)
                
                xlim((xl, xu)); ylim((-ylims[regime], ylims[regime]))
                if 0 == a_idx:
                    ylabel('rel. error', fontsize = 20)
    #            title('Relative errors for :$(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg_true[0], abg_true[1], abg_true[2]),
    #                  fontsize = 22)
    #            legend(loc = 'best')
                
                def relabel_x(x, pos):
                        labels ={apos:'$\\alpha$', bpos:'$\\beta$', gpos:'$\\gamma$'}
                        if x in [apos, bpos, gpos]:
                            return labels[x]
                        return ''
                def relabel_y(x, pos):
                    return '$%.1f$' %x
                    
                ax1.xaxis.set_major_formatter(FuncFormatter(relabel_x))
                ax1.yaxis.set_major_formatter(FuncFormatter(relabel_y))
                for label in ax1.xaxis.get_majorticklabels():
                    label.set_fontsize(20)
                for label in ax1.yaxis.get_majorticklabels():
                    label.set_fontsize(20)

                t = add_inner_title(ax1, chr(ord('A') + a_idx),
                                     loc=2, size=dict(size=ABCD_LABEL_SIZE))
                t = add_inner_title(ax2, chr(ord('A') + a_idx + len(table_file_names)),
                                    loc=2, size=dict(size=ABCD_LABEL_SIZE))
                t.patch.set_ec("none")
                t.patch.set_alpha(0.5) 
                    
                                 
                axes(ax2); hold(True)
                l1_norm = abs(diff_as)+abs(diff_bs)+abs(diff_gs) 
                #CAP IT:
                yu = .12;
                l1_norm = amin(c_[l1_norm, xlims[regime]*ones_like(l1_norm)], axis=1)
                l1_offsets = {'Initializer':5*yu/8.,
                              'FP':.0,
                              'Fortet' :-5*yu / 8.}
                offset = l1_offsets[mthd]
                jitter = (yu/5.) * (-.5 + 1.*rand(diff_as.size))
                
                plot(l1_norm, offset + jitter,
                     linestyle='None', color=plot_colours[mthd],
                     marker=markers[mthd], markersize = marker_size,
                     label = labels[mthd])
                
                ylim((-yu,yu)); xlim((-xlims[regime]/1.65, xlims[regime]))
                plot(zeros(3),[-.5,.0,.5],  linestyle='dashed', linewidth = 5)
    #            xlabel('$l^1$ relative error', fontsize = 20);
                xlabel('sum of absolute errors', fontsize = 20);
                
                setp(ax2.get_yticklabels(), visible=False)
                def relabel_major(x, pos):
                        labels ={apos:'$\\alpha$', bpos:'$\\beta$', gpos:'$\\gamma$'}
                        if x < 0:
                            return ''
                        else:
                            return '$' +str(x) + '$'
                ax2.xaxis.set_major_formatter(FuncFormatter(relabel_major))
                for label in ax2.xaxis.get_majorticklabels():
                    label.set_fontsize(24)
                legend(loc = 'upper left')
            
        file_name = os.path.join(FIGS_DIR, fig_id + regime+'_est_rel_errors_joint' +'.pdf')
        get_current_fig_manager().window.showMaximized()
        print 'saving to ', file_name
        
        savefig(file_name)


def postProcessTheta(table_file_name, fig_id = '',
                 Methods =  ['Initializer', 'Nelder-Mead', 'Fortet'],
                 regimeNames = ['superT', 'subT', 'crit', 'superSin'],
                 ylims = None, 
                 xlims = None):
    analyzer = DataAnalyzer(table_file_name)  
    
    if None==regimeNames:
        regimeNames =  analyzer.getAllRegimes()
    
    if None == ylims:
        ylims, xlims = calculateYXlims(table_file_name, Methods)
    
    for regime in regimeNames:
        ylims[regime] *= 1.05
        xlims[regime] *= 1.05    
    
    mpl.rcParams['figure.figsize'] = 17, 6
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.top'] = .925
    mpl.rcParams['figure.subplot.bottom'] = .125
    mpl.rcParams['figure.subplot.wspace'] = .3
    figure()
    
    lgs = GridSpec(2, len(regimeNames), height_ratios=[2, 1])
    def capit(diffs, dmax):
        e = ones_like(diffs); 
        d_capped = amin(c_[dmax*e, diffs], axis=1)
        d_capped = amax(c_[-dmax*e, d_capped], axis=1)
        return d_capped
    
    for regime_idx, regime in enumerate(regimeNames):
        abg_true = analyzer.getTrueParamValues(regime)
        
        print regime, abg_true
   
        ax1 = subplot(lgs[regime_idx]);
        ax2 = subplot(lgs[regime_idx + len(regimeNames)]);
        hold(True)
        for mthd in Methods:
            abgs = array( analyzer.getAllEstimates(regime, mthd) )

            diff_as = (abgs [:,0] - abg_true[0])/ abg_true[0];
            diff_bs = (abgs [:,1] - abg_true[1])/ abg_true[1];
            diff_gs = (abgs [:,2] - abg_true[2])/ abg_true[2];
            
            capped_as = capit(diff_as, ylims[regime])
            capped_bs = capit(diff_bs, ylims[regime])
            capped_gs = capit(diff_gs, ylims[regime])
            
            axes(ax1); hold (True)
            jitter = .4 *(-.5 + rand(len(diff_as)))
            apos = 1.; bpos= 2.; gpos= 3.;
            marker_size = 6;
            plot(apos + jitter, capped_as, linestyle='None',
                  color=  plot_colours[mthd], marker= markers[mthd],
                  markersize = marker_size, label = labels[mthd])
            plot(bpos + jitter, capped_bs, linestyle='None',
                  color=  plot_colours[mthd], marker=markers[mthd],
                  markersize = marker_size, label = None)
            plot(gpos + jitter, capped_gs, linestyle='None',
                  color=  plot_colours[mthd], marker=markers[mthd],
                  markersize = marker_size, label = None)
            xl = .5; xu = 3.5
            plot([xl, xu], zeros(2), 'b', linestyle='dashed', linewidth = 5)
            
            title(r'$\Omega = %s$' %regime[5:], fontsize = 36)
            
            y_lim = amax(ylims.values());
            
            xlim((xl, xu)); ylim((-y_lim, y_lim ))
            if 0 == regime_idx:
                ylabel('rel. error', fontsize = 20)
#            title('Relative errors for :$(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg_true[0], abg_true[1], abg_true[2]),
#                  fontsize = 22)
            if 0 == regime_idx:
                legend(loc = 'upper right')
                        
            def relabel_x(x, pos):
                    labels ={apos:'$\\alpha$', bpos:'$\\beta$', gpos:'$\\gamma$'}
                    if x in [apos, bpos, gpos]:
                        return labels[x]
                    return ''
            def relabel_y(x, pos):
                return '$%.1f$' %x
                
            ax1.xaxis.set_major_formatter(FuncFormatter(relabel_x))
            ax1.yaxis.set_major_formatter(FuncFormatter(relabel_y))
            for label in ax1.xaxis.get_majorticklabels():
                label.set_fontsize(20)
            for label in ax1.yaxis.get_majorticklabels():
                label.set_fontsize(20)

            t = add_inner_title(ax1, chr(ord('A') + regime_idx), loc=4,
                                 size=dict(size=ABCD_LABEL_SIZE))
            t = add_inner_title(ax2, chr(ord('A') + regime_idx + len(regimeNames)),
                                 loc=4, size=dict(size=ABCD_LABEL_SIZE))
            t.patch.set_ec("none")
            t.patch.set_alpha(0.5) 
                
                             
            axes(ax2); hold(True)
            l1_norm = abs(diff_as)+abs(diff_bs)+abs(diff_gs) 
            #CAP IT:
            yu = .12;
            l1_norm = amin(c_[l1_norm, xlims[regime]*ones_like(l1_norm)], axis=1)
            l1_offsets = {'Initializer':5*yu/8.,
                          'FP':.0,
                          'Fortet' :-5*yu / 8.}
            offset = l1_offsets[mthd]
            jitter = (yu/5.) * (-.5 + 1.*rand(diff_as.size))
            
            plot(l1_norm, offset + jitter,
                 linestyle='None', color=plot_colours[mthd],
                 marker=markers[mthd], markersize = marker_size,
                 label = labels[mthd])
            
            x_lim = amax(xlims.values())
            ylim((-yu,yu)); xlim((-.1,x_lim ))
            plot(zeros(3),[-.5,.0,.5],  linestyle='dashed', linewidth = 5)
#            xlabel('$l^1$ relative error', fontsize = 20);
            xlabel('sum of absolute errors', fontsize = 20);
            
            
            setp(ax2.get_yticklabels(), visible=False)
            def relabel_major(x, pos):
                    labels ={apos:'$\\alpha$', bpos:'$\\beta$', gpos:'$\\gamma$'}
                    if x < 0:
                        return ''
                    else:
                        return '$' +str(x) + '$'
            ax2.xaxis.set_major_locator(MaxNLocator(4))    
            ax2.xaxis.set_major_formatter(FuncFormatter(relabel_major))
            for label in ax2.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            
    file_name = os.path.join(FIGS_DIR, fig_id + 'thetas_est_rel_errors' +'.pdf')
    print 'saving to ', file_name
    savefig(file_name)
    

def postProcess(table_file_name, fig_id = '',
                 Methods =  ['Initializer', 'Nelder-Mead', 'Fortet'],
                 regimeNames = ['superT', 'subT', 'crit', 'superSin'],
                 ylims = {'superT':.1, 'subT':.5, 'crit':.1, 'superSin':.25},
                 xlims = {'superT':.5, 'subT':1.75, 'crit':.5, 'superSin':1.75},
                 rylims = {'crit': 0.63427540126341975, 'superT': 0.67129194736480713, 
                          'subT': 2.7882548006138981, 'superSin': 9.3433702834938419}, 
                 rxlims = {'crit': 1.4555448802684241, 'superT': 1.2055090582032593, 
                          'subT': 4.5787803089499128, 'superSin': 10.280992690142302}):
#    rylims = {'superT':.5, 'subT':1.5, 'crit':.5, 'superSin':1.}
#    rxlims = {'superT':.5, 'subT':2.5, 'crit':1., 'superSin':1.5}
    
    file_name = os.path.join(TABLES_DIR, table_file_name+ '.h5')
    h5file = openFile(file_name, mode = "r")

    Estimates = h5file.getNode('/', 'Estimates')
    Samples =  h5file.getNode('/', 'Samples')
    regimesTable =  h5file.getNode('/', 'Regimes')._f_getChild('Regimes')
    
#    Methods = ['Initializer', 'Nelder-Mead', 'BFGS', 'Fortet']
    
#    labels = {'Initializer':'Initializer', 'Nelder-Mead':'F-P absorb. bound.', 'BFGS':'F-P dtve', 'Fortet':'Fortet'}
     
    
    for regime in regimeNames:
        rylims[regime] *= 1.05
        rxlims[regime] *= 1.05
 
    labels = {'Initializer':'Initializer', 'Nelder-Mead':'Absorb. Bnd.',
              'BFGS':'F-P dtve', 'Fortet':'Fortet',
              'init_N10':'Initializer',
              'FP_N10': 'FP_N10',
              'WFP_N1': 'WFP_N1',
              'FP_L2':"L2", 'FP_Sup':'Sup',
              'FP':'Numerical FP'
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
    
    
    for regime in regimeNames:
        samplesT = Samples._f_getChild(regime)
        estimatesT = Estimates._f_getChild(regime)
        samplesCount = len(samplesT)
        
        abg_true = [];
        for row in  regimesTable.where('name == "'+ regime+'"'):
            abg_true = [row['alpha'], row['beta'], row['gamma']]
                
        print regime, abg_true
        
        mpl.rcParams['figure.subplot.left'] = .175
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .125
        mpl.rcParams['figure.subplot.top'] = .95
#AABSOLUTE ERRORS:
#        figure()
#        hold(True)
#        for mthd in Methods:
#            abgs = empty((samplesCount, 3))
#            idx= 0;
#            for row in estimatesT.where('method == "'+ mthd +'"'):
#                try:
#                    abgs[idx,:] = row['alpha'], row['beta'], row['gamma']
#                except:
#                    pass
#                idx +=1;
#            
#            diff_as = abgs [:,0] - abg_true[0];
#            diff_bs = abgs [:,1] - abg_true[1];
#            diff_gs = abgs [:,2] - abg_true[2];
#             
#            ax = subplot(211); hold (True)
#            plot(1. + -.1+ .1*rand(len(diff_bs)), diff_as, linestyle='None',
#                  color=  plot_colours[mthd], marker= markers[mthd], markersize = 12, label = labels[mthd])
#            plot(1.5 + -.1+ .1*rand(len(diff_bs)), diff_bs, linestyle='None',
#                  color=  plot_colours[mthd], marker=markers[mthd], markersize = 12, label = None)
#            plot(2.0 + -.1+ .1*rand(len(diff_bs)), diff_gs, linestyle='None',
#                  color=  plot_colours[mthd], marker=markers[mthd], markersize = 12, label = None)
#            plot([1., 1.5, 2.0], zeros(3), 'b', linestyle='dashed', linewidth = 5)
#            
#            xlim((.75, 3.0)); ylim((-ylims[regime], ylims[regime]))
#            ylabel('error', fontsize = 20)
##            title('Estimate Errors for ' + regime + ' :$(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg_true[0], abg_true[1], abg_true[2]),
##               fontsize = 18)
#            title('Absolute Estimate Errors for ' + regime + ': $(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg_true[0], abg_true[1], abg_true[2]),
#               fontsize = 18)
##            legend(loc = 'best')
#            
#            def relabel_major(x, pos):
#                    labels ={1.:'$\\alpha$', 1.5:'$\\beta$', 2.0:'$\\gamma$'}
#                    if x in [1., 1.5, 2.0]:
#                        return labels[x]
#                    return ''
#            ax.xaxis.set_major_formatter(FuncFormatter(relabel_major))
#            for label in ax.xaxis.get_majorticklabels():
#                label.set_fontsize(24)
#            for label in ax.yaxis.get_majorticklabels():
#                label.set_fontsize(24)
#                
#            ax = subplot(212); hold(True)
#            l1_norm = abs(diff_as)+abs(diff_bs)+abs(diff_gs) 
#            #CAP IT:
#            l1_norm = amin(c_[l1_norm, xlims[regime]*ones_like(l1_norm)], axis=1)
#          
#            plot(l1_norm, -.1 + .1*rand(diff_as.size),
#                 linestyle='None', color=plot_colours[mthd], marker=markers[mthd], markersize = 12, label = mthd)
#            ylim((-.25,.25)); xlim((-.01, xlims[regime]))
#            setp(ax.get_yticklabels(), visible=False)
#            plot(zeros(3),[-.5,.0,.5],  linestyle='dashed', linewidth = 5)
#            xlabel('$l^1$ error', fontsize = 20);
#            for label in ax.xaxis.get_majorticklabels():
#                label.set_fontsize(24)
#            legend(loc = 'best')
#            
#        file_name = os.path.join(FIGS_DIR, fig_id + regime + '_est_errors' + '.png')
#        get_current_fig_manager().window.showMaximized()
#        print 'saving to ', file_name
#        savefig(file_name)
#
#        close()
        
        def capit(diffs, dmax):
            e = ones_like(diffs); 
            d_capped = amin(c_[dmax*e, diffs], axis=1)
            d_capped = amax(c_[-dmax*e, d_capped], axis=1)
            return d_capped
        
        #RELATIVE ERRORS:
        figure()
        lgs = GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = subplot(lgs[0]);
        ax2 = subplot(lgs[1]);
        hold(True)
        for mthd in Methods:
            abgs = empty((samplesCount, 3))
            idx= 0;
            for row in estimatesT.where('method == "'+ mthd +'"'):
                abgs[idx,:] = row['alpha'], row['beta'], row['gamma']
                idx +=1;
            
            diff_as = (abgs [:,0] - abg_true[0])/ abg_true[0];
            diff_bs = (abgs [:,1] - abg_true[1])/ abg_true[1];
            diff_gs = (abgs [:,2] - abg_true[2])/ abg_true[2];
            
            capped_as = capit(diff_as, rylims[regime])
            capped_bs = capit(diff_bs, rylims[regime])
            capped_gs = capit(diff_gs, rylims[regime])
            
            
            
            axes(ax1); hold (True)
            jitter = .4 *(-.5 + rand(len(diff_as)))
            apos = 1.; bpos= 2.; gpos= 3.;
            marker_size = 8;
            plot(apos + jitter, capped_as, linestyle='None',
                  color=  plot_colours[mthd], marker= markers[mthd],
                  markersize = marker_size, label = labels[mthd])
            plot(bpos + jitter, capped_bs, linestyle='None',
                  color=  plot_colours[mthd], marker=markers[mthd],
                  markersize = marker_size, label = None)
            plot(gpos + jitter, capped_gs, linestyle='None',
                  color=  plot_colours[mthd], marker=markers[mthd],
                  markersize = marker_size, label = None)
            xl = .5; xu = 3.5
            plot([xl, xu], zeros(2), 'b', linestyle='dashed', linewidth = 5)
            
            xlim((xl, xu)); ylim((-rylims[regime], rylims[regime]))
            ylabel('rel. error', fontsize = 20)
            title('Relative errors for :$(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg_true[0], abg_true[1], abg_true[2]),
                  fontsize = 22)
#            legend(loc = 'best')
            
            def relabel_major(x, pos):
                    labels ={apos:'$\\alpha$', bpos:'$\\beta$', gpos:'$\\gamma$'}
                    if x in [apos, bpos, gpos]:
                        return labels[x]
                    return ''
              
            ax1.xaxis.set_major_formatter(FuncFormatter(relabel_major))
            for label in ax1.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            for label in ax1.yaxis.get_majorticklabels():
                label.set_fontsize(24)   
                
                             
            axes(ax2); hold(True)
            l1_norm = abs(diff_as)+abs(diff_bs)+abs(diff_gs) 
            #CAP IT:
            yu = .12;
            l1_norm = amin(c_[l1_norm, rxlims[regime]*ones_like(l1_norm)], axis=1)
            l1_offsets = {'Initializer':5*yu/8.,
                          'FP':.0,
                          'Fortet' :-5*yu / 8.}
            offset = l1_offsets[mthd]
            jitter = (yu/5.) * (-.5 + 1.*rand(diff_as.size))
            
            plot(l1_norm, offset + jitter,
                 linestyle='None', color=plot_colours[mthd],
                 marker=markers[mthd], markersize = marker_size,
                 label = labels[mthd])
            
            ylim((-yu,yu)); xlim((-rxlims[regime]/1.75, rxlims[regime]))
            plot(zeros(3),[-.5,.0,.5],  linestyle='dashed', linewidth = 5)
#            xlabel('$l^1$ relative error', fontsize = 20);
            xlabel('sum of absolute errors', fontsize = 20);
            
            setp(ax2.get_yticklabels(), visible=False)
            def relabel_major(x, pos):
                    labels ={apos:'$\\alpha$', bpos:'$\\beta$', gpos:'$\\gamma$'}
                    if x < 0:
                        return ''
                    else:
                        return '$' +str(x) + '$'
            ax2.xaxis.set_major_formatter(FuncFormatter(relabel_major))
            for label in ax2.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            legend(loc = 'upper left')
            
        file_name = os.path.join(FIGS_DIR, fig_id + regime+'_est_rel_errors' +'.png')
        get_current_fig_manager().window.showMaximized()
        print 'saving to ', file_name
        
        savefig(file_name)
         
        
    h5file.close()



def postProcess_AbsErrorsPerParam(table_file_name, ):
    file_name = os.path.join(TABLES_DIR, table_file_name+ '.h5')
    h5file = openFile(file_name, mode = "r")

    Estimates = h5file.getNode('/', 'Estimates')
    Samples =  h5file.getNode('/', 'Samples')
    regimesTable =  h5file.getNode('/', 'Regimes')._f_getChild('Regimes')
    
#    Methods = ['Initializer', 'Nelder-Mead', 'BFGS', 'Fortet']
    Methods =  ['Initializer', 'Nelder-Mead', 'Fortet']
#    labels = {'Initializer':'Initializer', 'Nelder-Mead':'F-P absorb. bound.', 'BFGS':'F-P dtve', 'Fortet':'Fortet'}
#    labels = {'Initializer':'Initializer', 'Nelder-Mead':'Absorb. Bnd.', 'BFGS':'F-P dtve', 'Fortet':'Fortet'}
    labels = {'Initializer':'Initializer', 'Nelder-Mead':"Numeric Soln to F-P", 'BFGS':'F-P dtve', 'Fortet':'Fortet'}

    regimeNames = ['superT', 'subT', 'crit', 'superSin'] 
    ylims = {'superT':.1, 'subT':.5, 'crit':.15, 'superSin':.25}
#    xlims = {'superT':.5, 'subT':1.75, 'crit':.5, 'superSin':1.75}
# 
#    rylims = {'superT':.5, 'subT':1.5, 'crit':.5, 'superSin':1.}
#    rxlims = {'superT':.5, 'subT':2.5, 'crit':1., 'superSin':1.5}
# 
    plot_colours = {'Fortet':"b", 'Initializer':"g", 'BFGS':"r",
                     'Nelder-Mead':"y"}
    markers = {'Fortet':"d", 'Initializer':"*", 'BFGS':'h',
                     'Nelder-Mead':"h"}
    
    for regime in regimeNames:
        samplesT = Samples._f_getChild(regime)
        estimatesT = Estimates._f_getChild(regime)
        samplesCount = len(samplesT)
        
        abg_true = [];
        for row in  regimesTable.where('name == "'+ regime+'"'):
            abg_true = [row['alpha'], row['beta'], row['gamma']]
                
        print regime, abg_true
        
        mpl.rcParams['figure.subplot.left'] = .175
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .125
        mpl.rcParams['figure.subplot.top'] = .95
    
        figure()
        hold(True)
        for mthd in Methods:
            abgs = empty((samplesCount, 3))
            idx= 0;
            for row in estimatesT.where('method == "'+ mthd +'"'):
                abgs[idx,:] = row['alpha'], row['beta'], row['gamma']
                idx +=1;
            
            diff_as = abgs [:,0] - abg_true[0];
            diff_bs = abgs [:,1] - abg_true[1];
            diff_gs = abgs [:,2] - abg_true[2];
             
            ax = subplot(111); hold (True)
            plot(1. + -.1+ .1*rand(len(diff_bs)), diff_as, linestyle='None',
                  color=  plot_colours[mthd], marker= markers[mthd], markersize = 12, label = labels[mthd])
            plot(1.5 + -.1+ .1*rand(len(diff_bs)), diff_bs, linestyle='None',
                  color=  plot_colours[mthd], marker=markers[mthd], markersize = 12, label = None)
            plot(2.0 + -.1+ .1*rand(len(diff_bs)), diff_gs, linestyle='None',
                  color=  plot_colours[mthd], marker=markers[mthd], markersize = 12, label = None)
            plot([1., 1.5, 2.0], zeros(3), 'b', linestyle='dashed', linewidth = 5)
            
            xlim((.75, 3.0)); ylim((-ylims[regime], ylims[regime]))
            ylabel('error', fontsize = 20)
            title('Estimate Errors for ' + regime + ' :$(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg_true[0], abg_true[1], abg_true[2]),
               fontsize = 18)
            legend(loc = 'best')
            
            def relabel_major(x, pos):
                    labels ={1.:'$\\alpha$', 1.5:'$\\beta$', 2.0:'$\\gamma$'}
                    if x in [1., 1.5, 2.0]:
                        return labels[x]
                    return ''
            ax.xaxis.set_major_formatter(FuncFormatter(relabel_major))
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            for label in ax.yaxis.get_majorticklabels():
                label.set_fontsize(24)
            
        file_name = os.path.join(FIGS_DIR, regime+ '_est_errors_params_only.png')
        get_current_fig_manager().window.showMaximized()
        print 'saving to ', file_name
        savefig(file_name)

        close()
        
    h5file.close()
    
def NM_SubT(table_file_name = 'GradedNM_SubTx16'):
    file_name = os.path.join(TABLES_DIR, table_file_name+ '.h5')
    h5file = openFile(file_name, mode = "r")

    Estimates = h5file.getNode('/', 'Estimates')
    Samples =  h5file.getNode('/', 'Samples')
    regimesTable =  h5file.getNode('/', 'Regimes')._f_getChild('Regimes')
    
    Methods = ['Initializer', 'Graded_Nelder-Mead', 'Fortet']
    regimeNames = ['subT'] 
    ylims = {'subT':2.0}
    xlims = {'subT':3.}
    plot_colours = {'Initializer':"g", 'Graded_Nelder-Mead':"y", 'Fortet': 'b'}
    for regime in regimeNames:
        samplesT = Samples._f_getChild(regime)
        estimatesT = Estimates._f_getChild(regime)
        samplesCount = len(samplesT)
        
        abg_true = [];
        for row in  regimesTable.where('name == "'+ regime+'"'):
            abg_true = [row['alpha'], row['beta'], row['gamma']]
                
        print regime, abg_true

        mpl.rcParams['figure.subplot.left'] = .175
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .125
        mpl.rcParams['figure.subplot.top'] = .95
        
        figure()
        hold(True)
        
        for mthd in Methods:
            abgs = empty((samplesCount, 3))
            idx= 0;
            for row in estimatesT.where('method == "'+ mthd +'"'):
                abgs[idx,:] = row['alpha'], row['beta'], row['gamma']
                idx +=1;
            
            diff_as = abgs [:,0] - abg_true[0];
            diff_bs = abgs [:,1] - abg_true[1];
            diff_gs = abgs [:,2] - abg_true[2];
             
            ax = subplot(211); hold (True)
            plot(1. + -.1+ .1*rand(len(diff_bs)), diff_as, linestyle='None',
                  color=  plot_colours[mthd], marker='o', markersize = 12, label = mthd)
            plot(1.5 + -.1+ .1*rand(len(diff_bs)), diff_bs, linestyle='None',
                  color=  plot_colours[mthd], marker='o', markersize = 12, label = None)
            plot(2.0 + -.1+ .1*rand(len(diff_bs)), diff_gs, linestyle='None',
                  color=  plot_colours[mthd], marker='o', markersize = 12, label = None)
            plot([1., 1.5, 2.0], zeros(3), 'b', linestyle='dashed', linewidth = 5)
            
            xlim((.75, 3.0)); ylim((-ylims[regime], ylims[regime]))
            ylabel('error', fontsize = 20)
            title('Estimate Errors for ' + regime + ' :$(\\alpha, \\beta, \\gamma) = (%.2g,%.2g,%.2g)$' %(abg_true[0], abg_true[1], abg_true[2]),
               fontsize = 16)
            legend(loc = 'best')
            
            def relabel_major(x, pos):
                    labels ={1.:'$\\alpha$', 1.5:'$\\beta$', 2.0:'$\\gamma$'}
                    if x in [1., 1.5, 2.0]:
                        return labels[x]
                    return ''
            ax.xaxis.set_major_formatter(FuncFormatter(relabel_major))
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            for label in ax.yaxis.get_majorticklabels():
                label.set_fontsize(24)
            
                
            ax = subplot(212); hold(True)
            l1_norm = abs(diff_as)+abs(diff_bs)+abs(diff_gs) 
            plot(l1_norm, -.1 + .1*rand(diff_as.size),
                 linestyle='None', color=plot_colours[mthd], marker='o', markersize = 12, label = mthd)
            ylim((-.5,.5)); xlim((-.01, xlims[regime]))
            setp(ax.get_yticklabels(), visible=False)
            plot(zeros(3),[-.5,.0,.5],  linestyle='dashed', linewidth = 5)
            xlabel('$l^1$ error', fontsize = 20);
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            
        file_name = os.path.join(FIGS_DIR, regime+'_refined_est_errors.png')
        get_current_fig_manager().window.showMaximized()
        print 'saving to ', file_name
        
        savefig(file_name)
         
        
    h5file.close()


def InitsComparison(table_file_name = 'inits_comparison'):
    file_name = os.path.join(TABLES_DIR, table_file_name+ '.h5')
    h5file = openFile(file_name, mode = "r")

    Estimates = h5file.getNode('/', 'Estimates')
    Samples =  h5file.getNode('/', 'Samples')
    regimesTable =  h5file.getNode('/', 'Regimes')._f_getChild('Regimes')
    
    Methods = ['Init5',  'Init_right_2std']  #'Init_right_1std',
    regimeNames = ['subT', 'crit', 'superSin', 'superT'] 
    ylims = {'subT':1.5, 'crit':.5, 'superSin':1.5, 'superT':.25}
    xlims = {'subT':2.5, 'crit':1.5, 'superSin':2.5, 'superT':.25}
    
    plot_colours = {'Init5':"g", 'Init_right_1std':"y", 'Init_right_2std':"r"}
    for regime in regimeNames:
        samplesT = Samples._f_getChild(regime)
        estimatesT = Estimates._f_getChild(regime)
        samplesCount = len(samplesT)
        
        abg_true = [];
        for row in  regimesTable.where('name == "'+ regime+'"'):
            abg_true = [row['alpha'], row['beta'], row['gamma']]
                
        print regime, abg_true

        mpl.rcParams['figure.subplot.left'] = .175
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .125
        mpl.rcParams['figure.subplot.top'] = .95
        
        figure()
        hold(True)
        
        for mthd in Methods:
            abgs = empty((samplesCount, 3))
            idx= 0;
            for row in estimatesT.where('method == "'+ mthd +'"'):
                abgs[idx,:] = row['alpha'], row['beta'], row['gamma']
                idx +=1;
            
            diff_as = abgs [:,0] - abg_true[0];
            diff_bs = abgs [:,1] - abg_true[1];
            diff_gs = abgs [:,2] - abg_true[2];
             
            ax = subplot(211); hold (True)
            plot(1. + -.1+ .1*rand(len(diff_bs)), diff_as, linestyle='None',
                  color=  plot_colours[mthd], marker='o', markersize = 12, label = mthd)
            plot(1.5 + -.1+ .1*rand(len(diff_bs)), diff_bs, linestyle='None',
                  color=  plot_colours[mthd], marker='o', markersize = 12, label = None)
            plot(2.0 + -.1+ .1*rand(len(diff_bs)), diff_gs, linestyle='None',
                  color=  plot_colours[mthd], marker='o', markersize = 12, label = None)
            plot([1., 1.5, 2.0], zeros(3), 'b', linestyle='dashed', linewidth = 5)
            
            xlim((.75, 3.0)); ylim((-ylims[regime], ylims[regime]))
            ylabel('error', fontsize = 20)
            title('Estimate Errors for ' + regime + ' :$(\\alpha, \\beta, \\gamma) = (%.2g,%.2g,%.2g)$' %(abg_true[0], abg_true[1], abg_true[2]),
               fontsize = 16)
            legend(loc = 'best')
            
            def relabel_major(x, pos):
                    labels ={1.:'$\\alpha$', 1.5:'$\\beta$', 2.0:'$\\gamma$'}
                    if x in [1., 1.5, 2.0]:
                        return labels[x]
                    return ''
            ax.xaxis.set_major_formatter(FuncFormatter(relabel_major))
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            for label in ax.yaxis.get_majorticklabels():
                label.set_fontsize(24)
            
                
            ax = subplot(212); hold(True)
            l1_norm = abs(diff_as)+abs(diff_bs)+abs(diff_gs) 
            plot(l1_norm, -.1 + .1*rand(diff_as.size),
                 linestyle='None', color=plot_colours[mthd], marker='o', markersize = 12, label = mthd)
            ylim((-.5,.5)); xlim((-.01, xlims[regime]))
            setp(ax.get_yticklabels(), visible=False)
            plot(zeros(3),[-.5,.0,.5],  linestyle='dashed', linewidth = 5)
            xlabel('$l^1$ error', fontsize = 20);
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            
            
        file_name = os.path.join(FIGS_DIR, regime+'_init_errors.png')
        get_current_fig_manager().window.showMaximized()
        print 'saving to ', file_name
        
        savefig(file_name)
        
    h5file.close()

        
#def timingInfo(table_file_name):
#    pass
##    file_name = os.path.join(TABLES_DIR, table_file_name+ '.h5')
##    h5file = openFile(file_name, mode = "r")
##
##    Estimates = h5file.getNode('/', 'Estimates')
##    Samples =  h5file.getNode('/', 'Samples')
##    regimesTable =  h5file.getNode('/', 'Regimes')._f_getChild('Regimes')
##    
##    Methods = ['Initializer', 'Nelder-Mead', 'BFGS', 'Fortet']
##    regimeNames = ['superT', 'subT', 'crit', 'superSin'] 
##    for mthd in Methods:
##        walltime = .0
##        for regime in regimeNames:
##            samplesT = Samples._f_getChild(regime)
##            estimatesT = Estimates._f_getChild(regime)
##            samplesCount = len(samplesT)
##            
##            abg_true = [];
##            for row in  regimesTable.where('name == "'+ regime+'"'):
##                abg_true = [row['alpha'], row['beta'], row['gamma']]
##                    
##            print regime, abg_true
##            
##            figure()
##            hold(True)
##            plot_colours = {'Fortet':"b", 'Initializer':"g", 'BFGS':"r",
##                             'Nelder-Mead':"y"}
##            abgs = empty((samplesCount, 3))
##            idx= 0;
##            for row in estimatesT.where('method == "'+ mthd +'"'):
##                abgs[idx,:] = row['alpha'], row['beta'], row['gamma']
##                idx +=1;
##        

def BFGS_Maxiters(table_file_name):
    file_name = os.path.join(TABLES_DIR, table_file_name+ '.h5')
    h5file = openFile(file_name, mode = "r")

    Estimates = h5file.getNode('/', 'Estimates')
    Samples =  h5file.getNode('/', 'Samples')
    regimesTable =  h5file.getNode('/', 'Regimes')._f_getChild('Regimes')
    
    regimeNames = ['crit', 'superSin'] 
    ylims = {'crit':.1, 'superSin':1.25}
    xlims = {'crit':.6, 'superSin':2.5}

    Methods = ['Initializer', 'BFGS_8', 'BFGS_16', 'BFGS_24']
    plot_colours = {'Initializer':"b", 'BFGS_8':'g', 'BFGS_16':'y', 'BFGS_24':'r'}
    for regime in regimeNames:
        samplesT = Samples._f_getChild(regime)
        estimatesT = Estimates._f_getChild(regime)
        samplesCount = len(samplesT)
        
        abg_true = [];
        for row in  regimesTable.where('name == "'+ regime+'"'):
            abg_true = [row['alpha'], row['beta'], row['gamma']]
                
        print regime, abg_true
        
        figure()
        hold(True)
        
        for mthd in Methods:
            abgs = empty((samplesCount, 3))
            idx= 0;
            for row in estimatesT.where('method == "'+ mthd +'"'):
                abgs[idx,:] = row['alpha'], row['beta'], row['gamma']
                idx +=1;
            
            diff_as = abgs [:,0] - abg_true[0];
            diff_bs = abgs [:,1] - abg_true[1];
            diff_gs = abgs [:,2] - abg_true[2];
             
            ax = subplot(211); hold (True)
            plot(1. + -.1+ .1*rand(len(diff_bs)), diff_as, linestyle='None',
                  color=  plot_colours[mthd], marker='o', markersize = 12, label = mthd)
            plot(1.5 + -.1+ .1*rand(len(diff_bs)), diff_bs, linestyle='None',
                  color=  plot_colours[mthd], marker='o', markersize = 12, label = None)
            plot(2.0 + -.1+ .1*rand(len(diff_bs)), diff_gs, linestyle='None',
                  color=  plot_colours[mthd], marker='o', markersize = 12, label = None)
            plot([1., 1.5, 2.0], zeros(3), 'b', linestyle='dashed', linewidth = 5)
            
            xlim((.75, 3.0)); ylim((-ylims[regime], ylims[regime]))
            ylabel('error', fontsize = 20)
            title('Estimate Errors for ' + regime + ' :$(\\alpha, \\beta, \\gamma) = (%.2g,%.2g,%.2g)$' %(abg_true[0], abg_true[1], abg_true[2]),
               fontsize = 16)
            legend(loc = 'best')
            
            def relabel_major(x, pos):
                    labels ={1.:'$\\alpha$', 1.5:'$\\beta$', 2.0:'$\\gamma$'}
                    if x in [1., 1.5, 2.0]:
                        return labels[x]
                    return ''
            ax.xaxis.set_major_formatter(FuncFormatter(relabel_major))
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            for label in ax.yaxis.get_majorticklabels():
                label.set_fontsize(24)
            
                
            ax = subplot(212); hold(True)
            l1_norm = abs(diff_as)+abs(diff_bs)+abs(diff_gs) 
            plot(l1_norm, -.1 + .1*rand(diff_as.size),
                 linestyle='None', color=plot_colours[mthd], marker='o', markersize = 12, label = mthd)
            ylim((-.5,.5)); xlim((-.01, xlims[regime]))
            setp(ax.get_yticklabels(), visible=False)
            plot(zeros(3),[-.5,.0,.5],  linestyle='dashed', linewidth = 5)
            xlabel('$l^1$ error', fontsize = 20);
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            
        file_name = os.path.join(FIGS_DIR, regime+'_est_errors_XMIN--.png')
        get_current_fig_manager().window.showMaximized()
        print 'saving to ', file_name
        
        savefig(file_name)         
        
    h5file.close()
    
    
def CvsPY(table_file_name='CvsPY_2x4'):
    file_name = os.path.join(TABLES_DIR, table_file_name+ '.h5')
    h5file = openFile(file_name, mode = "r")

    Estimates = h5file.getNode('/', 'Estimates')
    Samples =  h5file.getNode('/', 'Samples')
    regimesTable =  h5file.getNode('/', 'Regimes')._f_getChild('Regimes')
    
#    Methods = ['Initializer', 'FP-PY', 'BFGS', 'Fortet']
    Methods =  ['Initializer', 'FP-PY', 'FP-C']

    regimeNames = ['subT', 'superSin'] 
#    regimeNames = ['superT', 'subT', 'crit', 'superSin'] 
    ylims = {'superT':.1, 'subT':1.5, 'crit':.15, 'superSin':1.25}
#    xlims = {'superT':.5, 'subT':1.75, 'crit':.5, 'superSin':1.75}
#    rylims = {'superT':.5, 'subT':1.5, 'crit':.5, 'superSin':1.}
#    rxlims = {'superT':.5, 'subT':2.5, 'crit':1., 'superSin':1.5}
 
    labels = {'Initializer':'Initializer', 'FP-PY':"FP-Python", 'FP-C':'FP-C'}
    plot_colours = {'FP-C':"b", 'Initializer':"g", 'BFGS':"r",
                     'FP-PY':"y"}
    markers = {'FP-C':"d", 'Initializer':"*", 'BFGS':'h',
                     'FP-PY':"h"}
    
    for regime in regimeNames:
        samplesT = Samples._f_getChild(regime)
        estimatesT = Estimates._f_getChild(regime)
        samplesCount = len(samplesT)
        
        abg_true = [];
        for row in  regimesTable.where('name == "'+ regime+'"'):
            abg_true = [row['alpha'], row['beta'], row['gamma']]
                
        print regime, abg_true
        
        mpl.rcParams['figure.subplot.left'] = .175
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .125
        mpl.rcParams['figure.subplot.top'] = .95
    
        figure()
        hold(True)
        for mthd in Methods:
            abgs = empty((samplesCount, 3))
            idx= 0;
            for row in estimatesT.where('method == "'+ mthd +'"'):
                abgs[idx,:] = row['alpha'], row['beta'], row['gamma']
                idx +=1;
            
            diff_as = abgs [:,0] - abg_true[0];
            diff_bs = abgs [:,1] - abg_true[1];
            diff_gs = abgs [:,2] - abg_true[2];
             
            ax = subplot(111); hold (True)
            plot(1. + -.1+ .1*rand(len(diff_bs)), diff_as, linestyle='None',
                  color=  plot_colours[mthd], marker= markers[mthd], markersize = 12, label = labels[mthd])
            plot(1.5 + -.1+ .1*rand(len(diff_bs)), diff_bs, linestyle='None',
                  color=  plot_colours[mthd], marker=markers[mthd], markersize = 12, label = None)
            plot(2.0 + -.1+ .1*rand(len(diff_bs)), diff_gs, linestyle='None',
                  color=  plot_colours[mthd], marker=markers[mthd], markersize = 12, label = None)
            plot([1., 1.5, 2.0], zeros(3), 'b', linestyle='dashed', linewidth = 5)
            
            xlim((.75, 3.0)); ylim((-ylims[regime], ylims[regime]))
            ylabel('error', fontsize = 20)
            title('Estimate Errors for ' + regime + ' :$(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg_true[0], abg_true[1], abg_true[2]),
               fontsize = 18)
            legend(loc = 'best')
            
            def relabel_major(x, pos):
                    labels ={1.:'$\\alpha$', 1.5:'$\\beta$', 2.0:'$\\gamma$'}
                    if x in [1., 1.5, 2.0]:
                        return labels[x]
                    return ''
            ax.xaxis.set_major_formatter(FuncFormatter(relabel_major))
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            for label in ax.yaxis.get_majorticklabels():
                label.set_fontsize(24)
            
#        file_name = os.path.join(FIGS_DIR, regime+ '_est_errors_params_only.png')
#        get_current_fig_manager().window.showMaximized()
#        print 'saving to ', file_name
#        savefig(file_name)

#        close()
        
    h5file.close()


 
    
def FortetPostProcessor(table_file_name = 'FvsWF_4x16'):
    file_name = os.path.join(TABLES_DIR, table_file_name+ '.h5')
    h5file = openFile(file_name, mode = "r")

    Estimates = h5file.getNode('/', 'Estimates')
    Samples =  h5file.getNode('/', 'Samples')
    regimesTable =  h5file.getNode('/', 'Regimes')._f_getChild('Regimes')
    

    regimeNames = ['subT', 'superSin', 'superT', 'crit'] 
#    regimeNames = ['superT', 'subT', 'crit', 'superSin'] 
#    ylims = {'superT':.3, 'subT':.3, 'crit':.3, 'superSin':.3}
#    xlims = {'superT':.25, 'subT':1.5, 'crit':.25, 'superSin':.25}
    ylims = {'superT':.5, 'subT':.5, 'crit':.5, 'superSin':.5}
    xlims = {'superT':.3, 'subT':1.25, 'crit':.3, 'superSin':.3}
#    rylims = {'superT':.5, 'subT':1.5, 'crit':.5, 'superSin':1.}
#    rxlims = {'superT':.5, 'subT':2.5, 'crit':1., 'superSin':1.5}
 
#    Methods = ['Initializer', 'FP-PY', 'BFGS', 'Fortet']
    Methods =  ['FortetL2', 'FortetSup' ]

    labels = {'Initializer':'Initializer', 'FP-PY':"FP-Python", 'FP-C':'FP-C',
              'Fortet10':'Fortet_N=10','WeghtedFortet':'WeightedFortet',
              'Fortet64':'Fortet_N=64', 
              'Fortet':'Old Fortet', 'QuadFortet':'New Fortet',
              'FortetL2':'FortetL2', 'FortetSup' : 'FortetSup' }
    plot_colours = {'FP-C':"b", 'Initializer':"g", 'BFGS':"r",
                     'FP-PY':"y",
                     'Fortet10':'g','WeghtedFortet':'r', 'Fortet64':'b' ,
                     'Fortet':'b', 'QuadFortet':'r',
                     'FortetL2':'b', 'FortetSup':'r' }
    markers = {'FP-C':"d", 'Initializer':"*", 'BFGS':'h', 'FP-PY':"h",
               'Fortet10':'d','WeghtedFortet':'h', 'Fortet64':'*',
               'Fortet':'d', 'QuadFortet':'*',
                'FortetL2':'d', 'FortetSup':'*'
               }
    
    for regime in regimeNames:
        samplesT = Samples._f_getChild(regime)
        estimatesT = Estimates._f_getChild(regime)
        samplesCount = len(samplesT)
        
        abg_true = [];
        for row in  regimesTable.where('name == "'+ regime+'"'):
            abg_true = [row['alpha'], row['beta'], row['gamma']]
                
        print regime, abg_true
        
        mpl.rcParams['figure.subplot.left'] = .175
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .125
        mpl.rcParams['figure.subplot.top'] = .95
    
        figure()
        hold(True)
        for mthd in Methods:
            abgs = empty((samplesCount, 3))
            idx= 0;
            for row in estimatesT.where('method == "'+ mthd +'"'):
                abgs[idx,:] = row['alpha'], row['beta'], row['gamma']
                idx +=1;
            
            diff_as = abgs [:,0] - abg_true[0];
            diff_bs = abgs [:,1] - abg_true[1];
            diff_gs = abgs [:,2] - abg_true[2];
             
            ax = subplot(211); hold (True)
            plot(1. + -.1+ .1*rand(len(diff_bs)), diff_as, linestyle='None',
                  color=  plot_colours[mthd], marker= markers[mthd], markersize = 12, label = labels[mthd])
            plot(1.5 + -.1+ .1*rand(len(diff_bs)), diff_bs, linestyle='None',
                  color=  plot_colours[mthd], marker=markers[mthd], markersize = 12, label = None)
            plot(2.0 + -.1+ .1*rand(len(diff_bs)), diff_gs, linestyle='None',
                  color=  plot_colours[mthd], marker=markers[mthd], markersize = 12, label = None)
            plot([1., 1.5, 2.0], zeros(3), 'b', linestyle='dashed', linewidth = 5)
            
            xlim((.75, 3.0)); ylim((-ylims[regime], ylims[regime]))
            ylabel('error', fontsize = 20)
            title('Estimate Errors for ' + regime + ' :$(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg_true[0], abg_true[1], abg_true[2]),
               fontsize = 18)
            legend(loc = 'best')
            
            def relabel_major(x, pos):
                    labels ={1.:'$\\alpha$', 1.5:'$\\beta$', 2.0:'$\\gamma$'}
                    if x in [1., 1.5, 2.0]:
                        return labels[x]
                    return ''
            ax.xaxis.set_major_formatter(FuncFormatter(relabel_major))
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            for label in ax.yaxis.get_majorticklabels():
                label.set_fontsize(24)
                
            ax = subplot(212); hold(True)
            l1_norm = abs(diff_as)+abs(diff_bs)+abs(diff_gs) 
            #CAP IT:
            l1_norm = amin(c_[l1_norm, xlims[regime]*ones_like(l1_norm)], axis=1)
            plot(l1_norm, -.1 + .1*rand(diff_as.size),
                 linestyle='None', color=plot_colours[mthd], marker=markers[mthd], markersize = 12, label = mthd)
            ylim((-.25,.25)); xlim((-.01, xlims[regime]))
            setp(ax.get_yticklabels(), visible=False)
            plot(zeros(3),[-.5,.0,.5],  linestyle='dashed', linewidth = 5)
            xlabel('$l^1$ error', fontsize = 20);
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            
        file_name = os.path.join(FIGS_DIR, 'FortetPP_'+regime+ '_est_errors_abs.png')
        get_current_fig_manager().window.showMaximized()
        print 'saving to ', file_name
        savefig(file_name)
#        close()
    h5file.close()
    
    
    
    
def WeightedFortet_NBox(table_file_name = 'FvsWF_4x16'):
    file_name = os.path.join(TABLES_DIR, table_file_name+ '.h5')
    h5file = openFile(file_name, mode = "r")

    Estimates = h5file.getNode('/', 'Estimates')
    Samples =  h5file.getNode('/', 'Samples')
    regimesTable =  h5file.getNode('/', 'Regimes')._f_getChild('Regimes')
    
#    Methods = ['Initializer', 'FP-PY', 'BFGS', 'Fortet']
    Methods =  ['WeightedFortet_N1', 'WeghtedFortet']

    regimeNames = ['subT', 'superSin', 'superT', 'crit'] 
#    regimeNames = ['superT', 'subT', 'crit', 'superSin'] 
    ylims = {'superT':.3, 'subT':.3, 'crit':.3, 'superSin':.3}
    xlims = {'superT':.25, 'subT':1.5, 'crit':.25, 'superSin':.25}
#    rylims = {'superT':.5, 'subT':1.5, 'crit':.5, 'superSin':1.}
#    rxlims = {'superT':.5, 'subT':2.5, 'crit':1., 'superSin':1.5}
 
    labels = {'Initializer':'Initializer', 'FP-PY':"FP-Python", 'FP-C':'FP-C',
              'Fortet10':'Fortet_N=10','WeghtedFortet':'wFortet_Nth=10', 'Fortet64':'Fortet_N=64',
              'WeightedFortet_N1':'wFortet_Nth=1'}
    plot_colours = {'FP-C':"b", 'Initializer':"g", 'BFGS':"r",
                     'FP-PY':"y",
                     'Fortet10':'g','WeghtedFortet':'b', 'Fortet64':'b',
                     'WeightedFortet_N1':'r'}
    markers = {'FP-C':"d", 'Initializer':"*", 'BFGS':'h', 'FP-PY':"h",
               'Fortet10':'d','WeghtedFortet':'h', 'Fortet64':'*',
               'WeightedFortet_N1':'*'}
    
    for regime in regimeNames:
        samplesT = Samples._f_getChild(regime)
        estimatesT = Estimates._f_getChild(regime)
        samplesCount = len(samplesT)
        
        abg_true = [];
        for row in  regimesTable.where('name == "'+ regime+'"'):
            abg_true = [row['alpha'], row['beta'], row['gamma']]
                
        print regime, abg_true
        
        mpl.rcParams['figure.subplot.left'] = .175
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .125
        mpl.rcParams['figure.subplot.top'] = .95
    
        figure()
        hold(True)
        for mthd in Methods:
            abgs = empty((samplesCount, 3))
            idx= 0;
            for row in estimatesT.where('method == "'+ mthd +'"'):
                abgs[idx,:] = row['alpha'], row['beta'], row['gamma']
                idx +=1;
            
            diff_as = abgs [:,0] - abg_true[0];
            diff_bs = abgs [:,1] - abg_true[1];
            diff_gs = abgs [:,2] - abg_true[2];
             
            ax = subplot(211); hold (True)
            plot(1. + -.1+ .1*rand(len(diff_bs)), diff_as, linestyle='None',
                  color=  plot_colours[mthd], marker= markers[mthd], markersize = 12, label = labels[mthd])
            plot(1.5 + -.1+ .1*rand(len(diff_bs)), diff_bs, linestyle='None',
                  color=  plot_colours[mthd], marker=markers[mthd], markersize = 12, label = None)
            plot(2.0 + -.1+ .1*rand(len(diff_bs)), diff_gs, linestyle='None',
                  color=  plot_colours[mthd], marker=markers[mthd], markersize = 12, label = None)
            plot([1., 1.5, 2.0], zeros(3), 'b', linestyle='dashed', linewidth = 5)
            
            xlim((.75, 3.0)); ylim((-ylims[regime], ylims[regime]))
            ylabel('error', fontsize = 20)
            title('Estimate Errors for ' + regime + ' :$(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg_true[0], abg_true[1], abg_true[2]),
               fontsize = 18)
            legend(loc = 'best')
            
            def relabel_major(x, pos):
                    labels ={1.:'$\\alpha$', 1.5:'$\\beta$', 2.0:'$\\gamma$'}
                    if x in [1., 1.5, 2.0]:
                        return labels[x]
                    return ''
            ax.xaxis.set_major_formatter(FuncFormatter(relabel_major))
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            for label in ax.yaxis.get_majorticklabels():
                label.set_fontsize(24)
                
            ax = subplot(212); hold(True)
            l1_norm = abs(diff_as)+abs(diff_bs)+abs(diff_gs) 
            plot(l1_norm, -.1 + .1*rand(diff_as.size),
                 linestyle='None', color=plot_colours[mthd], marker=markers[mthd], markersize = 12, label = mthd)
            ylim((-.25,.25)); xlim((-.01, xlims[regime]))
            setp(ax.get_yticklabels(), visible=False)
            plot(zeros(3),[-.5,.0,.5],  linestyle='dashed', linewidth = 5)
            xlabel('$l^1$ error', fontsize = 20);
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            
        file_name = os.path.join(FIGS_DIR, 'WF_N_'+regime+ '_est_errors_abs.png')
        get_current_fig_manager().window.showMaximized()
        print 'saving to ', file_name
        savefig(file_name)
#        close()
    h5file.close()
    
    
    
################################
def drawSamples(fig, diff_as, diff_bs, diff_gs, abg_true,
                plot_colour, plot_marker, label, y_lim, x_lim):
    ax = fig.add_subplot(211); ax.hold (True)
    jitter = -.1+ .1*rand(len(diff_as));
    
    base_xs = [1., 1.5, 2.0]
    for diffs, base_x in zip([diff_as, diff_bs, diff_gs],
                             base_xs ):
        plot(base_x + jitter, diffs, linestyle='None',
          color= plot_colour , marker= plot_marker , markersize = 12, label = label)
    plot(base_xs, zeros(3), 'b', linestyle='dashed', linewidth = 5)
    
    xlim((.75, 3.0)); ylim((-y_lim, y_lim))
    ylabel('error', fontsize = 20)
    
    def relabel_major(x, pos):
            labels ={1.:'$\\alpha$', 1.5:'$\\beta$', 2.0:'$\\gamma$'}
            if x in base_xs:
                return labels[x]
            return ''
    ax.xaxis.set_major_formatter(FuncFormatter(relabel_major))
    for label in ax.xaxis.get_majorticklabels():
        label.set_fontsize(24)
    for label in ax.yaxis.get_majorticklabels():
        label.set_fontsize(24)
    
    #NOW PLOT_the L1Norm  
    ax = subplot(212); hold(True)
    l1_norm = abs(diff_as)+abs(diff_bs)+abs(diff_gs) 
    #CAP IT:
    l1_norm = amin(c_[l1_norm, x_lim*ones_like(l1_norm)], axis=1)
    
    plot(l1_norm, -.1 + .1*rand(l1_norm.size),
         linestyle='None', color= plot_colour , marker= plot_marker , markersize = 12, label = label)
    ylim((-.25,.25)); xlim((-.01, x_lim))
    setp(ax.get_yticklabels(), visible=False)
    plot(zeros(3),[-.5,.0,.5],  linestyle='dashed', linewidth = 5)
    xlabel('$l^1$ error', fontsize = 20);
    for label in ax.xaxis.get_majorticklabels():
        label.set_fontsize(24)
        #        close()

    

mthd_labels = {'FP':'FP',
               'MLE_nm':'ML',
               'MLE_nm32':'ML'}
    
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
            xlabel(mthd_labels[Method1],fontsize = xlabel_font_size); 
            ylabel(mthd_labels[Method2],fontsize = xlabel_font_size);
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


    
def crossAnalyzeSpecial(table_file_name,
                 fig_id = '', Method1 =  'Fortet', Method2 = 'FP',
                 regimeNames = None):
    '''This is b/c the data for the MLE in the SuperSin regime is truncated'''
    
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
        
        if 'superSin' == regime:
            abgs[Method1] = abgs[Method1][:75]
            abgs[Method2] = abgs[Method2][:75]
        
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
            xlabel(mthd_labels[Method1],fontsize = xlabel_font_size); 
            ylabel(mthd_labels[Method2],fontsize = xlabel_font_size);
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
    '''Show for both N=100, and N=1000 on the same plot'''
    
     
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

def compareTimes(table_names = ['FinalEstimate_4x100_N=100',
                                'FinalEstimate_4x100_N=1000'],
                 Methods = ['Fortet', 'FP', 'MLE_nm']):
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
            for mthd in Methods:
                walltimes = analyzer.getWallTimes(regime, mthd)
                mu, sigma = mean(walltimes), std(walltimes)
                print '%s & %.2f'%(mthd, mu), r'$\pm$ %.2f'%sigma
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
#    crossAnalyze('ThetaEstimate_4x100_N=1000', 
#                 fig_id='FP_vs_Fortet_thetas')
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

############################################################
############MLE ESTIMATES:
############################################################
    crossAnalyze('MLEEstimate_4x100_N=100',
                  fig_id='MLE_N100_32bins',
                  Method1 =  'MLE_nm32', Method2 = 'FP',
                  regimeNames = ['superT', 'superSin', 'crit', 'subT'])
    crossAnalyze('MLEEstimate_4x100_N=100',
                  fig_id='MLE_N100',
                  Method1 =  'MLE_nm', Method2 = 'FP',
                  regimeNames = ['superT', 'superSin', 'crit', 'subT'])
    crossAnalyzeSpecial('MLEEstimate_4x100_N=1000',
                  fig_id='MLE_N1000',
                  Method1 =  'MLE_nm', Method2 = 'FP',
                  regimeNames = ['superT', 'superSin', 'crit', 'subT'])
    
#    crossAnalyze('MLEEstimateGradOpts_4x100_N=100',
#                  fig_id='MLE_N100_bfgs',
#                  Method1 =  'MLE_bfgs', Method2 = 'FP')
#    crossAnalyze('MLEEstimateGradOpts_4x100_N=100',
#                  fig_id='MLE_N100_tnc',
#                  Method1 =  'MLE_tnc', Method2 = 'FP')
    
#    compareTimes(['MLEEstimate_4x100_N=100'],
#                   Methods=['FP', 'MLE_nm', 'MLE_nm32'])
    
    show() 
    