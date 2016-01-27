'''
Created on May 28, 2012

@author: alex
'''

from numpy import genfromtxt
import os
from PostProcessor import FIGS_DIR

if __name__ == "__main__":
    from pylab import *
    file_name = '../Results/Estimations/GradedNM_SubT.analysis'
    data = genfromtxt(file_name , unpack=True)
    
    data = data.transpose()
    print data.shape
#    data  = data[0:16,:]
    
    abg_true = [.4,.3,.4];
        
    mpl.rcParams['figure.subplot.left'] = .15
    mpl.rcParams['figure.subplot.right'] = .975
    mpl.rcParams['figure.subplot.bottom'] = .125
    mpl.rcParams['figure.subplot.top'] = .95
    mpl.rcParams['figure.subplot.hspace'] = .5
    mpl.rcParams['figure.subplot.wspace'] = .2
    
    figure()
    hold(True)
    Methods = ['T4', 'T8', 'T16', 'T32']
    plot_colours = {'T4':"g", 'T8':"y", 'T16':"r", 'T32':"k"}
    mthd_indices = {'T4':0, 'T8':1, 'T16':2, 'T32':3}
    
    PlotMethods = ['T8', 'T32']
    MethodLabels = ['4s', '8s', '16s', '32s']
    
    mean_errors = empty(len(Methods))
    median_errors =empty(len(Methods))
    for mthd in Methods:
        mthd_indx = mthd_indices[mthd]
        abgs = data[mthd_indx::4, :]
        
        diff_as = abgs [:,0] - abg_true[0];
        diff_bs = abgs [:,1] - abg_true[1];
        diff_gs = abgs [:,2] - abg_true[2];
        l1_norm = abs(diff_as)+abs(diff_bs)+abs(diff_gs)
         
        ax = subplot(3,1,1); hold (True)
        if mthd in PlotMethods:
            plot(1. + -.1+ .1*rand(len(diff_bs)), diff_as, linestyle='None',
                  color=  plot_colours[mthd], marker='o', markersize = 12, label = mthd)
            plot(1.5 + -.1+ .1*rand(len(diff_bs)), diff_bs, linestyle='None',
                  color=  plot_colours[mthd], marker='o', markersize = 12, label = None)
            plot(2.0 + -.1+ .1*rand(len(diff_bs)), diff_gs, linestyle='None',
                  color=  plot_colours[mthd], marker='o', markersize = 12, label = None)
            plot([1., 1.5, 2.0], zeros(3), 'b', linestyle='dashed', linewidth = 5)
            
            xlim((.75, 2.5)); 
            ylim((-.5, .5))
            ylabel('error', fontsize = 20)
            title('Estimate Errors : SubT : $(\\alpha, \\beta, \gamma) = (%.2g,%.2g,%.2g)$' %(abg_true[0], abg_true[1], abg_true[2]),
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
            
            # l1 errors:
            ax = subplot(3,2, 3); hold(True)
            plot(l1_norm, -.1 + .1*rand(diff_as.size),
                 linestyle='None', color=plot_colours[mthd], marker='o', markersize = 12, label = mthd)
            xlim((-.01, 1.))
            ylim((-.25,.25)); 
            setp(ax.get_yticklabels(), visible=False)
            plot(zeros(3),[-.5,.0,.5],  linestyle='dashed', linewidth = 5)
            title('$l^1$ error', fontsize = 20);
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(24)
            ax = subplot(3,2, 4); hold(True)
            plot(l1_norm, -.1 + .1*rand(diff_as.size),
                 linestyle='None', color=plot_colours[mthd], marker='o', markersize = 12, label = mthd)
#            xlim((-.01, 1.5))
            ylim((-.25,.25)); 
            setp(ax.get_yticklabels(), visible=False)
            plot(zeros(3),[-.5,.0,.5],  linestyle='dashed', linewidth = 5)
            title('$l^1$ error', fontsize = 20);
            for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(24)
        
        mean_errors[mthd_indx] = mean(l1_norm)
        median_errors[mthd_indx] = median(l1_norm) 
            
        #mean errors:
        ax = subplot(3,2,5);
        inds = arange(len(Methods)); half_width = .16
        bar(inds, mean_errors, 2*half_width);
        title('Mean Error', fontsize = 20)
        xticks(inds+half_width, MethodLabels, fontsize = 24)
        ax = subplot(3,2,6);
        bar(inds, median_errors, 2*half_width)
        title('Median Error', fontsize = 20)
        xticks(inds+half_width, MethodLabels , fontsize = 24)
    
    file_name = os.path.join(FIGS_DIR, 'SubT_errors_gradedFP.png')
    get_current_fig_manager().window.showMaximized()
    print 'saving to ', file_name
    
    savefig(file_name)

    show()