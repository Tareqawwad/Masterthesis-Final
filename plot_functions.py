# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 03:06:04 2022

@author: Sven Mordeja
"""
#%% imports
import help_functions
#import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from multiprocessing import Process, freeze_support
from data_manager import DataManager
from doe import DOE
from bays_model import BaysModel
from WoehlerParams import WoehlerCurve
from arviz import kde
from scipy.stats import gaussian_kde

'''
In this script several plotting functions are saved.

'''

#%% constants
PARAM_LIST = ['k','s_d', 'n_e','one_t_s']
VAR_LIST = ['s_a', 'n', 'outcome']
SMALL_SIZE = 16
MEDIUM_SIZE = 24
BIGGER_SIZE = 30
EVEN_BIGGER = 36

#%% settings for plots
# plt.rc('font', size=EVEN_BIGGER)          # controls default text sizes
# plt.rc('axes', titlesize=EVEN_BIGGER)     # fontsize of the axes title
# plt.rc('axes', labelsize=EVEN_BIGGER)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 300

#%% plotting functions
def plot_woehler(ax, woehler_points, color1='navy', label1='', markersize1 = None):
    '''
    Plots the experimental results of a wohler experiment.
 
    Parameters
    ----------
    ax : Axes
        The axes that are used for the plot.
    woehler_points : Dict
        A dictonary containing the measured data points. 
        (s_a, n, outcome)
    color1 : Sring, optional
        Color of the plot. The default is 'navy'.
    label1 : String, optional
        The label of the plot. The default is ''.
    markersize1 : Int, optional
        The markersize for the plot. The default is None.

    Returns
    -------
    None.

    '''
    ax.set_xlabel('$N$')
    ax.set_ylabel('$L_a~[MPa]$')
        
    s_a_failure, n_failure, s_a_runout, n_runout = help_functions.get_runout_and_failure(woehler_points)
    
    plt.loglog(n_failure, s_a_failure, linewidth = 0, markersize = markersize1, 
               marker='o',color=color1, label = label1 + ' Ausfall')
    
    plt.loglog(n_runout, s_a_runout, linewidth = 0, markersize=markersize1, 
               marker="^",color=color1, fillstyle = 'none',
               label =label1 + ' Durchläufer')
    ax.legend()
    
def plot_model(ax, woehler_points, woehler_params, label_1 = '', color_1 = 'navy'):
    '''
    PLots the woehler model lines (P_a = 10%, 50% and 90%). The woehler points are used to set the axes.

    Parameters
    ----------
    ax : Axes
        The axes that are used for the plot.
    woehler_points : Dict
        A dictonary containing the measured data points. 
        (s_a, n, outcome)
    woehler_params : Dict
        A dictionary containing the four woehler parameter.
    label_1 : String, optional
        The label of the plot. The default is ''.
    color_1 : Sring, optional
        Color of the plot. The default is 'navy'.

    Returns
    -------
    None.

    '''
    s_a = woehler_points['s_a']
    n = woehler_points['n']
    outcome = woehler_points['outcome']
    k = woehler_params['k']
    s_d_50 = woehler_params['s_d']
    n_e     = woehler_params['n_e']
    one_t_s = woehler_params['one_t_s']
    if one_t_s  != None and s_d_50 != None:
        if s_d_50 > s_a.max():
            s_d_max = s_d_50 + 1
            
        else:
            s_d_max = s_a.max()
        
        if n_e > n.max():
            n_max = n_e + 1111
        else:
            n_max = n.max()
        
        s_d_10 = s_d_50 / (10**(-stats.norm.ppf(0.1)*np.log10(one_t_s)/2.56))
        s_d_90 = s_d_50 / (10**(-stats.norm.ppf(0.9)*np.log10(one_t_s)/2.56))
        
        
        amp_s_d_50_1    = np.linspace(s_d_50, s_d_max, 10000)
        n_s_d_50_1      = 10**(np.log10(n_e)+k*(np.log10(s_d_50)-np.log10(amp_s_d_50_1)))
        n_s_d_50_2      = np.linspace(n_e, n_max, 10000)
        amp_s_d_50_2    = n_s_d_50_2 * 0 + s_d_50


        amp_s_d_10_1    = np.linspace(s_d_10, s_d_max, 10000)
        n_s_d_10_1      = 10**(np.log10(n_e)+k*(np.log10(s_d_10)-np.log10(amp_s_d_10_1)))
        amp_s_d_10_2    = n_s_d_50_2 * 0 + s_d_10


        amp_s_d_90_1    = np.linspace(s_d_90,s_d_max, 10000)
        n_s_d_90_1      = 10**(np.log10(n_e)+k*(np.log10(s_d_90)-np.log10(amp_s_d_90_1)))
        amp_s_d_90_2    = n_s_d_50_2 * 0 + s_d_90
        
        plt.loglog(n_s_d_50_1,amp_s_d_50_1,linewidth=3,color= color_1, label = label_1 + ' $N_{50\%}$')
        plt.loglog(n_s_d_50_2,amp_s_d_50_2,linewidth=3,color= color_1)
        
        plt.loglog(n_s_d_10_1,amp_s_d_10_1,linewidth=3,color= color_1, linestyle='dashed', label = label_1 + ' $N_{10\%}$ u. $N_{90\%}$')
        plt.loglog(n_s_d_50_2,amp_s_d_10_2,linewidth=3,color= color_1, linestyle='dashed')
        
        plt.loglog(n_s_d_90_1,amp_s_d_90_1,linewidth=3,color= color_1, linestyle='dashed')
        plt.loglog(n_s_d_50_2,amp_s_d_90_2,linewidth=3,color= color_1, linestyle='dashed')
        ax.legend()
    

def plot_many_woehler( trace, curve):
    '''
    Plots 1000 woehler models. It is a depiction of a trace

    Parameters
    ----------
    trace : MultiTrace
        The MultiTrace object containing the samples from
        the posterior. The samples for a parameter can be 
        accessed similar to a dictionary by giving the parameter
        name as a key: trace['n_e']
    curve : Curve
        This is a curve object as returned by the WoehlerParams
        class. It contains the data points and the Max-Likeli-
        Parmametrs..

    Returns
    -------
    None.

    '''
    new_verify = BaysModel()
    fig = plt.figure(figsize=(6.62 , 5.144))
    
    s_a  = curve.data['loads']
    n  = curve.data['cycles']
    outcome  = curve.data['outcome']
    ax = plt.gca()
    for ii in range(1000):
        woehler_params = {}
        woehler_params['k'] = trace['k'][ii]
        woehler_params['s_d'] = trace['s_d'][ii]
        woehler_params['n_e'] = trace['n_e'][ii]
        woehler_params['one_t_s'] = trace['one_t_s'][ii]
        plot_model_for_many_woehler(ax, s_a, n, woehler_params,
                                   label_1 = '', color_1 = 'black')
    plt.savefig('many_woe.png',dpi=800)
    ax.set_xlim([10**4,3*10**7])
    ax.set_ylim([400,504])
    plt.show()
    
def plot_model_for_many_woehler(ax, s_a, n, woehler_params, label_1 = '', color_1 = 'navy'):
    '''
    This function is specificly programmed for "plot_many_woehler". It plots  a very thin wohler curve.

    Parameters
    ----------
    ax : Axes
        The axes that are used for the plot.
    s_a : Array
        amplitudes.
    n : Array
        Number of cycles.
    woehler_params : Dict
        Diectionary of wohler parameters.
    label_1 : String, optional
        Lable of plot. The default is ''.
    color_1 : TYPE, optional
        Color of plot. The default is 'navy'.

    Returns
    -------
    None.

    '''
    k = woehler_params['k']
    s_d_50 = woehler_params['s_d']
    n_e     = woehler_params['n_e']
    one_t_s = woehler_params['one_t_s']
    if one_t_s  != None and s_d_50 != None:
        if s_d_50 > s_a.max():
            s_d_max = s_d_50 + 1
            
        else:
            s_d_max = s_a.max()
        
        if n_e > n.max():
            n_max = n_e + 1111
        else:
            n_max = n.max()
        
        s_d_10 = s_d_50 / (10**(-stats.norm.ppf(0.1)*np.log10(one_t_s)/2.56))
        s_d_90 = s_d_50 / (10**(-stats.norm.ppf(0.9)*np.log10(one_t_s)/2.56))
        
        
        amp_s_d_50_1    = np.linspace(s_d_50, s_d_max, 10000)
        n_s_d_50_1      = 10**(np.log10(n_e)+k*(np.log10(s_d_50)-np.log10(amp_s_d_50_1)))
        n_s_d_50_2      = np.linspace(n_e, n_max, 10000)
        amp_s_d_50_2    = n_s_d_50_2 * 0 + s_d_50
    
    
        amp_s_d_10_1    = np.linspace(s_d_10, s_d_max, 10000)
        n_s_d_10_1      = 10**(np.log10(n_e)+k*(np.log10(s_d_10)-np.log10(amp_s_d_10_1)))
        amp_s_d_10_2    = n_s_d_50_2 * 0 + s_d_10
    
    
        amp_s_d_90_1    = np.linspace(s_d_90,s_d_max, 10000)
        n_s_d_90_1      = 10**(np.log10(n_e)+k*(np.log10(s_d_90)-np.log10(amp_s_d_90_1)))
        amp_s_d_90_2    = n_s_d_50_2 * 0 + s_d_90
        alpha=0.3
        plt.loglog(n_s_d_50_1,amp_s_d_50_1,linewidth=0.1,
                   color= color_1, alpha=alpha)
        plt.loglog(n_s_d_50_2,amp_s_d_50_2,linewidth=0.1,
                   color= color_1, alpha=alpha)
        
        plt.loglog(n_s_d_10_1,amp_s_d_10_1,linewidth=0.1,
                   color= 'navy', linestyle='dashed', alpha=alpha)
        plt.loglog(n_s_d_50_2,amp_s_d_10_2,linewidth=0.1,
                   color= 'navy', linestyle='dashed', alpha=alpha)
        
        plt.loglog(n_s_d_90_1,amp_s_d_90_1,linewidth=0.1,
                   color= 'navy', linestyle='dashed', alpha=alpha)
        plt.loglog(n_s_d_50_2,amp_s_d_90_2,linewidth=0.1,
                   color= 'navy', linestyle='dashed', alpha=alpha)
    
def plot_scatter(trace):
    '''
    This function plots a scatter of all paraments.

    Parameters
    ----------
    trace : MultiTrace
        The MultiTrace object containing the samples from
        the posterior. The samples for a parameter can be 
        accessed similar to a dictionary by giving the parameter
        name as a key: trace['n_e']

    Returns
    -------
    None.

    '''
    
    xy = np.vstack([np.array(trace['one_t_s']),np.array(trace['k'])])
    z = gaussian_kde(xy)(xy)
    xy = np.vstack([np.array(trace['one_t_s']),np.array(trace['s_d'])])
    z2 = gaussian_kde(xy)(xy)
    df = pd.DataFrame({'$T_{L}$':np.array(trace['one_t_s']),
                   '$k$':np.array(trace['k']),
                   '$L_{a,L} \: [MPa]$':np.array(trace['s_d']),
                   '$N_K$':np.array(trace['n_e'])
                   })
    fig = plt.figure(figsize = (10, 10))
    ax=plt.gca()
    pd.plotting.scatter_matrix(df,  ax = ax, diagonal='kde')
    plt.show()  
    
def plot_many_samples(trace, curve):
    '''
    Plots sever fictive samples on many load levels to visualise the trace.

    Parameters
    ----------
    trace : MultiTrace
        The MultiTrace object containing the samples from
        the posterior. The samples for a parameter can be 
        accessed similar to a dictionary by giving the parameter
        name as a key: trace['n_e']
    curve : Curve
        This is a curve object as created by the WoehlerParams
        class. It contains the data points and the Max-Likeli-
        Parmametrs.

    Returns
    -------
    None.

    '''
    fig = plt.figure(figsize = (10, 10))
    new_doe = DOE()
    load_levels = np.linspace(400,520,60)
    new_verify = BaysModel()
    woehler_params_likeli = new_verify.transform_woehler_params(curve.Mali_4p_result)
    
    samples, samples_load_level = \
        new_doe.generate_distibutions_load_level( trace, curve, load_levels)
    
    s_a = samples['loads']
    n = samples['cycles']
    outcome = samples['outcome']
    ax=plt.gca()
    plot_woehler(ax, s_a, n, outcome, color1 = 'navy',label1='Posterior', markersize1 = 8)
    samples, samples_load_level = \
        new_doe.generate_distibutions_load_level_2( woehler_params_likeli, load_levels)
    
    s_a = samples['loads']
    n = samples['cycles']
    outcome = samples['outcome']
    ax=plt.gca()
    plot_woehler(ax, s_a, n, outcome, color1 = 'green',label1='ML', markersize1 = 8)
    plt.savefig('many_samp14.png',dpi=400)
    plt.show()

def plot_many_posterior(index):
    '''
    This is a visualisiation used for several posterior distributions.

    Parameters
    ----------
    index : Integer
        Index of a dataset (colum number) eg. 0, 1, 2,... .

    Returns
    -------
    None.

    '''
    fig, axes, gs =  create_fig_axes()
    xlim ={}
    linestyle = [':','-.', '--','-']
    num = 0
    new_bays = BaysModel()
    for ii in [0,4,8,12]:
        trace, curve, maxp = new_bays.load_trace(ii, index)
        for param in ['k','s_d', 'n_e','one_t_s']:
            smin, smax = np.min(trace[param]), np.max(trace[param])
            x = np.linspace(smin, smax, 1000)
            y = stats.gaussian_kde(trace[param])(x)
            grid, pdf = kde(trace[param])
            axes[param].plot(grid, pdf,label = str(ii+1)+' Proben', linewidth=3,
                             linestyle = linestyle[num], color ='navy')           
            xlim[param] = axes[param].get_xlim()
        num=num+1   
    set_axes(axes)
    plot_prior(axes, index)
    for param in PARAM_LIST:
        axes[param].set_xlim((xlim[param]))
        axes[param].legend(prop={'size': 12})
        
def set_axes(axes):
    '''
    This functions sets the ax-labels for four specified axes.

    Parameters
    ----------
    axes : Dict
        A dictionary containing four axes.

    Returns
    -------
    None.

    '''
    for param in PARAM_LIST:
        ax = axes[param]
        if param == 's_d':
            #ax.title.set_text('$L_{a,L}$')
            ax.set_xlabel('$L_{a,L} \: [MPa]$')
            ax.set_ylabel('$p(L_{a,L})$')
        elif param == 'k':
            #ax.title.set_text('$k$')
            ax.set_xlabel('$k$')
            ax.set_ylabel('$p(k)$')
        elif param == 'n_e':
            #ax.title.set_text('$N_K$')
            ax.set_xlabel('$N_K$')
            ax.set_ylabel('$p(N_K)$')
        elif param == 'one_t_s':
            #ax.title.set_text('$T_{L_a}$')
            ax.set_xlabel('$T_{L}$')
            ax.set_ylabel('$p(T_{L})$')
        ax.legend( fontsize=SMALL_SIZE)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        
def create_fig_axes(num = 4):
    '''
    This function creates four axes in a sorted in a gridsec. Since there are four parameters in the model this methode is useful.

    Parameters
    ----------
    num : Int, optional
        The number of axes to be created. The default is 4.

    Returns
    -------
    fig : Figure
        The figure in which the plot is created.
    axes : Dict
        A dictionary containing four axes.
    gs : GridSpec
        The gridspec used for the plots.

    '''
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(nrows = 2, ncols = 2)
    axes = {}
    axes['k'] = fig.add_subplot(gs[0, 0])
    axes['s_d'] = fig.add_subplot(gs[0, 1])
    axes['n_e'] = fig.add_subplot(gs[1, 0])
    axes['one_t_s'] = fig.add_subplot(gs[1, 1])
    gs.update(wspace = 0.28, hspace = 0.32)
    return fig, axes, gs       

def plot_prior(axes, index):
    '''
    PLots the prior distribution of a specified data set. The index is used to specify the dataset.

    Parameters
    ----------
    axes : Axes
        Axes of the plot.
    index : Integer
        Index of a dataset (colum number) eg. 0, 1, 2,... ..

    Returns
    -------
    None.

    '''
    
    new_data_manager = DataManager()
    new_verify = BaysModel()
    new_doe = DOE()
    new_data_manager.load_ki_predictions_from_csv_new(filename = 'ml-prediction.csv', delimiter = ',')
    prior_from_ki = new_data_manager.get_ki_predictions_by_index(index)
    for param in PARAM_LIST:
        ax = axes[param]
        mu = prior_from_ki[param]
        sd = np.sqrt(prior_from_ki[param + '_std'])
        plot_normal(ax, mu, sd,color_1 = 'red', label_1 = 'Prior')
        if param == 's_d':
            #ax.title.set_text('$L_{a,L}$')
            ax.set_xlabel('$L_{a,L} \: [MPa]$')
            ax.set_ylabel('$p(L_{a,L})$')
        elif param == 'k':
            #ax.title.set_text('$k$')
            ax.set_xlabel('$k$')
            ax.set_ylabel('$p(k)$')
        elif param == 'n_e':
            #ax.title.set_text('$N_K$')
            ax.set_xlabel('$N_K$')
            ax.set_ylabel('$p(N_K)$')
        elif param == 'one_t_s':
            #ax.title.set_text('$T_{L_a}$')
            ax.set_xlabel('$T_{L_a}$')
            ax.set_ylabel('$p(T_{L_a})$')
        ax.legend( fontsize=SMALL_SIZE)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        
def plot_nmse(all_nmses, nmses, load_levels, params_not_enough):
    '''
    Plots the NMSEs 

    Parameters
    ----------
    all_nmses :  Array
        A one dimensional np array containing the NMSE values computed for the load_levels.
    nmses : Dict
        A dictionary containing the NMSE values of the four parameters.
    params_not_enough : Array
        The parameters that are not well enough determined by the experimental data.

    Returns
    -------
    None.

    '''
    fig, axes, gs = create_fig_axes(num=4)
    for param in PARAM_LIST:
        # if the params that are well enough determined should be marked the following code can be used
        # if param in params_not_enough:
        #     color = 'red'
        #     label = 'Abbruchbedingung nicht erreicht'
        # else:
        #     color = 'green'
        #     label = 'Abbruchbedingung erreicht'
        axes[param].plot(load_levels, nmses[param], color = 'navy',linewidth = 3)
        axes[param].set_xlabel('$L_{a}$')
        axes[param].set_ylabel('$NMSE$')
        if param == 's_d':
            #if the title should be set
            #ax.title.set_text('$L_{a,L}$')
            axes[param].set_title('$L_{a,L}$')
          
        elif param == 'k':
            #ax.title.set_text('$k$')
            axes[param].set_title('$k$')
            
        elif param == 'n_e':
            #ax.title.set_text('$N_K$')
            axes[param].set_title('$N_K$')
        
        elif param == 'one_t_s':
            #ax.title.set_text('$T_{L_a}$')
            axes[param].set_title('$T_{L_a}$')
        axes[param].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.show() 
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
    ax.plot(load_levels, all_nmses, color = 'navy', linewidth =3)
    ax.set_title('$NMSE_{ges}$')
    ax.set_xlabel('$L_a$')
    ax.set_ylabel('$NMSE_{ges}$')
    plt.show()

def plot_k(all_t_prior, all_t_no_prior, mini, maxi):
    '''
    Plot function for calc_t_standard from the t.py script. It is a plot to compare 
    two fuctions of t over the number of experiments.

    Parameters
    ----------
    all_t_prior : Array
        All Ts from the experiment planning with prior.
    all_t_no_prior : Array
        All Ts from the experiment planning without prior.
        DESCRIPTION.
    mini : Int
        Min number of experiments.
    maxi : Int
        Max number of experiments.

    Returns
    -------
    None.

    '''
    all_t_prior_new = {}
    all_t_no_prior_new = {}
    for ii in range(mini, maxi+1):
        for param in PARAM_LIST:
            all_t_prior_new[param] = []
            all_t_no_prior_new[param] = []
    for ii in range(mini, maxi+1):
        for param in PARAM_LIST:
            all_t_prior_new[param] = np.append(all_t_prior_new[param],
                                               all_t_prior[ii][param])
            all_t_no_prior_new[param] = np.append(all_t_no_prior_new[param],
                                                  all_t_no_prior[ii][param])
    all_t_no_prior = all_t_no_prior_new
    all_t_prior = all_t_prior_new
    length  = len(all_t_no_prior_new['k'])
    for jj in range(length):
        n_s=range(mini, mini+jj+1)
        fig = plt.figure(figsize=(10, 10))
        #
        gs = GridSpec(nrows=4, ncols=4)
        ax0 = plt.subplot2grid((4, 1), (0, 0), colspan=4, rowspan=1)
        ax0.set_xticks([])
        ax1 = plt.subplot2grid((4, 1), (1, 0), colspan=4, rowspan=1)
        ax1.set_xticks([])
        ax2 = plt.subplot2grid((4, 1), (2, 0), colspan=4, rowspan=1)  
        ax2.set_xticks([])
        ax3 = plt.subplot2grid((4, 1), (3, 0), colspan=4, rowspan=1)
        axes = {}
        axes['k'] = ax0
        axes['s_d'] = ax1
        axes['n_e'] = ax2
        axes['one_t_s'] = ax3
        axes['one_t_s'].set_xlabel('Probenanzahl')
        for param in PARAM_LIST:
            axes[param].plot(n_s,all_t_no_prior_new[param][range(jj+1)], 
                             label = 'Standard', linewidth=4, color='tab:red')
            axes[param].plot(n_s,all_t_prior_new[param][range(jj+1)],
                             label = 'ML & Bayes', linewidth=4, color='navy')
            goal  = all_t_no_prior[param][-1]
            axes[param].axhline(y = goal, color ='green', linestyle = '--', label = '',  linewidth=2)
            axes[param].set_xlim([3,4+length])
            ymin =min(np.append(all_t_prior[param], all_t_no_prior[param]))
            ymax = ymin + (abs(all_t_prior_new[param][-1]-all_t_no_prior_new[param][-1])*10)
            if max(np.append(all_t_no_prior[param], all_t_prior[param])) <ymax:
                ymax=max(np.append(all_t_no_prior[param], all_t_prior[param]))
            ymin = ymin - (ymax -ymin)*0.05
            axes[param].set_ylim([ymin, ymax])
            if param == 'k':
                ylabel = '$T$ for $'+param +'$'
            elif param == 'n_e':
                ylabel = '$T$ for $N_{K}$'
            elif param == 's_d':
                ylabel = '$T$ for $L_{a,L}$'
            elif param == 'one_t_s':
                ylabel = '$T$ for $T_{L}$'    
            axes[param].set_ylabel(ylabel)
            axes[param].legend( loc = 'upper right')
    
def plot_prices(prices, all_nmses, load_levels,maxX=False, maxY=True):
    '''
    Plots a pareto frontier with the prices and the NMSEs.
    
    Source: https://sirinnes.wordpress.com/2013/04/25/pareto-frontier-graphic-via-python/
    and https://code.activestate.com/recipes/578230-pareto-front/

    Parameters
    ----------
    prices : Array
        The expectet prices for the load levels.
    all_nmses : Array
        The NMSEs for all params for the load levels.
    load_levels : Array
        A one dimensional np array of the empircally determined,
        viable load levels.
    maxX : Boolean, optional
        Determines if a high or low x value is better. The default is False.
    maxY : Boolean, optional
        Determines if a high or low y value is better. The default is True.

    Returns
    -------
    None.

    '''
    fig = plt.figure(figsize=(10, 10))
    Xs = prices
    Ys = all_nmses
    
    #Pareto frontier selection process 
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    
    #Plotting
    plt.scatter(Xs,Ys, marker = "P", s=400, label ='Lastniveau', color='tab:red')
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    plt.plot(pf_X, pf_Y, linewidth = 4, label ='Paretofront', color ='navy')
    plt.xlabel("Preis [€]")
    plt.ylabel("NMSE")
    for i, txt in enumerate(load_levels):
        # if txt > 418:
            if Xs[i] in pf_X:
                txt = np.round(txt)
                txt = str(int(txt)) + '$~$MPa'
                plt.annotate(txt, (Xs[i]+12, Ys[i]))
            #plt.text(Xs[i], Ys[i],txt)
        
    ax = plt.gca()
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_xlim([200, 1000])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
    plt.show()
    
def plot_uniform(lower, upper, color_1= 'red', label_1 = 'Prior'):
    '''
    Plots a uniform distribution.

    Parameters
    ----------
    lower : Int
        The lower bound of the uniform distribution.
    upper : Int
        The upper bound of the uniform distribution.
    color_1 : String, optional
        The color of the plot. The default is 'red'.
    label_1 : String, optional
        The label of the plot. The default is 'Prior'.

    Returns
    -------
    None.

    '''
    x = np.linspace(lower - 0.01, upper + 0.01, 500)
    [l,u] = [lower, upper]
    y = np.zeros(500)
    y[(x<u) & (x>l)] = 1.0/(u-l)
    plt.plot(x, y, color= color_1 , label = label_1)
    
def plot_prior(axes, prior_from_ki,color,label,linestyle='solid'):
    '''
    Plots the prior distribution of a specified data set. The index is used to specify the dataset.
    
    Parameters
    ----------
    prior_from_ki: String
        priors from the AI
    axes : Axes
        Axes of the plot.
    index : Integer
        Index of a dataset (colum number) eg. 0, 1, 2,... ..
    
    Returns
    -------
    None.
    
    '''
    for param in PARAM_LIST:
        ax = axes[param]
        mu = prior_from_ki[param]
        sd = prior_from_ki[param + '_std']
        plot_normal(ax, mu, sd, color_1 = color, label_1 = label,linestyle_1=linestyle)
        if param == 'k':
            #ax.title.set_text('$k$')
            ax.set_xlabel('$k$')
            ax.set_ylabel('$p(k)$')
        elif param == 's_d':
            #ax.title.set_text('$L_{a,L}$')
            ax.set_xlabel('$L_{a,L} \: [MPa]$')
            ax.set_ylabel('$p(L_{a,L})$')
        elif param == 'n_e':
            #ax.title.set_text('$N_K$')
            ax.set_xlabel('$N_K$')
            ax.set_ylabel('$p(N_K)$')
        elif param == 'one_t_s':
            #ax.title.set_text('$T_{L_a}$')
            ax.set_xlabel('$T_{L_a}$')
            ax.set_ylabel('$p(T_{L_a})$')
        ax.legend( fontsize=SMALL_SIZE)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        
def plot_normal(ax, mu, sd, color_1 = 'red', label_1 = 'Prior',linestyle='solid'):
    '''
    Plots a normal distribution.

    Parameters
    ----------
    mu : Int
        The mean of the normal distribution.
    sd : Int
        Stadard deviation of the distribution.
    color_1 : Sring, optional
        The color of the line. The default is 'red'.
    label_1 : Sting, optional
        The label of the plot. The default is 'Prior'.
    ax : Axes, optional
        The axes for the plot. The default is plt.gca().

    Returns
    -------
    None.

    '''
    x = np.linspace(mu - sd * 3, mu + sd * 3, 500)
    ax.plot(x, stats.norm.pdf(x,mu,sd),color= color_1 , label = label_1)    
    
    
if __name__ == '__main__':
    __spec__ = None
    freeze_support()