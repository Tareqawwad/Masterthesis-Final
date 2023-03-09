# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 03:06:04 2022

@author: Sven Mordeja modified by Tareq Awwad
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
from data_manager_OG import DataManager
from doe import DOE
from bays_model import BaysModel
from WoehlerParams import WoehlerCurve
from arviz import kde
from scipy.stats import gaussian_kde
from collections import defaultdict

'''
In this script several plotting functions are saved.

'''

#%% constants
PARAM_LIST = ['k','s_d', 'n_e','one_t_s']
VAR_LIST = ['s_a', 'n', 'outcome']
SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 24
EVEN_BIGGER = 30

def plot_prior(axes, prior_from_ki,woehler_params_original,color,label,linestyle='solid'):
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
        OG = woehler_params_original[param]
        plot_normal(ax, mu, sd, OG, color_1 = color, label_1 = label,linestyle_1=linestyle)
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
        
def plot_normal(ax, mu, sd, OG, color_1 = 'red', label_1 = 'Prior', linestyle_1 = 'dashed'):
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
    ax.plot(x, stats.norm.pdf(x,mu,sd),color= color_1 , label = label_1, linestyle = linestyle_1)
    ax.axvline(OG, color='k', linestyle=linestyle_1, linewidth=1)
    
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
        fig = plt.figure(figsize=(20, 20))
        #
        gs = GridSpec(nrows=4, ncols=4)
        ax0 = plt.subplot2grid((8, 1), (0, 0), colspan=4, rowspan=2)
        ax0.set_xticks([])
        ax1 = plt.subplot2grid((8, 1), (2, 0), colspan=4, rowspan=2)
        ax1.set_xticks([])
        ax2 = plt.subplot2grid((8, 1), (4, 0), colspan=4, rowspan=2)  
        ax2.set_xticks([])
        ax3 = plt.subplot2grid((8, 1), (6, 0), colspan=4, rowspan=2)
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
            axes[param].set_xlim([1,4+length])
            ymin = min(np.append(all_t_prior[param], all_t_no_prior[param]))
            ymax = ymin + goal
            #ymax = ymin + (abs(all_t_prior_new[param][-1]-all_t_no_prior_new[param][-1])*10)
            if max(np.append(all_t_no_prior[param], all_t_prior[param])) < ymax:
                ymax = max(np.append(all_t_no_prior[param], all_t_prior[param]))
            ymin = ymin - (ymax - ymin)*0.05
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
      
    
def plot_single_k(all_t_prior, all_t_no_prior, mini, maxi):
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
    all_t_prior_new = defaultdict(list)
    all_t_no_prior_new = defaultdict(list)
    for ii in range(mini, maxi):
        for param in PARAM_LIST:
            all_t_prior_new[param] = []
            all_t_no_prior_new[param] = []
    for ii in range(mini, maxi):
        for param in PARAM_LIST:
            all_t_prior_new[param] = np.append(all_t_prior_new[param],
                                               all_t_prior[ii][param])
            all_t_no_prior_new[param] = np.append(all_t_no_prior_new[param],
                                                  all_t_no_prior[ii][param])
    all_t_no_prior = all_t_no_prior_new
    all_t_prior = all_t_prior_new
    length  = len(all_t_no_prior_new['k'])

    n_s = range(mini, maxi)
    fig = plt.figure(figsize=(20, 20))
    #
    gs = GridSpec(nrows=4, ncols=4)
    ax0 = plt.subplot2grid((8, 1), (0, 0), colspan=4, rowspan=2)
    ax0.set_xticks([])
    ax1 = plt.subplot2grid((8, 1), (2, 0), colspan=4, rowspan=2)
    ax1.set_xticks([])
    ax2 = plt.subplot2grid((8, 1), (4, 0), colspan=4, rowspan=2)  
    ax2.set_xticks([])
    ax3 = plt.subplot2grid((8, 1), (6, 0), colspan=4, rowspan=2)
    axes = {}
    axes['k'] = ax0
    axes['s_d'] = ax1
    axes['n_e'] = ax2
    axes['one_t_s'] = ax3
    axes['one_t_s'].set_xlabel('Probenanzahl')
    
    for param in PARAM_LIST:
        axes[param].plot(n_s,all_t_no_prior_new[param][range(maxi)], 
                         label = 'Standard', linewidth=4, color='tab:red')
        axes[param].plot(n_s,all_t_prior_new[param][range(maxi)],
                         label = 'ML & Bayes', linewidth=4, color='navy')
        goal  = all_t_no_prior[param][-1]
        axes[param].axhline(y = goal, color ='green', linestyle = '--', label = '',  linewidth=2)
        axes[param].set_xlim([0,4+length])
        ymin = min(np.append(all_t_prior[param], all_t_no_prior[param]))
        ymax = ymin + goal
        #if max(np.append(all_t_no_prior[param], all_t_prior[param])) < ymax:   ## add it if the ymax value is too high
        #    ymax = max(np.append(all_t_no_prior[param], all_t_prior[param]))
        ymin = ymin - (ymax - ymin)*0.05
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
        
def plot_multi_k(all_t_prior1, all_t_prior2, all_t_prior3, all_t_prior4, mini, maxi):
    '''
    Plot function for calc_t_standard from the t.py script. It is a plot to compare 
    two fuctions of t over the number of experiments.

    Parameters
    ----------
    all_t_prior : Array
        All Ts from the experiment planning with prior.
        DESCRIPTION.
    mini : Int
        Min number of experiments.
    maxi : Int
        Max number of experiments.

    Returns
    -------
    None.

    '''
    all_t_prior_new1 = {}

    for ii in range(mini, maxi+1):
        for param in PARAM_LIST:
            all_t_prior_new1[param] = []

    for ii in range(mini, maxi+1):
        for param in PARAM_LIST:
            all_t_prior_new1[param] = np.append(all_t_prior_new1[param],
                                               all_t_prior1[ii][param])


    all_t_prior1 = all_t_prior_new1
    length1  = len(all_t_prior_new1['k'])
    
    all_t_prior_new2 = {}

    for ii in range(mini, maxi+1):
        for param in PARAM_LIST:
            all_t_prior_new2[param] = []

    for ii in range(mini, maxi+1):
        for param in PARAM_LIST:
            all_t_prior_new2[param] = np.append(all_t_prior_new2[param],
                                               all_t_prior2[ii][param])

    all_t_prior2 = all_t_prior_new2
    length2  = len(all_t_prior_new2['k'])
    
    all_t_prior_new3 = {}

    for ii in range(mini, maxi+1):
        for param in PARAM_LIST:
            all_t_prior_new3[param] = []

    for ii in range(mini, maxi+1):
        for param in PARAM_LIST:
            all_t_prior_new3[param] = np.append(all_t_prior_new3[param],
                                               all_t_prior3[ii][param])

    all_t_prior3 = all_t_prior_new3
    length3  = len(all_t_prior_new3['k'])
    
    all_t_prior_new4 = {}

    for ii in range(mini, maxi+1):
        for param in PARAM_LIST:
            all_t_prior_new4[param] = []

    for ii in range(mini, maxi+1):
        for param in PARAM_LIST:
            all_t_prior_new4[param] = np.append(all_t_prior_new4[param],
                                               all_t_prior4[ii][param])

    all_t_prior4 = all_t_prior_new4
    length4  = len(all_t_prior_new4['k'])
    
    

    n_s = range(mini, maxi)
    fig = plt.figure(figsize=(20, 20))
    #
    gs = GridSpec(nrows=4, ncols=4)
    ax0 = plt.subplot2grid((8, 1), (0, 0), colspan=4, rowspan=2)
    ax0.set_xticks([])
    
    ax1 = plt.subplot2grid((8, 1), (2, 0), colspan=4, rowspan=2)
    ax1.set_xticks([])
    
    ax2 = plt.subplot2grid((8, 1), (4, 0), colspan=4, rowspan=2)  
    ax2.set_xticks([])
    
    ax3 = plt.subplot2grid((8, 1), (6, 0), colspan=4, rowspan=2)
    axes = {}
    axes['k'] = ax0
    axes['s_d'] = ax1
    axes['n_e'] = ax2
    axes['one_t_s'] = ax3
    axes['one_t_s'].set_xlabel('Probenanzahl')
    
    for param in PARAM_LIST:
        
        axes[param].plot(n_s,all_t_prior_new1[param][range(maxi)],
                         label = 'ML & Bayes OG', linewidth=3, color='navy')
        
        axes[param].plot(n_s,all_t_prior_new2[param][range(maxi)],
                         label = 'ML & Bayes mu', linewidth=3, color='red')
        
        axes[param].plot(n_s,all_t_prior_new3[param][range(maxi)],
                         label = 'ML & Bayes sigma', linewidth=3, color='blue')
        
        axes[param].plot(n_s,all_t_prior_new4[param][range(maxi)],
                         label = 'ML & Bayes on mu', linewidth=3, color='black')
        
        
        goal1 = all_t_prior1[param][-1]
        goal2 = all_t_prior2[param][-1]
        goal3 = all_t_prior3[param][-1]
        goal4 = all_t_prior4[param][-1]
        goal = max(goal1, goal2, goal3, goal4)
        
        axes[param].axhline(y = goal1, color ='navy', linestyle = '-.', label = '',  linewidth=2)
        axes[param].set_xlim([1,4+length1])
        axes[param].axhline(y = goal2, color ='red', linestyle = '-.', label = '',  linewidth=2)
        axes[param].set_xlim([1,4+length2])
        axes[param].axhline(y = goal3, color ='blue', linestyle = '-.', label = '',  linewidth=2)
        axes[param].set_xlim([1,4+length3])
        axes[param].axhline(y = goal4, color ='black', linestyle = '-.', label = '',  linewidth=2)
        axes[param].set_xlim([1,4+length4])
        
        ymin1 = min(np.append(all_t_prior1[param], all_t_prior1[param]))
        ymin2 = min(np.append(all_t_prior2[param], all_t_prior2[param]))
        ymin3 = min(np.append(all_t_prior3[param], all_t_prior3[param]))
        ymin4 = min(np.append(all_t_prior4[param], all_t_prior4[param]))
        
        ymax1 = max(np.append(all_t_prior1[param], all_t_prior1[param]))
        ymax2 = max(np.append(all_t_prior2[param], all_t_prior2[param]))
        ymax3 = max(np.append(all_t_prior3[param], all_t_prior3[param]))
        ymax4 = max(np.append(all_t_prior4[param], all_t_prior4[param]))
        
        ymax_ini = max(ymax1, ymax2, ymax3, ymax4)
        ymin_ini = min(ymin1, ymin2, ymin3, ymin4)
        
        ymax = ymin_ini + goal
        
        if ymax_ini < ymax:
            ymax = ymax_ini
            
        ymin = ymin_ini - (ymax - ymin_ini)*0.05
        
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
        print(goal)
        
def plot_quantile(quantile50,quantile95,quantile05,mini, maxi,tolerance):
    
    length = len(quantile50['k'])
    n_s = range(mini+3, maxi+3)
    fig = plt.figure(figsize=(20, 20))
    #
    gs = GridSpec(nrows=4, ncols=4)
    ax0 = plt.subplot2grid((8, 1), (0, 0), colspan=4, rowspan=2)
    ax0.set_xticks([])
    ax1 = plt.subplot2grid((8, 1), (2, 0), colspan=4, rowspan=2)
    ax1.set_xticks([])
    ax2 = plt.subplot2grid((8, 1), (4, 0), colspan=4, rowspan=2)  
    ax2.set_xticks([])
    ax3 = plt.subplot2grid((8, 1), (6, 0), colspan=4, rowspan=2)
    axes = {}
    axes['k'] = ax0
    axes['s_d'] = ax1
    axes['n_e'] = ax2
    axes['one_t_s'] = ax3
    
    for param in PARAM_LIST:
        
        axes[param].plot(n_s,quantile50[param][range(maxi)],
                         label = 'quantile_50', linewidth=4, color='black')
        axes[param].plot(n_s,quantile95[param][range(maxi)],
                         label = 'quantile_95', linewidth=4, color='blue')
        axes[param].plot(n_s,quantile05[param][range(maxi)],
                         label = 'quantile_05', linewidth=4, color='red')

        goal  = quantile50[param][-1]
        if goal == 0:
            goal_05 = -tolerance
            goal_95 = tolerance
        else:
            goal_05 = goal - goal*tolerance
            goal_95 = goal + goal*tolerance
        
        axes[param].axhline(y = goal_95, color ='orange', linestyle = '--', label = '',  linewidth=2)
        axes[param].axhline(y = goal, color ='green', linestyle = '--', label = '',  linewidth=2)
        #axes[param].axhline(y = goal_05, color ='orange', linestyle = '--', label = '',  linewidth=2)
        
        axes[param].set_xlim([3,4+length])
        ymini = min(quantile05[param])
        ymin = min(ymini, goal_05)
        ymaxi = max(quantile95[param])
        ymax = max(ymaxi, goal_95)
        
        #if max(quantile50[param]) < ymax:   ## add it if the ymax value is too high
        #    ymax = max(quantile50[param])
        ymin = ymin - (ymax - ymin)*0.05
        ymax = ymax + (ymax - ymin)*0.05
        axes[param].set_ylim([ymin, ymax])
        
        if goal == 0:
            if param == 'k':
                ylabel = 'Error for k'
            elif param == 'n_e':
                ylabel = 'Error for $N_{K}$'
            elif param == 's_d':
                ylabel = 'Error for $L_{a,L}$'
            elif param == 'one_t_s':
                ylabel = 'Error for $T_{La}$'
        if goal != 0:
            if param == 'k':
                ylabel = 'MLE for k'
            elif param == 'n_e':
                ylabel = 'MLE for $N_{K}$'
            elif param == 's_d':
                ylabel = 'MLE for $L_{a,L}$'
            elif param == 'one_t_s':
                ylabel = 'MLE for $T_{La}$'
                
        axes[param].set_ylabel(ylabel, fontsize = 18)
        axes[param].set_xlabel('number of tests', fontsize = 18)
        axes[param].legend( loc = 'upper right')
        
def plot_quantile_with_hdi(quantile50,quantile95,quantile05,hdi_low,hdi_high,mini, maxi,tolerance):
    
    length = len(quantile50['k'])
    n_s = range(mini, maxi)
    fig = plt.figure(figsize=(20, 20))
    #
    gs = GridSpec(nrows=4, ncols=4)
    ax0 = plt.subplot2grid((8, 1), (0, 0), colspan=4, rowspan=2)
    ax0.set_xticks([])
    ax1 = plt.subplot2grid((8, 1), (2, 0), colspan=4, rowspan=2)
    ax1.set_xticks([])
    ax2 = plt.subplot2grid((8, 1), (4, 0), colspan=4, rowspan=2)  
    ax2.set_xticks([])
    ax3 = plt.subplot2grid((8, 1), (6, 0), colspan=4, rowspan=2)
    axes = {}
    axes['k'] = ax0
    axes['s_d'] = ax1
    axes['n_e'] = ax2
    axes['one_t_s'] = ax3
    
    for param in PARAM_LIST:
        
        axes[param].plot(n_s,quantile50[param][range(maxi)],
                         label = 'quantile_50', linewidth=4, color='black')
        axes[param].plot(n_s,quantile95[param][range(maxi)],
                         label = 'quantile_95', linewidth=4, color='blue')
        axes[param].plot(n_s,quantile05[param][range(maxi)],
                         label = 'quantile_05', linewidth=4, color='red')
        
        axes[param].plot(n_s,hdi_low[param][range(maxi)],
                         label = 'hdi_low', linewidth=2, color='orange', linestyle = '--')
        axes[param].plot(n_s,hdi_high[param][range(maxi)],
                         label = 'hdi_high', linewidth=2, color='orange', linestyle = '--')

        goal  = quantile50[param][-1]
        
        axes[param].axhline(y = goal, color ='green', linestyle = '--', label = '',  linewidth=2)
        
        axes[param].set_xlim([0,4+length])
        ymini = min(quantile05[param])
        ymin = min(ymini, goal_05)
        ymaxi = max(quantile95[param])
        ymax = max(ymaxi, goal_95)
        
        #if max(quantile50[param]) < ymax:   ## add it if the ymax value is too high
        #    ymax = max(quantile50[param])
        ymin = ymin - (ymax - ymin)*0.05
        ymax = ymax + (ymax - ymin)*0.05
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
        

def plot_cont(self, ax=None):
    '''
    Plotting pymc3 distributions, for example (Normal, Uniform, Beta, ...etc)
    
    Parameters
    ----------
    ax : Axes, optional
        The axes for the plot. The default is None.
    '''
    if ax is None:
        _, ax = plt.subplots()
    samples = self.random(size=1000)
    x = np.linspace(np.min(samples), np.max(samples), 1000)
    ax.plot(x, np.exp(self.logp(x)).eval())
    return ax

def prior_data_conflict_plot(mean,info_ppd_std,noninfo_ppd_std,sigma_Ts,MLE,confidence,cdf_og,x,
                             prior_from_ki,prior_from_ki_std,non_inf,non_inf_std,conflict_test):
    psi = 0
    mixture_mean = mean
    alpha = 1 - confidence
    informative_prior = stats.norm.pdf(x,prior_from_ki,prior_from_ki_std+sigma_Ts)
    noninformative_prior = stats.norm.pdf(x,non_inf,non_inf_std+sigma_Ts)
    
    if cdf_og >= 0.5:
        p_value = 2*(1-cdf_og)
        if p_value > alpha:  
            print('psi =',psi)
            print('P_value =',p_value)
            print('')
    else:
        p_value = 2*cdf_og
        if p_value > alpha:  
            print('psi =',psi)
            print('P_value =',p_value)
            print('')

    num_try = 0
    while p_value < alpha:

        mixture_std = (1-psi)*info_ppd_std + psi*noninfo_ppd_std
        mixture_prior = stats.norm.pdf(x,mixture_mean,mixture_std)
        cdf_mixture = stats.norm(loc= mixture_mean, scale=mixture_std).cdf(MLE)

        if cdf_mixture >= 0.5:
            p_value = 2*(1-cdf_mixture)
        else:
            p_value = 2*cdf_mixture

        print('psi =',psi)
        print('P_value =',p_value)
        print('')
        if p_value >= (alpha - 0.03):
            psi = psi + 0.01
        elif p_value <= (alpha - 0.08):
            psi = psi + 0.05
        else:
            psi = psi + 0.02
        num_try += 1
    if p_value > alpha and num_try == 0: 
        mixture_prior = informative_prior
    
    return informative_prior, noninformative_prior, mixture_prior, x, sigma_Ts, num_try