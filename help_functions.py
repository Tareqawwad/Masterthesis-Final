# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 11:26:27 2021

@author: Sven Mordeja
@modified by: Tareq Awwad

"""
import numpy as np
import scipy.stats as stats
from WoehlerParams import WoehlerCurve
from WoehlerParams2 import WoehlerCurve2
from collections import defaultdict
from data_manager import DataManager as DataManager
from data_manager_OG import DataManager as DataManager2
from random import shuffle
'''
In this script some helperfunctions are collected. Some functions are used in multiple scripts.

'''
new_data_manager = DataManager()

#%% help functions
def calc_n(woehler_params, s):
    '''
    Calculates the number of cycles for a 50% failure probaility.

    Parameters
    ----------
    woehler_params : Dict
        A dictionary containing the four woehler parameter.
    s : Int
        The amlitude with 50% failure probaility at which n is calculated.

    Returns
    -------
    n : Int
        Number of cycles for a 50% failure probaility.

    '''
    k = woehler_params['k']
    s_d_50 = woehler_params['s_d']
    n_e     = woehler_params['n_e']
    one_t_s = woehler_params['one_t_s']
    n      = 10**(np.log10(n_e)+k*(np.log10(s_d_50)-np.log10(s)))
    return n
def calc_s_long(woehler_params, p):
    '''
    Calculates the amplitude for a runout probaility p.

    Parameters
    ----------
    woehler_params : Dict
        A dictionary containing the four woehler parameter.
    p : Int
        Runout probability.

    Returns
    -------
    s : Int
        The amlitude with a runout probaility p.

    '''
    k = woehler_params['k']
    s_d_50 = woehler_params['s_d']
    n_e     = woehler_params['n_e']
    one_t_s = woehler_params['one_t_s']
    factor_p = stats.norm.ppf(p)
    s_p = 1/(2.564)*factor_p*np.log10(one_t_s)
    s = 10**(s_p + np.log10(s_d_50))
    return s
    
def calc_s_short(woehler_params, n):
    '''
    Calculates the amplitude for a 50% failure probaility.

    Parameters
    ----------
    woehler_params : Dict
        A dictionary containing the four woehler parameter.
    n : Int
        Number of cycles for a 50% failure probaility.

    Returns
    -------
    s : Int
        The amlitude with 50% failure probaility.

    '''
    k = woehler_params['k']
    s_d_50 = woehler_params['s_d']
    n_e     = woehler_params['n_e']
    one_t_s = woehler_params['one_t_s']
    s= 10**(-1*((np.log10(n)-np.log10(n_e))/k-np.log10(s_d_50)))
    
    return s
def get_runout_and_failure(woehler_points):
    '''
    Returns the amplitudes and number of cycles of the runouts and the failures seperatly.

    Parameters
    ----------
    woehler_points : Dict
        A dictonary containing the measured data points. 
        (s_a, n, outcome)

    Returns
    -------
    s_a_failure : Array
        All failure amplitudes.
    n_failure : Array
        All failure amplitudes.
    s_a_runout : Array
        All runout amplitudes.
    n_runout : Array
        All runout amplitudes.

    '''
    s_a = woehler_points['s_a']
    n = woehler_points['n']
    outcome = woehler_points['outcome']
    
    n_failure = np.array([])
    n_runout = np.array([])
    s_a_failure = np.array([])
    s_a_runout = np.array([])
    for ii in range(len(outcome)):
           
        if outcome[ii] == 'failure':
            n_failure = np.append(n_failure, n[ii])
            s_a_failure = np.append(s_a_failure, s_a[ii])
            
        else:
            n_runout = np.append(n_runout, n[ii])
            s_a_runout = np.append(s_a_runout, s_a[ii])
    return s_a_failure, n_failure, s_a_runout, n_runout

def sort_failure_and_runout(Out):
    
    runout_indexes = []
    failure_indexes = []
    i = 0
    length = len(Out)

    while i < length:
        if 'runout' in Out[i]:
            runout_indexes.append(i)

        if 'failure' in Out[i]:
            failure_indexes.append(i)
        i += 1
    return runout_indexes, failure_indexes

def sort_arrays(experiments,num,length_of_exp_all,l,s_a_rand_all,n_rand_all,outcome_rand_all):
    '''
    it fills out the s,s1,s2,s3 arrays which are used to get 
    the overall error later on
    
    s  : a dictionary that is used to fill out s0
    s0 : a dictionary of all MLE paramter values
    s1 : a three dimensional array (num_of_samp X length X 4)
    s2 : a three dimensional array (length X num_of_samp X 4) #collects all the sample from each test together
    s3[] : a three dimensional array (num_of_samp of all exp combined X length X 4)

    '''
    
    s1=defaultdict(dict)
    s1[0] = [[[0 for i in range(4)] for j in range(length_of_exp_all[0]-2)]for s in range(num)]
    s1[1] = [[[0 for i in range(4)] for j in range(length_of_exp_all[1]-2)]for s in range(num)]
    s1[2] = [[[0 for i in range(4)] for j in range(length_of_exp_all[2]-2)]for s in range(num)]
    s1[3] = [[[0 for i in range(4)] for j in range(length_of_exp_all[3]-2)]for s in range(num)]
    s1[4] = [[[0 for i in range(4)] for j in range(length_of_exp_all[4]-2)]for s in range(num)]
    s1[5] = [[[0 for i in range(4)] for j in range(length_of_exp_all[5]-2)]for s in range(num)]
    s1[6] = [[[0 for i in range(4)] for j in range(length_of_exp_all[6]-2)]for s in range(num)]
    s1[7] = [[[0 for i in range(4)] for j in range(length_of_exp_all[7]-2)]for s in range(num)]

    s2=defaultdict(dict)
    s2[0] = [[[0 for i in range(4)] for j in range(num)]for s in range(length_of_exp_all[0]-2)]
    s2[1] = [[[0 for i in range(4)] for j in range(num)]for s in range(length_of_exp_all[1]-2)]
    s2[2] = [[[0 for i in range(4)] for j in range(num)]for s in range(length_of_exp_all[2]-2)]
    s2[3] = [[[0 for i in range(4)] for j in range(num)]for s in range(length_of_exp_all[3]-2)]
    s2[4] = [[[0 for i in range(4)] for j in range(num)]for s in range(length_of_exp_all[4]-2)]
    s2[5] = [[[0 for i in range(4)] for j in range(num)]for s in range(length_of_exp_all[5]-2)]
    s2[6] = [[[0 for i in range(4)] for j in range(num)]for s in range(length_of_exp_all[6]-2)]
    s2[7] = [[[0 for i in range(4)] for j in range(num)]for s in range(length_of_exp_all[7]-2)]

    s3 = defaultdict(dict)
    s3['1st'] = [[[0 for i in range(4)] for j in range(length_of_exp_all[0]-2)]for s in range(num)]
    s3['2nd'] = [[[0 for i in range(4)] for j in range(length_of_exp_all[1]-2)]for s in range(num)]
    s3['3rd'] = [[[0 for i in range(4)] for j in range(length_of_exp_all[2]-2)]for s in range(num)]
    s3['4th'] = [[[0 for i in range(4)] for j in range(length_of_exp_all[3]-2)]for s in range(num)]
    s3['5th'] = [[[0 for i in range(4)] for j in range(length_of_exp_all[4]-2)]for s in range(num)]
    s3['6th'] = [[[0 for i in range(4)] for j in range(length_of_exp_all[5]-2)]for s in range(num)]
    s3['7th'] = [[[0 for i in range(4)] for j in range(length_of_exp_all[6]-2)]for s in range(num)]
    s3['8th'] = [[[0 for i in range(4)] for j in range(length_of_exp_all[7]-2)]for s in range(num)]

    for exp in range(8):

        curve = []

        s0 = defaultdict(dict)
        s0['k'] = []
        s0['s_d'] = []
        s0['n_e'] = []
        s0['one_t_s'] = []

        for iii in range(num):
            s = {}
            s['k'] = {}
            s['s_d'] = {}
            s['n_e'] = {}
            s['one_t_s'] = {}

            for ii in range (l): 

                curve_s = WoehlerCurve2(s_a_rand_all[exp][iii][0:ii+3], n_rand_all[exp][iii][0:ii+3], outcome_rand_all[exp][iii][0:ii+3])
                curve = np.append(curve,curve_s)
                s = np.append(s,new_data_manager.transform_woehler_params(curve_s.Mali_4p_result))

                if ii== l-1:
                    for i in range(experiments[exp]):
                        s0[iii][i] = s[i+1]

            for ii in range(experiments[exp]):
                s1[exp][iii][ii][0] = s[ii+1]['k']
                s1[exp][iii][ii][1] = s[ii+1]['s_d']
                s1[exp][iii][ii][2] = s[ii+1]['n_e']
                s1[exp][iii][ii][3] = s[ii+1]['one_t_s']


            for i in range(experiments[exp]):
                s2[exp][i][iii] = s1[exp][iii][i]

        if exp == 0:
            s3['1st'] = s1[0]
        if exp == 1:
            s3['2nd'] = s1[1]
        if exp == 2:
            s3['3rd'] = s1[2]
        if exp == 3:
            s3['4th'] = s1[3]
        if exp == 4:
            s3['5th'] = s1[4]
        if exp == 5:
            s3['6th'] = s1[5]
        if exp == 6:
            s3['7th'] = s1[6]
        if exp == 7:
            s3['8th'] = s1[7]
            
    return s1,s2,s3

def exp_error(length_of_exp_all,length_of_error1,length_of_error2,num,s3):
    '''
    it finds the error of each expirement and then separate the 8 expirements into two groups er1 & er2
    er1: has less than 20 tests (18)
    er2: has more than 20 tests (30)
    '''


    error1 = [[[0 for i in range(4)] for j in range(length_of_exp_all[0]-2)]for s in range(num)]
    error2 = [[[0 for i in range(4)] for j in range(length_of_exp_all[1]-2)]for s in range(num)]
    error3 = [[[0 for i in range(4)] for j in range(length_of_exp_all[2]-2)]for s in range(num)]
    error4 = [[[0 for i in range(4)] for j in range(length_of_exp_all[3]-2)]for s in range(num)]
    error5 = [[[0 for i in range(4)] for j in range(length_of_exp_all[4]-2)]for s in range(num)]
    error6 = [[[0 for i in range(4)] for j in range(length_of_exp_all[5]-2)]for s in range(num)]
    error7 = [[[0 for i in range(4)] for j in range(length_of_exp_all[6]-2)]for s in range(num)]
    error8 = [[[0 for i in range(4)] for j in range(length_of_exp_all[7]-2)]for s in range(num)]

    er1 = [[[0 for i in range(4)] for j in range(num*4)]for s in range(length_of_error1)]
    er2 = [[[0 for i in range(4)] for j in range(num*4)]for s in range(length_of_error2)]

    for exp in range(4):
        length = length_of_exp_all[exp]-1
        true_MLE = 17

        for iii in range(num):
            for ii in range(true_MLE):
                for i in range(4):
                    if exp == 0:

                        error1[iii][ii][i] = abs(s3['1st'][iii][ii][i] - s3['1st'][iii][true_MLE][i])/s3['1st'][iii][true_MLE][i]
                    elif exp == 1:
                        error2[iii][ii][i] = abs(s3['2nd'][iii][ii][i] - s3['2nd'][iii][true_MLE][i])/s3['2nd'][iii][true_MLE][i]
                    elif exp == 2:
                        error3[iii][ii][i] = abs(s3['3rd'][iii][ii][i] - s3['3rd'][iii][true_MLE][i])/s3['3rd'][iii][true_MLE][i]
                    elif exp == 3:
                        error4[iii][ii][i] = abs(s3['4th'][iii][ii][i] - s3['4th'][iii][true_MLE][i])/s3['4th'][iii][true_MLE][i]


            for l in range(length_of_error1):
                if exp == 0:
                    er1[l][iii] = error1[iii][l]
                elif exp == 1:
                    er1[l][98+iii] = error2[iii][l]
                elif exp == 2:
                    er1[l][196+iii] = error3[iii][l]
                elif exp == 3:
                    er1[l][294+iii] = error4[iii][l]

    for exp in range(4):
        print(s3['5th'][0][25][0])
        print(s3['5th'][0][26][0])
        print(s3['5th'][0][29][0])
        length = length_of_exp_all[exp+4]-1
        true_MLE = 29
        
        for iii in range(num):
            for ii in range(true_MLE):
                for i in range(4):
                    if exp == 0:
                        error5[iii][ii][i] =abs(s3['5th'][iii][ii][i] - s3['5th'][iii][true_MLE][i])/s3['5th'][iii][true_MLE][i]
                    elif exp == 1:
                        error6[iii][ii][i] = abs(s3['6th'][iii][ii][i] - s3['6th'][iii][true_MLE][i])/s3['6th'][iii][true_MLE][i]
                    elif exp == 2:
                        error7[iii][ii][i] = abs(s3['7th'][iii][ii][i] - s3['7th'][iii][true_MLE][i])/s3['7th'][iii][true_MLE][i]
                    elif exp == 3:
                        error8[iii][ii][i] = abs(s3['8th'][iii][ii][i] - s3['8th'][iii][true_MLE][i])/s3['8th'][iii][true_MLE][i]


            for l in range(length_of_error2):
                
                if exp == 0:
                    er2[l][iii] = error5[iii][l]
                elif exp == 1:
                    er2[l][98+iii] = error6[iii][l]
                elif exp == 2:
                    er2[l][196+iii] = error7[iii][l]
                elif exp == 3:
                    er2[l][294+iii] = error8[iii][l]
    return er1,er2
def arrange_errors(length_of_error1,length_of_error2,err1_50,err1_95,err1_05,err2_50,err2_95,err2_05):
    
    error1_50 = defaultdict(dict)
    error1_50['k'] = []
    error1_50['s_d'] = []
    error1_50['n_e'] = []
    error1_50['one_t_s'] = []

    error1_05 = defaultdict(dict)
    error1_05['k'] = []
    error1_05['s_d'] = []
    error1_05['n_e'] = []
    error1_05['one_t_s'] = []

    error1_95 = defaultdict(dict)
    error1_95['k'] = []
    error1_95['s_d'] = []
    error1_95['n_e'] = []
    error1_95['one_t_s'] = []

    error2_50 = defaultdict(dict)
    error2_50['k'] = []
    error2_50['s_d'] = []
    error2_50['n_e'] = []
    error2_50['one_t_s'] = []

    error2_05 = defaultdict(dict)
    error2_05['k'] = []
    error2_05['s_d'] = []
    error2_05['n_e'] = []
    error2_05['one_t_s'] = []

    error2_95 = defaultdict(dict)
    error2_95['k'] = []
    error2_95['s_d'] = []
    error2_95['n_e'] = []
    error2_95['one_t_s'] = []

    for i in range(length_of_error1):
        error1_50['k'] = np.append(error1_50['k'],err1_50[i][0])
        error1_50['s_d'] = np.append(error1_50['s_d'],err1_50[i][1])
        error1_50['n_e'] = np.append(error1_50['n_e'],err1_50[i][2])
        error1_50['one_t_s'] = np.append(error1_50['one_t_s'],err1_50[i][3])

    for i in range(length_of_error1):
        error1_95['k'] = np.append(error1_95['k'],err1_95[i][0])
        error1_95['s_d'] = np.append(error1_95['s_d'],err1_95[i][1])
        error1_95['n_e'] = np.append(error1_95['n_e'],err1_95[i][2])
        error1_95['one_t_s'] = np.append(error1_95['one_t_s'],err1_95[i][3])

    for i in range(length_of_error1):
        error1_05['k'] = np.append(error1_05['k'],err1_05[i][0])
        error1_05['s_d'] = np.append(error1_05['s_d'],err1_05[i][1])
        error1_05['n_e'] = np.append(error1_05['n_e'],err1_05[i][2])
        error1_05['one_t_s'] = np.append(error1_05['one_t_s'],err1_05[i][3])

    for i in range(length_of_error2):
        error2_50['k'] = np.append(error2_50['k'],err2_50[i][0])
        error2_50['s_d'] = np.append(error2_50['s_d'],err2_50[i][1])
        error2_50['n_e'] = np.append(error2_50['n_e'],err2_50[i][2])
        error2_50['one_t_s'] = np.append(error2_50['one_t_s'],err2_50[i][3])

    for i in range(length_of_error2):
        error2_95['k'] = np.append(error2_95['k'],err2_95[i][0])
        error2_95['s_d'] = np.append(error2_95['s_d'],err2_95[i][1])
        error2_95['n_e'] = np.append(error2_95['n_e'],err2_95[i][2])
        error2_95['one_t_s'] = np.append(error2_95['one_t_s'],err2_95[i][3])

    for i in range(length_of_error2):
        error2_05['k'] = np.append(error2_05['k'],err2_05[i][0])
        error2_05['s_d'] = np.append(error2_05['s_d'],err2_05[i][1])
        error2_05['n_e'] = np.append(error2_05['n_e'],err2_05[i][2])
        error2_05['one_t_s'] = np.append(error2_05['one_t_s'],err2_05[i][3])
    return error1_50, error1_05, error1_95,error2_50, error2_05, error2_95
    
def randomize(n,s_a,outcome):
    '''
    function to get random sample of the desired experiment
    '''
    random = {}
    random['n'] = n
    random['s_a'] = s_a
    random['outcome'] = outcome

    rand = list(zip(n, s_a, outcome))
    shuffle(rand)

    random['n'] = n
    random['s_a'] = s_a
    random['outcome'] = outcome

    random['n'], random['s_a'], random['outcome'] = zip(*rand)
    
    random['n'] = list(random['n'])
    random['s_a'] = list(random['s_a'])
    random['outcome'] = list(random['outcome'])
    
    return random

def organize(random):
    '''
    function to organize the random sample of the desired experiment
    '''
    rand = {}
    rand['n']= {}
    rand['s_a']= {}
    rand['outcome']= {}
    
    rand['n'] = random['n']
    rand['s_a'] = random['s_a']
    rand['outcome'] = random['outcome']
    
    return rand