# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 20:23:02 2021

@author: Sven Mordeja
"""
#%%imports and constants
import a_plot_functions
import arviz as az
import help_functions
import numpy as np
import pickle
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
from a_data_manager import DataManager
from a_doe import DOE
from a_bays_model import BaysModel
from WoehlerParams import WoehlerCurve
from multiprocessing import Process, freeze_support

DB_IDS = ['NIMS103','NIMS679','KOL2RNG.9']
PARAM_LIST = ['k', 's_d', 'n_e', 'one_t_s']

#%% funtions to calculate the spread (Streuspanne)
def calc_t_standard(index):
    '''
    This function calculates the spread (Streuespanne) for a standard planning methode. The staircase methode and the "Perlenschnurverfahren" is used for the planning.
    
    Parameters
    ----------
    index : Integer
        Index of a dataset (colum number) eg. 0, 1, 2,... .

    Returns
    -------
    None.

    '''
    new_data_manager = DataManager()
    new_bays = BaysModel()
    new_doe = DOE()
    new_data_manager.load_ki_predictions_from_csv_new(filename = 'ml-prediction.csv',
                                                      delimiter = ',')
    prior_from_ki = new_data_manager.get_ki_predictions_by_index(index)
    db_id = DB_IDS[index]
    n, s_a, outcome = new_data_manager.get_woehler_data_by_db_id(db_id)                         
    curve = WoehlerCurve(s_a, n, outcome)
    woehler_params_original = new_bays.transform_woehler_params(curve.Mali_4p_result)
    #get the wohler points from the monte methode:
    all_t, woehler_points = calc_t_monte(index)
    woehler_points_new ={}
    woehler_points_new['n'] = np.array([])
    woehler_points_new['s_a'] = np.array([])
    woehler_points_new['outcome'] = np.array([])
    all_t_no_prior = {}
    all_t_prior = {}
    
    for ii in range(len(woehler_points['outcome'])+1):
        all_t_no_prior[ii] ={}
        all_t_prior[ii] ={}
        for param in PARAM_LIST:
            all_t_no_prior[ii][param] = []
            all_t_prior[ii][param] = []
    for ii in range(len(woehler_points['outcome'])):
        woehler_points_new['n'] = np.append(woehler_points_new['n'],
                                          woehler_points['n'][ii])
        woehler_points_new['s_a'] = np.append(woehler_points_new['s_a'],
                                            woehler_points['s_a'][ii])
        woehler_points_new['outcome'] = np.append(woehler_points_new['outcome'],
                                                woehler_points['outcome'][ii])
        woehler_params_bayes, trace, prior_distribution = new_bays.calc_model(
            woehler_points_new, num_samples = 1000, prior_from_ki = prior_from_ki)
        woehler_params_bayes, trace_no_prior, prior_distribution = new_bays.calc_model(
            woehler_points_new, num_samples = 1000)
        for parameter in PARAM_LIST:
            hdi_interval = az.hdi(trace[parameter], 0.8)
            t_current = hdi_interval[1] / hdi_interval[0]
            all_t_prior[len(woehler_points_new['s_a'])][parameter] = \
                np.append(all_t_prior[len(woehler_points_new['s_a'])][parameter], t_current)
            
            hdi_interval = az.hdi(trace_no_prior[parameter], 0.8)
            t_current = hdi_interval[1] / hdi_interval[0]
            all_t_no_prior[len(woehler_points_new['s_a'])][parameter] = \
                np.append(all_t_no_prior[len(woehler_points_new['s_a'])][parameter], t_current)
    
    index = 10 + index
    save_t(index,all_t_prior, all_t_no_prior, woehler_points, woehler_points)
    a_plot_functions.plot_k(all_t_prior, all_t_no_prior,4,32)
    
def calc_t_doe(index):
    '''
    This function calculates the spread (Streuespanne) for the new planning methode. 
    
    Parameters
    ----------
    index : Integer
        Index of a dataset (colum number) eg. 0, 1, 2,... .

    Returns
    -------
    None.

    '''
    new_data_manager = DataManager()
    new_bays = BaysModel()
    new_doe = DOE()
    new_data_manager.load_ki_predictions_from_csv_new(filename = 'ml-prediction.csv',
                                                      delimiter = ',')
    db_id = DB_IDS[index]
    n, s_a, outcome = new_data_manager.get_woehler_data_by_db_id(db_id) 
    prior_from_ki = new_data_manager.get_ki_predictions_by_index(index)
    db_id = new_data_manager.get_db_id_by_index(index)
    n, s_a, outcome = new_data_manager.get_woehler_data_by_db_id(db_id)                
    curve = WoehlerCurve(s_a, n, outcome)
    woehler_params_original = new_bays.transform_woehler_params(curve.Mali_4p_result)
    load_levels = new_doe.generate_load_levels(woehler_params_original)
    all_t_prior = {}
    for ii in range(33):
        all_t_prior[ii] ={}
        for param in PARAM_LIST:
            all_t_prior[ii][param] = []  
    woehler_points = {}
    woehler_points['n'] = []
    woehler_points['s_a'] = []
    woehler_points['outcome'] = [] 
    load_level_1 = load_levels[1]
    load_levels = new_doe.generate_load_levels_2(woehler_params_original, load_levels)
    outcome, n_sample = new_doe.generate_woehler_point(woehler_params_original, max(load_levels))
    woehler_points['n'] = np.append(woehler_points['n'],n_sample)
    woehler_points['s_a'] = np.append(woehler_points['s_a'], max(load_levels))
    woehler_points['outcome'] = np.append(woehler_points['outcome'],outcome)
    
    outcome, n_sample = new_doe.generate_woehler_point(woehler_params_original, load_level_1)
    woehler_points['n'] = np.append(woehler_points['n'],n_sample)
    woehler_points['s_a'] = np.append(woehler_points['s_a'],load_level_1)
    woehler_points['outcome'] = np.append(woehler_points['outcome'],outcome)
    
    woehler_points['n'] =  np.append(woehler_points['n'],2e7)
    woehler_points['s_a'] = np.append(woehler_points['s_a'],woehler_params_original['s_d'])
    woehler_points['outcome'] = np.append(woehler_points['outcome'],'runout')
    curve = WoehlerCurve(woehler_points['s_a'], woehler_points['n'] , woehler_points['outcome'])
    
    woehler_params_bayes, trace, prior_distribution = new_bays.calc_model(
        woehler_points, num_samples = 1000, prior_from_ki = prior_from_ki)
    
    for parameter in PARAM_LIST:
        hdi_interval = az.hdi(trace[parameter], 0.8)
        t_current = hdi_interval[1] / hdi_interval[0]
        all_t_prior[len(woehler_points['s_a'])][parameter] = \
            np.append(all_t_prior[len(woehler_points['s_a'])][parameter] ,t_current)
    #plan the raimaning experiments
    for ii in range(29):
        load_level = new_doe.find_best_load_level(trace, curve, load_levels)
        outcome, n_sample = new_doe.generate_woehler_point(woehler_params_original, load_level)
        woehler_points['n'] = np.append(woehler_points['n'],n_sample)
        woehler_points['s_a'] = np.append(woehler_points['s_a'],load_level)
        woehler_points['outcome'] = np.append(woehler_points['outcome'],outcome)
        curve = WoehlerCurve(woehler_points['s_a'], woehler_points['n'] , woehler_points['outcome'])
        
        woehler_params_bayes, trace, prior_distribution = new_bays.calc_model(
            woehler_points, num_samples = 1000, prior_from_ki = prior_from_ki)
        for parameter in PARAM_LIST:
            hdi_interval = az.hdi(trace[parameter], 0.8)
            t_current = hdi_interval[1] / hdi_interval[0]
            all_t_prior[len(woehler_points['s_a'])][parameter] = \
                np.append(all_t_prior[len(woehler_points['s_a'])][parameter],t_current)
    
    #while the save function was not programmedfor this application it still works by simply saving the same data twice.
    #Same goes for the plot function
    save_t(index,all_t_prior, all_t_prior, woehler_points, woehler_points)
    a_plot_functions.plot_k(all_t_prior, all_t_prior, 4, 32)     
            
def save_t(index, all_t_prior, all_t_no_prior, woehler_points_prior, woehler_points_no_prior):
    '''
    This function saves the results gathered from the t experiments.

    Parameters
    ----------
    index : Integer
        Index of a dataset (colum number) eg. 0, 1, 2,... .
    all_t_prior : Array
        All Ts from the experiment planning with prior.
    all_t_no_prior : Array
        All Ts from the experiment planning without prior.
    woehler_points_prior : Dict
        A dictonary containing the measured data points from the experiment planning with prior. 
        (s_a, n, outcome)
    woehler_points_no_prior : Dict
        A dictonary containing the measured data points from the experiment planning without prior. 
        (s_a, n, outcome)

    Returns
    -------
    None.

    '''

    with open('all_t_'+str(index)+'_3.pkl', 'wb') as outp:
        pickle.dump(all_t_prior, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_t_no_prior, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(woehler_points_prior, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(woehler_points_no_prior, outp, pickle.HIGHEST_PROTOCOL)
        
def load_t(index):
    '''
    This function loads the results gathered from the t experiments.

    Parameters
    ----------
    index : Integer
        Index of a dataset (colum number) eg. 0, 1, 2,... .

    Returns
    -------
    all_t_prior : Array
        All Ts from the experiment planning with prior.
    all_t_no_prior : Array
        All Ts from the experiment planning without prior.
    woehler_points_prior : Dict
        A dictonary containing the measured data points from the experiment planning with prior. 
        (s_a, n, outcome)
    woehler_points_no_prior : Dict
        A dictonary containing the measured data points from the experiment planning without prior. 
        (s_a, n, outcome)

    '''
    with open('all_t_'+str(index)+'.pkl', 'rb') as inp:
        all_t_prior = pickle.load(inp)
        all_t_no_prior = pickle.load(inp)
        woehler_points_prior = pickle.load(inp)
        woehler_points_no_prior = pickle.load(inp)
    return all_t_prior, all_t_no_prior, woehler_points_prior, woehler_points_no_prior

def simulate_bad_prior(index):
    '''
    This function happens what happens when a "bad prior" exists. For this the prior is mutiplied by a factor.

    Parameters
    ----------
    index : Integer
        Index of a dataset (colum number) eg. 0, 1, 2,... .

    Returns
    -------
    None.

    '''
    t_enough ={}
    t_enough['k'] = 1
    t_enough['s_d'] = 1
    t_enough['n_e'] = 1
    t_enough['one_t_s'] = 1
    factors = [0.5, 1, 2]
    for kk in factors:
        
        for jj in factors:
            factor_mean = kk
            factor_std = jj
            
            new_doe = DOE()
            
            new_data_manager = DataManager()
            new_bays = BaysModel()
            
            new_doe = DOE()
            new_data_manager.load_ki_predictions_from_csv_new(filename = 'ml-prediction.csv',
                                               delimiter = ',')
            db_id = DB_IDS[index]
            n, s_a, outcome = new_data_manager.get_woehler_data_by_db_id(db_id) 

            prior_from_ki = new_data_manager.get_ki_predictions_by_index(index)
            for param in PARAM_LIST:
                prior_from_ki[param] = prior_from_ki[param] * factor_mean
                prior_from_ki[param +'_std'] = prior_from_ki[param +'_std'] * factor_std
            db_id = new_data_manager.get_db_id_by_index(index)
            n, s_a, outcome = new_data_manager.get_woehler_data_by_db_id(db_id)                
            curve = WoehlerCurve(s_a, n, outcome)
            woehler_params_original = new_bays.transform_woehler_params(curve.Mali_4p_result)
            load_levels = new_doe.generate_load_levels(prior_from_ki)
            load_levels_true = new_doe.generate_load_levels(woehler_params_original)
            all_t_no_prior = {}
            all_t_prior = {}
            for ii in range(33):
                all_t_no_prior[ii] ={}
                all_t_prior[ii] ={}
                for param in PARAM_LIST:
                    all_t_no_prior[ii][param] = []
                    all_t_prior[ii][param] = []   
            woehler_points = {}
            woehler_points['n'] = []
            woehler_points['s_a'] = []
            woehler_points['outcome'] = [] 
            load_level_1 = load_levels_true[1]
            load_levels = new_doe.generate_load_levels_2(prior_from_ki, load_levels)
            outcome, n_sample = new_doe.generate_woehler_point(
                woehler_params_original, max(load_levels_true))
            woehler_points['n'] = np.append(woehler_points['n'],n_sample)
            woehler_points['s_a'] = np.append(woehler_points['s_a'],max(load_levels_true))
            woehler_points['outcome'] = np.append(woehler_points['outcome'],outcome)
            outcome, n_sample = new_doe.generate_woehler_point(woehler_params_original, 
                                                               load_level_1)
            woehler_points['n'] = np.append(woehler_points['n'],n_sample)
            woehler_points['s_a'] = np.append(woehler_points['s_a'],load_level_1)
            woehler_points['outcome'] = np.append(woehler_points['outcome'],outcome)
            
            woehler_points['n'] =  np.append(woehler_points['n'],2e7)
            
            woehler_points['s_a'] = np.append(woehler_points['s_a'],
                                            woehler_params_original['s_d'])
            woehler_points['outcome'] = np.append(woehler_points['outcome'],'runout')
            curve = WoehlerCurve(woehler_points['s_a'], woehler_points['n'] , woehler_points['outcome'])
            
            woehler_params_bayes, trace, prior_distribution = new_bays.calc_model(
                woehler_points, num_samples = 1000, prior_from_ki = prior_from_ki)
            
            for parameter in PARAM_LIST:
                hdi_interval = az.hdi(trace[parameter], 0.8)
                t_current = hdi_interval[1] / hdi_interval[0]
                all_t_prior[len(woehler_points['s_a'])][parameter] = \
                    np.append(all_t_prior[len(woehler_points['s_a'])][parameter] ,
                              t_current)
            
            save_storage ={}
            for ii in range(10):#29 The number of experiments can be increased.

                load_level = new_doe.find_best_load_level(
                    trace, curve, load_levels, t_enough)
            
                outcome, n_sample = new_doe.generate_woehler_point(
                    woehler_params_original, load_level)
                woehler_points['n'] = np.append(woehler_points['n'],n_sample)
                woehler_points['s_a'] = np.append(woehler_points['s_a'],load_level)
                woehler_points['outcome'] = np.append(woehler_points['outcome'],outcome)
                curve = WoehlerCurve(woehler_points['s_a'], woehler_points['n'] , woehler_points['outcome'])
                woehler_params_bayes, trace, prior_distribution = new_bays.calc_model(
                    woehler_points, num_samples = 1000, prior_from_ki = prior_from_ki)
                save_storage[ii] ={}
                save_storage[ii]['factor_mean'] = factor_mean
                save_storage[ii]['factor_std'] = factor_std
                save_storage[ii]['trace'] = trace
                save_storage[ii]['curve'] = curve
                save_storage[ii]['load_levels'] = load_levels
                save_storage[ii]['t_enough'] = t_enough
                for parameter in PARAM_LIST:
                    hdi_interval = az.hdi(trace[parameter], 0.8)
                    t_current = hdi_interval[1] / hdi_interval[0]
                    all_t_prior[len(woehler_points['s_a'])][parameter] = \
                        np.append(all_t_prior[len(woehler_points['s_a'])][parameter],t_current)
            #all necessary data is saved. Can get quite a large file
            save_bad_prior(index, factor_mean, factor_std, save_storage)
            
def save_bad_prior(index, factor_mean, factor_std, save_storage):
    '''
    This function saves the results gathered from the "bad prior" experiments.

    Parameters
    ----------
    index : Integer
        Index of a dataset (colum number) eg. 0, 1, 2,... .
    factor_mean : Int
        DESCRIPTION.
    factor_std : Int
        DESCRIPTION.
    save_storage : Dict
        A dictionary containing all the data from the "bad prior" experiment. Can be relativly large.

    Returns
    -------
    None.

    '''
    with open('bad_prior_'+str(index)+'_'+str(factor_mean) +'_' +str(factor_std)
              + '.pkl', 'wb') as outp:
        pickle.dump(save_storage, outp, pickle.HIGHEST_PROTOCOL)

def load_bad_prior(index, factor_mean, factor_std):
    '''
    This function loads the results gathered from the "bad prior" experiments.

    Parameters
    ----------
    index : Integer
        Index of a dataset (colum number) eg. 0, 1, 2,... .
    factor_mean : Int
        The factor by which the mean is shiftet.
    factor_std : Int
        The factor by which the std is shiftet.

    Returns
    -------
    save_storage : Dict
        A dictionary containing all the data from the "bad prior" experiment. Can be relativly large.

    '''
    with open('bad_prior_'+str(index)+'_'+str(factor_mean) +'_' +str(factor_std)
              + '.pkl', 'rb') as inp:
        save_storage = pickle.load(inp)
    
    return save_storage
def calc_t_monte(index, m = 1):
    '''
    This function is used for a monte carlo experiment, where the same woehler params are used. 
    Then fictive experiment results are repetetly generated. 
    From these the fictive woehler params are calculated and from these the spread over these paramters is calculated.
    The staircase methode and the "Perlenschnurverfahren" is used for the planning.
    The fictive data points are returned. m should be increaded to ca. 100 000 for viable results.
    If only the data points are needed m can be set to 1.

    Parameters
    ----------
    index : Integer
        Index of a dataset (colum number) eg. 0, 1, 2,... .
    m : Integer, optional
        Number of monte carlo exeriments done. The default is 1.

    Returns
    -------
    all_t : Dict
        All Ts from the experiment planning calculated using the monte carlo experiment.
    woehler_points : Dict
        A dictonary containing the fictive data points gathered from the planning algorithm.
        (s_a, n, outcome)

    '''
    new_data_manager = DataManager()
    new_bays = BaysModel()
    new_doe = DOE()
    new_data_manager.load_ki_predictions_from_csv_new(filename = 'ml-prediction.csv',
                                                      delimiter = ',')
    new_data_manager.all_ki_predictions
    prior_from_ki = new_data_manager.get_ki_predictions_by_index(index)
    db_id = new_data_manager.get_db_id_by_index(index)
    db_id = DB_IDS[index]
    n, s_a, outcome = new_data_manager.get_woehler_data_by_db_id(db_id)                         
    curve = WoehlerCurve(s_a, n, outcome)
    woehler_params_original = new_bays.transform_woehler_params(curve.Mali_4p_result)
        
    woehler_params_dict = {}
    for param in PARAM_LIST:
        woehler_params_dict[param] ={}
        for ii in range(32):
            woehler_params_dict[param][ii] = []
    load_levels = new_doe.generate_load_levels(woehler_params_original)
    load_levels_new = np.array([560, 520, 480, -1000, -1000, -1000, -1000, -1000, -1000, -1000, 520, -1000, -1000,
                                560, -1000, 560, -1000, -1000, -1000, 600, 600, 600, -1000 ,-1000, 640,
                                640, 640, -1000, 640, -1000, 640, -1000])
    a  = woehler_params_original['s_d']
    b = max(load_levels)
    min_x = 480
    max_x = 640
    load_levels = a + (load_levels_new-min_x)*(b-a)/(max_x-min_x)
    woehler_points = {}
    for kk in range(m):
        last_staircase = -1
        woehler_points = {}
        woehler_points['n'] = []
        woehler_points['s_a'] = []
        woehler_points['outcome'] = []
        
        for ii in range(len(load_levels)):
            
            load_level = load_levels[ii]
            
            if load_level <= 0:
                if last_staircase <=0:
                    last_load_level = woehler_points['s_a'][ii-1]
                    last_outcome = woehler_points['outcome'][ii-1]
                    t = woehler_params_original['one_t_s']
                    load_level = new_doe.find_load_level_staircase(
                        last_load_level, last_outcome, t)
                    last_staircase = ii
                else:   
                    last_load_level = woehler_points['s_a'][last_staircase]
                    last_outcome = woehler_points['outcome'][last_staircase]
                    t = woehler_params_original['one_t_s']
                    load_level = new_doe.find_load_level_staircase(
                        last_load_level, last_outcome, t)
                    last_staircase = ii
            
            outcome, n_sample = new_doe.generate_woehler_point(woehler_params_original,
                                                               load_level)
            woehler_points['n'] = np.append(woehler_points['n'],n_sample)
            woehler_points['s_a'] = np.append(woehler_points['s_a'],load_level)
            woehler_points['outcome'] = np.append(woehler_points['outcome'],outcome)

            try:
                curve = WoehlerCurve(woehler_points['s_a'],
                                     woehler_points['n'],
                                     woehler_points['outcome'])
                woehler_params = new_bays.transform_woehler_params(curve.Mali_4p_result)
                for param in PARAM_LIST:
                    woehler_params_dict[param][len(woehler_points['s_a'])] = \
                        np.append(woehler_params_dict[param][len(woehler_points['s_a'])],
                                  woehler_params[param])
                
            except:
                print('an error occured')
    all_t = {}
    for ii in range(33):
        all_t[ii] ={}
        
        for param in PARAM_LIST:
            all_t[ii][param] = []
     
    for param in PARAM_LIST:
        
        for ii in range(len(woehler_params_dict[param])):
            if len(woehler_params_dict[param][ii]) >= 100:
                hdi_interval = az.hdi(woehler_params_dict[param][ii], 0.8)
                t_current = hdi_interval[1] / hdi_interval[0]
                all_t[ii][param] = t_current
            else:
                all_t[ii][param] = []
    save_t_monte(str(index), all_t)     
    return all_t, woehler_points

def save_t_monte(index, t):
    '''
    This function saves the results gathered from the monte carlo experiments.

    Parameters
    ----------
    index : Integer
        Index of a dataset (colum number) eg. 0, 1, 2,... .
    t : Dict
        All Ts from the experiment planning from the monte carlo experiment.

    Returns
    -------
    None.

    '''
    with open('t_monte'+str(index)+'.pkl', 'wb') as outp:
        pickle.dump(t, outp, pickle.HIGHEST_PROTOCOL)
def load_t_monte(index):
    '''
    This function loads the results gathered from the monte carlo experiments.

    Parameters
    ----------
    index : Integer
        Index of a dataset (colum number) eg. 0, 1, 2,... .

    Returns
    -------
    t : Dict
        All Ts from the experiment planning from the monte carlo experiment.

    '''
    with open('t_monte'+str(index)+'.pkl', 'rb') as inp:
        t = pickle.load(inp)
    return t
    
if __name__ == "__main__":
    __spec__ = None
    freeze_support()
    #The Ts can be calculated as such:
    #index = 1
    #calc_t_standard(index)
    #calc_t_doe(index)
    #The results are saved at the end of the function and can be loaded and thus plotted.
    
    
    
    

