#imports
from bays_model import BaysModel
from data_manager import DataManager as DataManager
from data_manager_OG import DataManager as DataManager2
from doe import DOE
from WoehlerParams import WoehlerCurve
from WoehlerParams2 import WoehlerCurve2
import plot_functions
import plot_functions_T
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import scipy.stats as stats
import pandas as pd
import help_functions
import pickle

#constants
PARAM_LIST = ['k','s_d', 'n_e','one_t_s']

woehler_data = pd.read_csv('new_pred_16.csv')
woehler_data.head()
    
new_data_manager = DataManager()

db_id1 = ['KOL2RNG.29'] #>25
db_id2 = ['DaBef1342']  #<25

new_data_manager.load_ki_predictions_16_from_csv()
index1 = new_data_manager.get_index_from_db_id(db_id1)
index1 = index1[0]
index2 = new_data_manager.get_index_from_db_id(db_id2)
index2 = index2[0]

n1, s_a1, outcome1 = new_data_manager.get_woehler_data_from_index(index1)
curve1 = WoehlerCurve2(s_a1, n1, outcome1)
woehler_params_original1 = new_data_manager.transform_woehler_params(curve1.Mali_4p_result)
n2, s_a2, outcome2 = new_data_manager.get_woehler_data_from_index(index2)
curve2 = WoehlerCurve2(s_a2, n2, outcome2)
woehler_params_original2 = new_data_manager.transform_woehler_params(curve2.Mali_4p_result)


#>25 same mu & std
prior_from_ki1 = new_data_manager.get_wl_ki_predictions_by_index(index1)


#>25 excluding mu
prior_from_ki2 = {}
prior_from_ki2['n_e']     = prior_from_ki1['n_e']*3 + prior_from_ki1['n_e']
prior_from_ki2['n_e_std'] = prior_from_ki1['n_e_std']
prior_from_ki2['k']       = prior_from_ki1['k']*2 + prior_from_ki1['k']
prior_from_ki2['k_std'] = prior_from_ki1['k_std']
prior_from_ki2['one_t_s'] = prior_from_ki1['one_t_s']*0.25 + prior_from_ki1['one_t_s']
prior_from_ki2['one_t_s_std'] = prior_from_ki1['one_t_s_std']
prior_from_ki2['s_d']     = prior_from_ki1['s_d']*0.3 + prior_from_ki1['s_d']
prior_from_ki2['s_d_std'] = prior_from_ki1['s_d_std']
 
#>25 weakly informative
prior_from_ki3 = {}
prior_from_ki3['n_e']     = prior_from_ki1['n_e']
prior_from_ki3['n_e_std'] = prior_from_ki1['n_e_std']*2 + prior_from_ki1['n_e_std']
prior_from_ki3['k']       = prior_from_ki1['k']
prior_from_ki3['k_std'] = prior_from_ki1['k_std']*2 + prior_from_ki1['k_std']
prior_from_ki3['one_t_s'] = prior_from_ki1['one_t_s']
prior_from_ki3['one_t_s_std'] = prior_from_ki1['one_t_s_std']*2 + prior_from_ki1['one_t_s_std']
prior_from_ki3['s_d']     = prior_from_ki1['s_d']
prior_from_ki3['s_d_std'] = prior_from_ki1['s_d_std']*2 + prior_from_ki1['s_d_std']

#>25 on mu
prior_from_ki4 = {}
prior_from_ki4['n_e']     = woehler_params_original1['n_e']
prior_from_ki4['n_e_std'] = prior_from_ki1['n_e_std']
prior_from_ki4['k']       = woehler_params_original1['k']
prior_from_ki4['k_std'] = prior_from_ki1['k_std']
prior_from_ki4['one_t_s'] = woehler_params_original1['one_t_s']
prior_from_ki4['one_t_s_std'] = prior_from_ki1['one_t_s_std']
prior_from_ki4['s_d']     = woehler_params_original1['s_d']
prior_from_ki4['s_d_std'] = prior_from_ki1['s_d_std']

#<25 same mu & std
prior_from_ki5 = new_data_manager.get_wl_ki_predictions_by_index(index2)


#<25 excluding mu 
prior_from_ki6 = {}
prior_from_ki6['n_e']     = -prior_from_ki5['n_e']*3 + prior_from_ki5['n_e']
prior_from_ki6['n_e_std'] = prior_from_ki5['n_e_std']
prior_from_ki6['k']       = -prior_from_ki5['k']*0.3 - prior_from_ki5['k']
prior_from_ki6['k_std'] = prior_from_ki5['k_std']
prior_from_ki6['one_t_s'] = -prior_from_ki5['one_t_s']*0.5 + prior_from_ki5['one_t_s']
prior_from_ki6['one_t_s_std'] = prior_from_ki5['one_t_s_std']
prior_from_ki6['s_d']     = -prior_from_ki5['s_d']*0.8 + prior_from_ki5['s_d']
prior_from_ki6['s_d_std'] = prior_from_ki5['s_d_std']
 
#<25 weakly informative
prior_from_ki7 = {}
prior_from_ki7['n_e']     = prior_from_ki5['n_e']
prior_from_ki7['n_e_std'] = prior_from_ki5['n_e_std']*2 + prior_from_ki5['n_e_std']
prior_from_ki7['k']       = prior_from_ki5['k']
prior_from_ki7['k_std'] = prior_from_ki5['k_std']*2 + prior_from_ki5['k_std']
prior_from_ki7['one_t_s'] = prior_from_ki5['one_t_s']
prior_from_ki7['one_t_s_std'] = prior_from_ki5['one_t_s_std']*2 + prior_from_ki5['one_t_s_std']
prior_from_ki7['s_d']     = prior_from_ki5['s_d']
prior_from_ki7['s_d_std'] = prior_from_ki5['s_d_std']*2 + prior_from_ki5['s_d_std']

#<25 on mu
prior_from_ki8 = {}
prior_from_ki8['n_e']     = woehler_params_original2['n_e']
prior_from_ki8['n_e_std'] = prior_from_ki5['n_e_std']
prior_from_ki8['k']       = woehler_params_original2['k']
prior_from_ki8['k_std'] = prior_from_ki5['k_std']
prior_from_ki8['one_t_s'] = woehler_params_original2['one_t_s']
prior_from_ki8['one_t_s_std'] = prior_from_ki5['one_t_s_std']
prior_from_ki8['s_d']     = woehler_params_original2['s_d']
prior_from_ki8['s_d_std'] = prior_from_ki5['s_d_std']

woehler_points1 = {}
woehler_points1['s_a'] = s_a1
woehler_points1['outcome'] = outcome1
woehler_points1['n'] = n1

woehler_points2 = {}
woehler_points2['s_a'] = s_a2
woehler_points2['outcome'] = outcome2
woehler_points2['n'] = n2


trace1 = []
trace2 = []
trace3 = []
trace4 = []
trace5 = []
trace6 = []
trace7 = []
trace8 = []

t_enough = {}

woehler_points1 = {}
woehler_points1['s_a'] = []
woehler_points1['n'] = []
woehler_points1['outcome'] = []

woehler_points2 = {}
woehler_points2['s_a'] = []
woehler_points2['n'] = []
woehler_points2['outcome'] = []

all_t_prior = {}
all_t_prior1 = {}
all_t_prior2 = {}
all_t_prior3 = {}
all_t_prior4 = {}
all_t_prior5 = {}
all_t_prior6 = {}
all_t_prior7 = {}
all_t_prior8 = {}

for i in range (8):
    if i <= 3:
        index = index1
    else:
        index = index2
    print(woehler_data['N_Versuche'][index])
    for ii in range(woehler_data['N_Versuche'][index]):
        all_t_prior[ii] = {}
        for param in PARAM_LIST:
            all_t_prior[ii][param] = []

    if i == 0:
        all_t_prior1 = all_t_prior
    if i == 1:
        all_t_prior2 = all_t_prior
    if i == 2:
        all_t_prior3 = all_t_prior
    if i == 3:
        all_t_prior4 = all_t_prior
    if i == 4:
        all_t_prior5 = all_t_prior
    if i == 5:
        all_t_prior6 = all_t_prior
    if i == 6:
        all_t_prior7 = all_t_prior
    if i == 7:
        all_t_prior8 = all_t_prior
    all_t_prior = {}
    
print(all_t_prior)


for i in range(woehler_data['N_Versuche'][index1]):
    new_bays_model = BaysModel()
    woehler_params_bayes1, trace1, prior_distribution1 = new_bays_model.calc_model(woehler_points1,num_samples=1000,prior_from_ki=prior_from_ki1)
    woehler_points1['s_a'] = np.append(woehler_points1['s_a'],s_a1[i])
    woehler_points1['n'] = np.append(woehler_points1['n'],n1[i])
    woehler_points1['outcome'] = np.append(woehler_points1['outcome'],outcome1[i])
    for parameter in PARAM_LIST:
        hdi_interval = az.hdi(trace1[parameter], 0.8)
        t_current = hdi_interval[1] / hdi_interval[0]
        all_t_prior1[i][parameter] = np.append(all_t_prior1[i][parameter] ,t_current)

for i in range(woehler_data['N_Versuche'][index1]):
    new_bays_model = BaysModel()
    woehler_params_bayes2, trace2, prior_distribution2 = new_bays_model.calc_model(woehler_points1,num_samples=1000,prior_from_ki=prior_from_ki2)
    woehler_points1['s_a'] = np.append(woehler_points1['s_a'],s_a1[i])
    woehler_points1['n'] = np.append(woehler_points1['n'],n1[i])
    woehler_points1['outcome'] = np.append(woehler_points1['outcome'],outcome1[i])
    for parameter in PARAM_LIST:
        hdi_interval = az.hdi(trace2[parameter], 0.8)
        t_current = hdi_interval[1] / hdi_interval[0]
        all_t_prior2[i][parameter] = np.append(all_t_prior2[i][parameter] ,t_current)
        
for i in range(woehler_data['N_Versuche'][index1]):
    new_bays_model = BaysModel()
    woehler_params_bayes3, trace3, prior_distribution3 = new_bays_model.calc_model(woehler_points1,num_samples=1000,prior_from_ki=prior_from_ki3)
    woehler_points1['s_a'] = np.append(woehler_points1['s_a'],s_a1[i])
    woehler_points1['n'] = np.append(woehler_points1['n'],n1[i])
    woehler_points1['outcome'] = np.append(woehler_points1['outcome'],outcome1[i])
    for parameter in PARAM_LIST:
        hdi_interval = az.hdi(trace3[parameter], 0.8)
        t_current = hdi_interval[1] / hdi_interval[0]
        all_t_prior3[i][parameter] = np.append(all_t_prior3[i][parameter] ,t_current)

for i in range(woehler_data['N_Versuche'][index1]):
    new_bays_model = BaysModel()
    woehler_params_bayes4, trace4, prior_distribution4 = new_bays_model.calc_model(woehler_points1,num_samples=1000,prior_from_ki=prior_from_ki4)
    woehler_points1['s_a'] = np.append(woehler_points1['s_a'],s_a1[i])
    woehler_points1['n'] = np.append(woehler_points1['n'],n1[i])
    woehler_points1['outcome'] = np.append(woehler_points1['outcome'],outcome1[i])
    for parameter in PARAM_LIST:
        hdi_interval = az.hdi(trace3[parameter], 0.8)
        t_current = hdi_interval[1] / hdi_interval[0]
        all_t_prior4[i][parameter] = np.append(all_t_prior4[i][parameter] ,t_current)

for i in range(woehler_data['N_Versuche'][index2]):
    new_bays_model = BaysModel()
    woehler_params_bayes5, trace5, prior_distribution5 = new_bays_model.calc_model(woehler_points2,num_samples=1000,prior_from_ki=prior_from_ki5)
    woehler_points2['s_a'] = np.append(woehler_points2['s_a'],s_a2[i])
    woehler_points2['n'] = np.append(woehler_points2['n'],n2[i])
    woehler_points2['outcome'] = np.append(woehler_points2['outcome'],outcome2[i])
    for parameter in PARAM_LIST:
        hdi_interval = az.hdi(trace5[parameter], 0.8)
        t_current = hdi_interval[1] / hdi_interval[0]
        all_t_prior5[i][parameter] = np.append(all_t_prior5[i][parameter] ,t_current)

for i in range(woehler_data['N_Versuche'][index2]):
    new_bays_model = BaysModel()
    woehler_params_bayes6, trace6, prior_distribution6 = new_bays_model.calc_model(woehler_points2,num_samples=1000,prior_from_ki=prior_from_ki6)
    woehler_points2['s_a'] = np.append(woehler_points2['s_a'],s_a2[i])
    woehler_points2['n'] = np.append(woehler_points2['n'],n2[i])
    woehler_points2['outcome'] = np.append(woehler_points2['outcome'],outcome2[i])
    for parameter in PARAM_LIST:
        hdi_interval = az.hdi(trace6[parameter], 0.8)
        t_current = hdi_interval[1] / hdi_interval[0]
        all_t_prior6[i][parameter] = np.append(all_t_prior6[i][parameter] ,t_current)
        
for i in range(woehler_data['N_Versuche'][index2]):
    new_bays_model = BaysModel()
    woehler_params_bayes7, trace7, prior_distribution7 = new_bays_model.calc_model(woehler_points2,num_samples=1000,prior_from_ki=prior_from_ki7)
    woehler_points2['s_a'] = np.append(woehler_points2['s_a'],s_a2[i])
    woehler_points2['n'] = np.append(woehler_points2['n'],n2[i])
    woehler_points2['outcome'] = np.append(woehler_points2['outcome'],outcome2[i])
    for parameter in PARAM_LIST:
        hdi_interval = az.hdi(trace7[parameter], 0.8)
        t_current = hdi_interval[1] / hdi_interval[0]
        all_t_prior7[i][parameter] = np.append(all_t_prior7[i][parameter] ,t_current)
        
for i in range(woehler_data['N_Versuche'][index2]):
    new_bays_model = BaysModel()
    woehler_params_bayes8, trace8, prior_distribution8 = new_bays_model.calc_model(woehler_points2,num_samples=1000,prior_from_ki=prior_from_ki8)
    woehler_points2['s_a'] = np.append(woehler_points2['s_a'],s_a2[i])
    woehler_points2['n'] = np.append(woehler_points2['n'],n2[i])
    woehler_points2['outcome'] = np.append(woehler_points2['outcome'],outcome2[i])
    for parameter in PARAM_LIST:
        hdi_interval = az.hdi(trace8[parameter], 0.8)
        t_current = hdi_interval[1] / hdi_interval[0]
        all_t_prior8[i][parameter] = np.append(all_t_prior8[i][parameter] ,t_current)


with open('all_t_1.pkl', 'wb') as outp1:
    pickled_trace1 = pickle.dump(trace1,outp1)
    pickled_all_t_prior1 = pickle.dump(all_t_prior1,outp1)
    pickled_trace2 = pickle.dump(trace2,outp1)
    pickled_all_t_prior2 = pickle.dump(all_t_prior2,outp1)
    pickled_trace3 = pickle.dump(trace3,outp1)
    pickled_all_t_prior3 = pickle.dump(all_t_prior3,outp1)
    pickled_trace4 = pickle.dump(trace4,outp1)
    pickled_all_t_prior4 = pickle.dump(all_t_prior4,outp1)

with open('all_t_2.pkl', 'wb') as outp2:    
    pickled_trace5 = pickle.dump(trace5,outp2)
    pickled_all_t_prior5 = pickle.dump(all_t_prior5,outp2)
    pickled_trace6 = pickle.dump(trace6,outp2)
    pickled_all_t_prior6 = pickle.dump(all_t_prior6,outp2)
    pickled_trace7 = pickle.dump(trace7,outp2)
    pickled_all_t_prior7 = pickle.dump(all_t_prior7,outp2)
    pickled_trace8 = pickle.dump(trace8,outp2)
    pickled_all_t_prior8 = pickle.dump(all_t_prior8,outp2)