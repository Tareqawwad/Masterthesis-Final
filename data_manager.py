# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:58:19 2021

@author: Sven Mordeja
"""

#imports
import numpy as np
import pandas as pd
from multiprocessing import Process, freeze_support

class DataManager:
    def __init__(self):
        '''
        The DataManager class is responsible for all interactions with the 
        databases (csvs'). This includes e.g. loading the preknowledge and
        experiment data from e.g. NIST or DaBef

        Returns
        -------
        None.

        '''
        self.PARAM_LIST = ['k', 's_d', 'n_e', 'one_t_s']
        #self.load_woehler_data_from_csv()
        #self.load_ki_predictions_from_csv()
        #self.woehler_data_sorted_by_number_of_runs = \
        #    self.get_woehler_data_sorted_by_number_of_runs()
        
    def load_woehler_data_from_csv(self, filename = 'allData.csv'):
        '''
        Loads all data from the database and stores it in self.woehler_data

        Parameters
        ----------
        filename : String, optional
            Name of the csv database. The default is 'allData.csv'.

        Returns
        -------
        None.

        '''
        self.woehler_data = pd.read_csv(filename)
        self.woehler_data.rename(columns = {'Unnamed: 0':'db_id'}, inplace = True)
        
    def load_ki_predictions_32_from_csv(self, filename = 'new_pred_32.csv'):
        '''
        Loads all KI-predictions from the database and stores it in 
        self.all_ki_predictions.
        NOTE: There are two versions in which the file comes. Both are supported
        by this file. See:load_ki_predictions_from_csv_new
        The biggest differnce is that in one the variance is stored in the other
        the standard deviation. In this verison the std is stored in the file.

        Parameters
        ----------
        filename : String, optional
            Specifies the filename of the csv that stores the KI-predictions.
            The default is 'new_pred_32.csv'.
        delimiter : String, optional
            is the delimiter that seperates the entries of the csv. The default is ';'.

        Returns
        -------
        None.

        '''
        self.all_ki_predictions = pd.read_csv(filename, index_col=False)
        self.all_ki_predictions.drop(columns=['Unnamed: 0'], inplace=True)
        #some renames of certain colums
        self.all_ki_predictions.rename(columns = {'dataset_names':'db_id'}, inplace = True)    

        self.all_ki_predictions.rename(columns = {'std_test_c_1Ts':'oneT_Std_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'std_test_c_Sd':'Sd_Std_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'std_test_c_k':'k_Std_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'std_test_c_NE':'NE_Std_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'std_test_c_ll_sd_095':'ll_sd_095_Std_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'std_test_c_ll_10e4':'ll_10e4_Std_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'std_test_c_max_s_a':'ll_max_s_a_Std_ki'}, inplace = True)
        
        
        self.all_ki_predictions.rename(columns = {'mean_test_1Ts':'oneT_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'mean_test_Sd':'Sd_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'mean_test_k':'k_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'mean_test_NE':'NE_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'mean_test_ll_sd_095':'ll_sd_095_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'mean_test_ll_10e4':'ll_10e4_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'mean_test_max_s_a':'ll_max_s_a_ki'}, inplace = True)
        
    def load_ki_predictions_16_from_csv(self, filename = 'new_pred_16.csv'):
        '''
        Loads all KI-predictions from the database and stores it in 
        self.all_ki_predictions.
        NOTE: There are two versions in which the file comes. Both are supported
        by this file. See:load_ki_predictions_from_csv_new
        The biggest differnce is that in one the variance is stored in the other
        the standard deviation. In this verison the std is stored in the file.

        Parameters
        ----------
        filename : String, optional
            Specifies the filename of the csv that stores the KI-predictions.
            The default is 'new_pred_16.csv'.
        delimiter : String, optional
            is the delimiter that seperates the entries of the csv. The default is ';'.

        Returns
        -------
        None.

        '''
        self.all_ki_predictions = pd.read_csv(filename, index_col=False)
        self.all_ki_predictions.drop(columns=['Unnamed: 0'], inplace=True)
        #some renames of certain colums
        self.all_ki_predictions.rename(columns = {'dataset_names':'db_id'}, inplace = True)    

        self.all_ki_predictions.rename(columns = {'std_test_c_1Ts':'oneT_Std_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'std_test_c_Sd':'Sd_Std_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'std_test_c_k':'k_Std_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'std_test_c_NE':'NE_Std_ki'}, inplace = True)
                
        self.all_ki_predictions.rename(columns = {'mean_test_1Ts':'oneT_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'mean_test_Sd':'Sd_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'mean_test_k':'k_ki'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'mean_test_NE':'NE_ki'}, inplace = True)    
        
    def load_ki_predictions_from_csv_old(self, filename = 'Pred.csv', delimiter = ','):
        '''
        Loads all KI-predictions from the database and stores it in 
        self.all_ki_predictions. 
        NOTE: There are two versions in which the file comes. Both are supported
        by this file. See:load_ki_predictions_from_csv
        The biggest differnce is that in one the variance is stored in the other
        the standard deviation. In this verison the var is stored in the file.
        
        Parameters
        ----------
        filename : String, optional
            Specifies the filename of the csv that stores the KI-predictions.
            The default is 'Pred.csv'.
        delimiter : String, optional
            is the delimiter that seperates the entries of the csv. The default is ';'.

        Returns
        -------
        None.

        '''
        self.all_ki_predictions = pd.read_csv(filename, delimiter = delimiter)
        self.all_ki_predictions.rename(columns = {'dataset_names':'db_id'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'1Ts':'one_t_s'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'1Ts_Var':'one_t_s_std'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'Sd_Var':'Sd_StD'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'k_Var':'k_StD'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'NE_Var':'NE_StD'}, inplace = True)
        
        self.all_ki_predictions.NE_StD = np.sqrt(self.all_ki_predictions.NE_StD)
        self.all_ki_predictions.k_StD = np.sqrt(self.all_ki_predictions.k_StD)
        self.all_ki_predictions.one_t_s_std = np.sqrt(self.all_ki_predictions.one_t_s_std)
        self.all_ki_predictions.Sd_StD = np.sqrt(self.all_ki_predictions.Sd_StD)
        
    def get_wl_ki_predictions_by_index(self, index_for_db):
        '''
        Returs the KI predicions for a dataset in the database. The dataset is 
        specified by the index (colum number).

        Parameters
        ----------
        index_for_db : Integer
            Index of a dataset(colum number) eg. 0, 1, 2,... .

        Returns
        -------
        ki_predictions : Dict
            A dictionary of the KI predictions. In the dictionary the mean and 
            the std are saved.

        '''
        ki_predictions = {}
        ki_predictions['k'] = self.all_ki_predictions.k_ki[index_for_db]
        ki_predictions['k_std'] = self.all_ki_predictions.k_Std_ki[index_for_db]
        
        ki_predictions['s_d'] = self.all_ki_predictions.Sd_ki[index_for_db]
        ki_predictions['s_d_std'] = self.all_ki_predictions.Sd_Std_ki[index_for_db]
        
        ki_predictions['n_e'] = self.all_ki_predictions.NE_ki[index_for_db]
        ki_predictions['n_e_std'] = self.all_ki_predictions.NE_Std_ki[index_for_db]

        ki_predictions['one_t_s'] = self.all_ki_predictions.oneT_ki[index_for_db]
        ki_predictions['one_t_s_std'] = self.all_ki_predictions.oneT_Std_ki[index_for_db]
        
        
        
        
        #ki_predictions['Rm'] = self.all_ki_predictions.Rm[index_for_db]
        
        return ki_predictions
        
    def get_fkm_predictions_by_index(self, index_for_db):
        '''
        Returns the FKM predicion and Tensile Strength (RM) for a dataset in the database. The dataset is 
        specified by the index (colum number).

        Parameters
        ----------
        index_for_db : Integer
            Index of a dataset(colum number) eg. 0, 1, 2,... .

        Returns
        -------
        FKM_predictions : Dict
            A dictionary of the FKM prediction for SD. In the dictionary the mean and 
            the std are saved.

        '''
        fkm_predictions = {}
        fkm_predictions['SDFKM_lokal'] = self.all_ki_predictions.SDFKM_lokal[index_for_db]
        fkm_predictions['Rm'] = self.all_ki_predictions.Rm[index_for_db]

        return fkm_predictions
        
        
    def get_ll_sd_orig_predictions_by_index(self, index_for_db):
    
        '''
        Returns the FKM predicion and Tensile Strength (RM) for a dataset in the database. The dataset is 
        specified by the index (colum number).

        Parameters
        ----------
        index_for_db : Integer
            Index of a dataset(colum number) eg. 0, 1, 2,... .

        Returns
        -------
        FKM_predictions : Dict
            A dictionary of the FKM prediction for SD. In the dictionary the mean and 
            the std are saved.

        '''
        
        ll_predictions = {}
        ll_predictions['ll_sd_095_ki'] = self.all_ki_predictions.mean_test_Sd_orig_ll_sd_095[index_for_db]
        ll_predictions['ll_sd_095_Std_ki'] = self.all_ki_predictions.std_test_c_Sd_orig_ll_sd_095[index_for_db]
        ll_predictions['ll_10e4_ki'] = self.all_ki_predictions.mean_test_Sd_orig_ll_10e4[index_for_db]
        ll_predictions['ll_10e4_Std_ki'] = self.all_ki_predictions.std_test_c_Sd_orig_ll_10e4[index_for_db]
        ll_predictions['ll_max_s_a_ki'] = self.all_ki_predictions.mean_test_Sd_orig_max_s_a[index_for_db]
        ll_predictions['ll_max_s_a_Std_ki'] = self.all_ki_predictions.std_test_c_Sd_orig_max_s_a[index_for_db]

        return ll_predictions
        
    def get_ll_sd_pred_predictions_by_index(self, index_for_db):
    
        '''
        Returns the FKM predicion and Tensile Strength (RM) for a dataset in the database. The dataset is 
        specified by the index (colum number).

        Parameters
        ----------
        index_for_db : Integer
            Index of a dataset(colum number) eg. 0, 1, 2,... .

        Returns
        -------
        FKM_predictions : Dict
            A dictionary of the FKM prediction for SD. In the dictionary the mean and 
            the std are saved.

        '''
        
        ll_predictions = {}
        ll_predictions['ll_sd_095_ki'] = self.all_ki_predictions.mean_test_Sd_pred_ll_sd_095[index_for_db]
        ll_predictions['ll_sd_095_Std_ki'] = self.all_ki_predictions.std_test_c_Sd_pred_ll_sd_095[index_for_db]
        ll_predictions['ll_10e4_ki'] = self.all_ki_predictions.mean_test_Sd_pred_ll_10e4[index_for_db]
        ll_predictions['ll_10e4_Std_ki'] = self.all_ki_predictions.std_test_c_Sd_pred_ll_10e4[index_for_db]
        ll_predictions['ll_max_s_a_ki'] = self.all_ki_predictions.mean_test_Sd_pred_max_s_a[index_for_db]
        ll_predictions['ll_max_s_a_Std_ki'] = self.all_ki_predictions.std_test_c_Sd_pred_max_s_a[index_for_db]

        return ll_predictions
        
    
    def get_ll_true_values_by_index(self, index_for_db):
    
        '''
        Returns the FKM predicion and Tensile Strength (RM) for a dataset in the database. The dataset is 
        specified by the index (colum number).

        Parameters
        ----------
        index_for_db : Integer
            Index of a dataset(colum number) eg. 0, 1, 2,... .

        Returns
        -------
        FKM_predictions : Dict
            A dictionary of the FKM prediction for SD. In the dictionary the mean and 
            the std are saved.

        '''
        
        ll_true = {}
        ll_true['ll_sd_095'] = self.all_ki_predictions.ll_sd_095[index_for_db]
        ll_true['ll_10e4'] = self.all_ki_predictions.ll_10e4[index_for_db]
        ll_true['ll_max_s_a'] = self.all_ki_predictions.ll_max_s_a[index_for_db]

        return ll_true
        
        
    def get_Sd_Rm_by_index(self, index_for_db):
        '''
        Returns the measured Sd and Tensile Strength (RM) for a dataset in the database. One can show the differenceto fkm Prediction,
        in case FKM is more accurate. The dataset is specified by the index (colum number).
        

        Parameters
        ----------
        index_for_db : Integer
            Index of a dataset(colum number) eg. 0, 1, 2,... .

        Returns
        -------
        FKM_predictions : Dict
            A dictionary of the FKM prediction for SD. In the dictionary the mean and 
            the std are saved.

        '''
        fkm_predictions = {}
        fkm_predictions['Sd_Kt'] = self.all_ki_predictions.Sd_Kt[index_for_db]
        fkm_predictions['Rm'] = self.all_ki_predictions.Rm[index_for_db]

        return fkm_predictions
        
    def get_woehler_data_from_index(self, index_for_db):
        '''
        returns the woehler data of a dataset. The dataste is specified by the
        index (colum number). 
        The woehler data include the number of cycles n, the amplitudes and the
        outocomes of the old experiemts stored in the DB.
        NOTE: Function similar to get_woehler_data_by_db_id
        
        Parameters
        ----------
        index_for_db : Int
            Index of a dataset(colum number) eg. 0, 1, 2,... .

        Returns
        -------
        n : array
            Number of cycles n of the old experiemts stored in the DB.
        s_a : array
            Amplitudes of the old experiemts stored in the DB.
        outcome : array
            Outocomes of the old experiemts stored in the DB.

        '''
        n = self.all_ki_predictions.N[index_for_db]
        n =  n.strip('[]')
        n = np.fromstring(n, sep=',')
        s_a = self.all_ki_predictions.Sa[index_for_db]
        s_a = s_a.strip('[]')
        s_a = np.fromstring(s_a, sep=',')
        outcome = self.all_ki_predictions.outcome[index_for_db]
        outcome = outcome.replace("'",'')
        outcome = outcome.replace(" ",'')
        outcome = outcome.strip("'[]")
        outcome = outcome.split(',')
        # we are using local fkm --> nominal * kt 
        kt = self.all_ki_predictions.Kt[index_for_db]
        
        return n, s_a*kt, outcome
        
    def get_woehler_data_from_db_id(self, db_id_for_db):
        '''
        returns the woehler data of a dataset. The dataste is specified by the
        DB ID. 
        The woehler data include the number of cycles n, the amplitudes and the
        outocomes of the old experiemts stored in the DB.
        NOTE: Function similar to get_woehler_data_from_index
        
        Parameters
        ----------
        db_id_for_db : Str
            ID of the dataset e.g.: NIMS1 or DaBef1.
        
        Returns
        -------
        n : array
            Number of cycles n of the old experiemts stored in the DB.
        s_a : array
            Amplitudes of the old experiemts stored in the DB.
        outcome : array
            Outocomes of the old experiemts stored in the DB.
        '''
        index_for_db = []
        for ii in range(self.all_ki_predictions.db_id.index.stop):
            if self.all_ki_predictions.db_id[ii] == db_id_for_db:
                index_for_db = ii
                break
        if index_for_db != []:
            n, s_a, outcome = self.get_woehler_data_from_index(index_for_db)
            # as we are using FKM local = FKM nominal * Kt, all results have to be multiplied by Kt 
            # --> only Sa changes.. everything else stays the same
            kt = self.all_ki_predictions.Kt[index_for_db]
            return n, s_a*kt, outcome
        else:
            n =[]
            s_a = []
            outcome =[]
            return n, s_a, outcome
        
    def get_db_id_from_index(self, index_for_db):
        '''
        returns the DB ID of a dataset index.
        

        Parameters
        ----------
        index_for_db : Int
            Index of a dataset(colum number) eg. 0, 1, 2,... .

        Returns
        -------
        db_id : Str
            ID of the dataset e.g.: NIMS1 or DaBef1.

        '''
        db_id = self.all_ki_predictions.db_id[index_for_db]
        return db_id
        
    def get_index_from_db_id(self, db_id_for_db):    
        """
        returns index of a list of DBnames
        
        Parameters
        ----------
        db_id_for_db : list of str
            List of ID of the dataset e.g.: [NIMS1, Dabef1]
            
        Returns
        -------
        ndarray: with index
        
        """
    
        filtered_df = self.all_ki_predictions[self.all_ki_predictions['db_id'].isin(db_id_for_db)]
        
        index = filtered_df.index
        index_list = []
        for i in index:
            index_list.append(i)
        return index_list
        
    def filter_pred_for_steel(self, steeltype):
        """
        returns pandas dataframe filtered for specific steel type
        
        Parameters
        ----------
        steeltype : str
            Most be one of the following steel types: 'Steel', 'CaseHard_Steel' or 'Stainless_Steel'
            
        Returns
        -------
        pandas dataframe filtered: 
        
        """       
        
    
        return self.all_ki_predictions[self.all_ki_predictions['MatGroupFKM']==steeltype]  
            
    def get_woehler_data_sorted_by_number_of_runs(self):
        '''
        returns all woehler data sorted by the number of data points. This is
        useful as some datasets have very few data points and are therefore of
        little us in the context of these scripts.

        Returns
        -------
        woehler_data_sorted_by_number_of_runs : DataFrame
            A dataframe of all current data sets sorted by the number of data 
            points 

        '''
        woehler_data_sorted_by_number_of_runs = self.woehler_data
        woehler_data_sorted_by_number_of_runs = woehler_data_sorted_by_number_of_runs.reindex(
            woehler_data_sorted_by_number_of_runs.N.str.len().sort_values(
                ascending=False).index)
        return woehler_data_sorted_by_number_of_runs
    
    def transform_woehler_params(self, woehler_params):
        '''
        This function transforms the format of the woehler_params dictionary.
        The Max-Likelihood was coded by Kolyshkin and therefore uses a differnt
        format. The variables have differnd names and are changed into the format
        used in this script.
    
        Parameters
        ----------
        woehler_params : Dict
            Dictionary of the four woehler parameters as saved by the
            WoehlerParams script.

        Returns
        -------
        woehler_params_transformed : Dict
            Dictionary of the four woehler parameters as used in this script.

        '''
        woehler_params_transformed = {}
        woehler_params_transformed['k'] = woehler_params['k_1']
        woehler_params_transformed['s_d'] = woehler_params['SD_50']
        woehler_params_transformed['n_e'] = woehler_params['ND_50']
        woehler_params_transformed['one_t_s'] = woehler_params['1/TS']
        
        return woehler_params_transformed
if __name__ == "__main__":
    #for multiproccesing
    freeze_support()
    #example to get some woehler data from the csv
    dataManager = DataManager()
    n,s_a, outcome = dataManager.get_woehler_data_by_index(0)
    n,s_a, outcome = dataManager.get_woehler_data_by_db_id('DaBef113')
    