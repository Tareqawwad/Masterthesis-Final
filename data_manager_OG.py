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
        self.load_woehler_data_from_csv()
        self.load_ki_predictions_from_csv()
        self.woehler_data_sorted_by_number_of_runs = \
            self.get_woehler_data_sorted_by_number_of_runs()
        
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
        
    def load_ki_predictions_from_csv(self, filename = 'Pred.csv', delimiter = ';'):
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
            The default is 'Pred.csv'.
        delimiter : String, optional
            is the delimiter that seperates the entries of the csv. The default is ';'.

        Returns
        -------
        None.

        '''
        self.all_ki_predictions = pd.read_csv(filename, delimiter = delimiter)
        #some renames of certain colums
        self.all_ki_predictions.rename(columns = {'NumInDB':'db_id'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'1/T':'one_t_s'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'1/T_StD':'one_t_s_std'}, inplace = True)
    def load_ki_predictions_from_csv_new(self, filename = 'Pred.csv', delimiter = ';'):
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
        self.all_ki_predictions.rename(columns = {'Unnamed: 0':'db_id'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'1Ts':'one_t_s'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'1Ts_Var':'one_t_s_std'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'Sd_Var':'Sd_StD'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'k_Var':'k_StD'}, inplace = True)
        self.all_ki_predictions.rename(columns = {'NE_Var':'NE_StD'}, inplace = True)
        
        self.all_ki_predictions.NE_StD = np.sqrt(self.all_ki_predictions.NE_StD)
        self.all_ki_predictions.k_StD = np.sqrt(self.all_ki_predictions.k_StD)
        self.all_ki_predictions.one_t_s_std = np.sqrt(self.all_ki_predictions.one_t_s_std)
        self.all_ki_predictions.Sd_StD = np.sqrt(self.all_ki_predictions.Sd_StD)
        
    def get_ki_predictions_by_index(self, index_for_db):
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
        ki_predictions['n_e'] = self.all_ki_predictions.NE[index_for_db]
        ki_predictions['n_e_std'] = self.all_ki_predictions.NE_StD[index_for_db]
        ki_predictions['k'] = self.all_ki_predictions.k[index_for_db]
        ki_predictions['k_std'] = self.all_ki_predictions.k_StD[index_for_db]
        
        ki_predictions['one_t_s'] = self.all_ki_predictions.one_t_s[index_for_db]
        ki_predictions['one_t_s_std'] = self.all_ki_predictions.one_t_s_std[index_for_db]
        
        ki_predictions['s_d'] = self.all_ki_predictions.Sd[index_for_db]
        ki_predictions['s_d_std'] = self.all_ki_predictions.Sd_StD[index_for_db]
        return ki_predictions
    
    def get_db_id_by_index(self, index_for_db):
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
        
    def get_woehler_data_by_index(self, index_for_db):
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
        n = self.woehler_data.N[index_for_db]
        n =  n.strip('[]')
        n = np.fromstring(n, sep=',')
        s_a = self.woehler_data.Sa[index_for_db]
        s_a = s_a.strip('[]')
        s_a = np.fromstring(s_a, sep=',')
        outcome = self.woehler_data.outcome[index_for_db]
        outcome = outcome.replace("'",'')
        outcome = outcome.replace(" ",'')
        outcome = outcome.strip("'[]")
        outcome = outcome.split(',')
        return n, s_a, outcome
    
    def get_woehler_data_by_db_id(self, db_id_for_db):
        '''
        returns the woehler data of a dataset. The dataste is specified by the
        DB ID. 
        The woehler data include the number of cycles n, the amplitudes and the
        outocomes of the old experiemts stored in the DB.
        NOTE: Function similar to get_woehler_data_by_index
        
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
        for ii in range(self.woehler_data.db_id.index.stop):
            if self.woehler_data.db_id[ii] == db_id_for_db:
                index_for_db = ii
                break
        if index_for_db != []:
            n, s_a, outcome = self.get_woehler_data_by_index(index_for_db)
            return n, s_a, outcome
        else:
            n =[]
            s_a = []
            outcome =[]
            return n, s_a, outcome
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
    