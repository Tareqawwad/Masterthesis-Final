# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 14:06:39 2021

@author: Sven Mordeja
"""
import arviz as az
import help_functions
import numpy as np
import random
import statistics
import time
from bays_model import BaysModel
from data_manager import DataManager
from WoehlerParams import WoehlerCurve

from multiprocessing import Process, freeze_support
import scipy.stats as stats


class DOE:
    def __init__(self):
        '''
        In this class is used for the experiment planning.

        Returns
        -------
        None.

        '''
        self.PARAM_LIST = ['k', 's_d', 'n_e', 'one_t_s']
        self.MAX_N = 2e7
        self.params_not_enough = ['k', 's_d', 'n_e', 'one_t_s']
        self.params_not_enough_all = ['k', 's_d', 'n_e','one_t_s']
        self.bays_model = BaysModel()
        self.data_manager = DataManager()
    #%% functions for the planning algorithm
    def plan_next_experiment(self, trace, woehler_points, ki_predictions, t_enough):
        '''
        This function is used to compute the next load level.
        If enough data points are available find_best_load_level
        is called. Note that the actual test planning algorithm
        starts after two data points are availible.

        Parameters
        ----------
        trace : MultiTrace
            The MultiTrace object containing the samples from
            the posterior. The samples for a parameter can be 
            accessed similar to a dictionary by giving the parameter
            name as a key: trace['n_e']
        woehler_points : Dict
            A dictonary containing the measured data points. 
            (s_a, n, outcome)
        ki_predictions : Dict, optional
            This dictionary contains the prior knowledge. For
            every parameter the mean and the std deviation is 
            given. When None is given a uniform distribution is
            automatically choosen. The default is None.
        t_enough : dict
            A dictionary that defines how accurate the spread
            (Streuspanne) for every parameter should be. If 
            that accuray is reached this parameter is not considered
            for the planning. Thus, the optimal load level
            only depend on the other parameters.

        Returns
        -------
        load_level : Int
            The aplitude for the next expeiment.

        '''
        
        s_a_failure, n_failure, s_a_runout, n_runout  = \
            help_functions.get_runout_and_failure(woehler_points)
        
        #since the actual alogithem riqures a certain amount of data points,
        #a case selection is made and the first two load levels are choosen
        #empirically.
        if len(s_a_failure) == 0:
            load_levels= self.generate_load_levels(ki_predictions)
            return load_levels[-1]
        elif len(s_a_failure) == 1:
            load_levels= self.generate_load_levels(ki_predictions)
            return load_levels[-2]
        elif len(n_runout) == 0:
            #generate fictive runout
            n = np.append(woehler_points['n'], self.MAX_N)
            outcome = np.append(woehler_points['outcome'],'runout')
            load_levels = self.generate_load_levels(ki_predictions)
            #if the load_level leads to a numerical error a higher reansonalble number can be choosen
            s_a = np.append(woehler_points['s_a'], 1.05 * load_levels[0])
            curve = WoehlerCurve(s_a, n, outcome)
            woehler_params = {}
            woehler_params['k'] = az.plots.plot_utils.calculate_point_estimate(
                'mode', trace['k'])
            
            woehler_params['s_d'] = az.plots.plot_utils.calculate_point_estimate(
                'mode', trace['s_d'])
            
            woehler_params['n_e'] = az.plots.plot_utils.calculate_point_estimate(
                'mode', trace['n_e'])
            
            woehler_params['one_t_s'] = az.plots.plot_utils.calculate_point_estimate(
                'mode', trace['one_t_s'])
            
            print(load_levels)#remove this line
            
            load_levels = self.generate_load_levels_2(ki_predictions)
            load_level = self.find_best_load_level(trace, curve, load_levels,
                                              t_enough)
            return load_level
        else:
            n = woehler_points['n']
            outcome = woehler_points['outcome']
            s_a = woehler_points['s_a']
            curve = WoehlerCurve(s_a, n, outcome)
            woehler_params = {}
            woehler_params['k'] = az.plots.plot_utils.calculate_point_estimate(
                'mode', trace['k'])
            
            woehler_params['s_d'] = az.plots.plot_utils.calculate_point_estimate(
                'mode', trace['s_d'])
            
            woehler_params['n_e'] = az.plots.plot_utils.calculate_point_estimate(
                'mode', trace['n_e'])
            
            woehler_params['one_t_s'] = az.plots.plot_utils.calculate_point_estimate(
                'mode', trace['one_t_s'])
            load_levels = self.generate_load_levels_2(ki_predictions) #using ki-predictions appears to lead to better results than using woehler_params for small data points.
            load_level = self.find_best_load_level(trace, curve, load_levels,
                                              t_enough)
            return load_level

    def find_best_load_level(self, trace, curve, load_levels, t_enough):
        '''
        This function contains the algorithem for finding the
        next load level.

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
            Parmametrs.
        load_levels : Array
            A one dimensional np array of the empircally determined,
            viable load levels.
         t_enough : dict
             A dictionary that defines how accurate the spread
             (Streuspanne) for every parameter should be. If 
             that accuray is reached this parameter is not considered
             for the planning. Thus, the optimal load level
             only depend on the other parameters.

        Returns
        -------
        load_level : Int
            The aplitude for the next expeiment.

        '''
        #it is checked if any spreads are below the requred t_enough
        self.check_if_accurate_enough(trace, t_enough)
        
        distributions, samples, samples_load_level = \
            self.generate_distributions_load_level(trace, curve, load_levels)
        
        woehler_params = self.data_manager.transform_woehler_params(
            curve.Mali_4p_result)
        
        #the functions to caluculate the utlity functions are called
        all_nmses, nmses = self.calculate_nmses(distributions, woehler_params,
                                               load_levels)
        #examples for other utitlity functions are given. Note:
        #If a different utility function is used, the function also
        #needs to be changed in the local search.
        #mses = self.calculate_mbf_multidimensional(
        #    distributions, woehler_params, load_levels)
        #all_det_cov = self.calculate_det_cov(distributions, load_levels)
        #self.calculate_vars(distributions, load_levels)
        max_index = np.argmax(all_nmses)
        samples_before = samples
        load_levels_append, all_nmses_append, samples_append, samples_app = \
            self.local_search(load_levels,woehler_params, all_nmses,
                              samples, trace, curve)
        
        load_levels = np.append(load_levels,load_levels_append)
        all_nmses = np.append(all_nmses,all_nmses_append)
        samples['loads'] = np.append(samples['loads'],samples_append['loads'])
        samples['cycles'] = np.append(samples['cycles'],samples_append['cycles'])
        samples['outcome'] = np.append(samples['outcome'],samples_append['outcome'])
        load_levels = np.array(sorted(load_levels))
        distributions, samples, samples_load_level = \
            self.generate_distributions_load_level( trace, curve, load_levels)
        
        all_nmses, nmses = self.calculate_nmses(distributions, woehler_params,
                                               load_levels)
        #if the expected prece is needed it can be calulated as folows:
        prices = self.calculate_prices(samples_load_level, load_levels)
        
        return load_levels[max_index]
    def local_search(self,load_levels, woehler_params, all_nmses,samples,trace,curve):
        '''
        This function implements a local search around the previously as
        optimal determined load_level.

        Parameters
        ----------
        load_levels : Array
            A one dimensional np array of the empircally determined,
            viable load levels.
        woehler_params : Dict
            A dictionary containing the four woehler parameter.
        all_nmses : Array
            A one dimensional np array containing the NMSE values computed for the load_levels.
        samples : Dict
            All computed samples of possible future experiment results.
        trace : MultiTrace
            The MultiTrace object containing the samples from
            the posterior. The samples for a parameter can be 
            accessed similar to a dictionary by giving the parameter
            name as a key: trace['n_e']
        curve : Curve
            This is a curve object as returned by the WoehlerParams
            class. It contains the data points and the Max-Likeli-
            Parmametrs.

        Returns
        -------
        load_levels : Array
            A one dimensional np array of the empircally determined,
            viable load levels. This time including the load levels determined to be viable by the local search.
        all_nmses : Array
            A one dimensional np array containing the NMSE values computed for the load_levels.
        samples : Dict
            All samples of possible future experiment results computed.
        samples_app : Dict
            All samples of possible future experiment results computed on the load levels of the loacal search.

        '''
        all_nmses_og = all_nmses
        samples_app ={}
        samples_app['loads'] = []
        samples_app['cycles'] = []
        samples_app['outcome'] = []
        for factor in [-1,1]:
            max_index = np.argmax(all_nmses_og)
            run = True
            while run:
                load_level_append = np.array([load_levels[max_index]+3*factor])
                if load_level_append >= max(load_levels) or load_level_append <= min(load_levels):
                    break
                load_levels = np.append(load_levels, load_level_append)
                distributions, samples_append, samples_load_level = \
                    self.generate_distributions_load_level(trace, curve,
                                                          load_level_append)
                all_nmses_append, nmses_append = self.calculate_nmses(distributions, woehler_params,
                                                       load_level_append)
                #mses_append = self.calculate_mbf_multidimensional(
               #     distributions, woehler_params, load_level_append)
                all_nmses = np.append(all_nmses,all_nmses_append)
                max_index = np.argmax(all_nmses)
                samples_app['loads'] = np.append(samples_app['loads'],
                                                 samples_append['loads'])
                samples_app['cycles'] = np.append(samples_app['cycles'],
                                                  samples_append['cycles'])
                samples_app['outcome'] = np.append(samples_app['outcome'],
                                                   samples_append['outcome'])
                samples['loads'] = np.append(samples['loads'],samples_append['loads'])
                samples['cycles'] = np.append(samples['cycles'],samples_append['cycles'])
                samples['outcome'] = np.append(samples['outcome'],samples_append['outcome'])
                if  load_level_append != load_levels[max_index]:
                    run =False
                
        return load_levels, all_nmses, samples, samples_app
    
    def find_load_level_staircase(self, last_load_level, last_outcome, t):
        '''
        This function plannes the next experiment using the staircase methode.

        Parameters
        ----------
        last_load_level : Int
            The last load_level where an exeriment was run to be given.
        last_outcome : String
            The outcome of the last exeriment has to be given.
        t : Int
            The estimated spread (Streuspanne) of the current material.

        Returns
        -------
        load_level_staircase : Int
            The load level calculated by the staticase methode.

        '''
        std_log = self.calc_slog_from_t(t)
        if last_outcome == 'failure':
            load_level_staircase = last_load_level*(10**(std_log/1.05))**(-1)
        else:
            load_level_staircase = last_load_level*(10**(std_log/1.05))**(1)
        return load_level_staircase
    
    def calc_slog_from_t(self, t):
        '''
        Calculates slog from the spread (Streuspanne).

        Parameters
        ----------
        t : Int
            The spread.

        Returns
        -------
        std_log : int
            The logarithmic standard deviation.

        '''
        std_log = np.log10(t)/2.564
        return std_log
    def generate_load_levels(self, woehler_params):
        '''
        This function generates 4 viable load levels on the basis of esitimated or known woehler parameters. One for a failure probability of 0.7 one for P_r =0.3.
        The other are in the HCF. One at the upper end, one at the lower end.

        Parameters
        ----------
        woehler_params : Dict
            The four woehler parameters given in a dictionary.

        Returns
        -------
        load_levels : Array
            A 1-D np array containing the four load levels.

        '''
        s_d_50 = woehler_params['s_d']
        load_level_3 = help_functions.calc_s_long(woehler_params, 0.7)
        load_level_4 = help_functions.calc_s_long(woehler_params, 0.3)
        load_level_1 = help_functions.calc_s_short(woehler_params, 10000)
        load_level_1 = load_level_3 + 9 / 10 * (load_level_1-load_level_3)
        
        load_level_2 = load_level_3 + 1 / 10 * (load_level_1-load_level_3)
        load_levels = [load_level_1,load_level_2,load_level_3,load_level_4]
        return np.array(sorted(load_levels))
    def generate_load_levels_2(self, woehler_params):
        '''
        This function is an extention of generate_load_levels. This function generates viable load levels on the basis of esitimated or known woehler parameters. 

        Parameters
        ----------
        woehler_params : Dict
            The four woehler parameters given in a dictionary.

        Returns
        -------
        load_levels : Array
            A 1-D np array containing the four load levels.

        '''
        load_levels = self.generate_load_levels(woehler_params)
        dif= ( load_levels[1]-load_levels[0]) / 3
        load_levels_2 = []
        for ii in range(-1,4):
            load_levels_2 = np.append(
                load_levels_2,load_levels[0]+dif*ii)
            
        for ii in range(2):
            load_levels_2 = np.append(
                load_levels_2,load_levels_2[1] + dif * 2 * (ii + 1))
        
        for ii in range(2):
            load_levels_2 = np.append(
                load_levels_2,load_levels_2[2] + dif * 2 * (ii))
        load_levels_2 = np.append(load_levels_2,load_levels[3])
        load_levels_2 = np.append(load_levels_2,load_levels[3] - 5)
        dif= ( load_levels[3]-load_levels[2]) / 5
        for ii in range(3):
            load_levels_2 = np.append(
                load_levels_2, load_levels_2[3] + dif * (ii + 1))
        return np.array(sorted(load_levels_2))
    
    def check_if_accurate_enough(self, trace, t_enough):
        '''
        This function checks if the necessary spread is reached. Then the list self.params_not_enough is updated accordingly.

        Parameters
        ----------
        trace : MultiTrace
            The MultiTrace object containing the samples from
            the posterior. The samples for a parameter can be 
            accessed similar to a dictionary by giving the parameter
            name as a key: trace['n_e']
        t_enough : dict
            A dictionary that defines how accurate the spread
            (Streuspanne) for every parameter should be. If 
            that accuray is reached this parameter is not considered
            for the planning. Thus, the optimal load level
            only depend on the other parameters.

        Returns
        -------
        accurate_enough : Boolean
            True, when all parameters are determined accurate enough.

        '''
        
        
        self.params_not_enough = []
        for parameter in self.params_not_enough_all:
            
            hdi_interval = az.hdi(trace[parameter], 0.8)
            t_current = hdi_interval[1] / hdi_interval[0]
            if t_current >= t_enough[parameter]:
                self.params_not_enough = np.append(self.params_not_enough,parameter)
        
        if len(self.params_not_enough) == 0:
            accurate_enough = True
            self.params_not_enough = self.params
        else:
            accurate_enough = False
        return accurate_enough    

    def generate_distributions_load_level(self, trace, curve, load_levels):
        '''
        Estimates the distributions of the parameteters (expected after the next experiment) on every load level. Samples from the distributions are calculated and returned.

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
            Parmametrs.
        load_levels : Array
            A one dimensional np array of the empircally determined,
            viable load levels.

        Returns
        -------
        distributions : Dict
            A dictionary containing samples from the distribition of the parameters, that is expected after the next experiment.
        samples : Dict
            All computed samples of possible future experiment results.
        samples_load_level : Dict
            All computed samples of possible future experiment results. The load level is saved additionally.

        '''
        woehler_params = {}
        distributions = {}
        distributions['k'] = {}
        distributions['s_d'] = {}
        distributions['n_e'] = {}
        distributions['one_t_s'] = {}
        samples = {}
        samples['loads']        = []
        samples['cycles']       = []
        samples_load_level      = {}
        samples['outcome']      = []
        for load_level in load_levels:
            distributions['k'][load_level]       = []
            distributions['s_d'][load_level]     = []
            distributions['n_e'][load_level]     = []
            distributions['one_t_s'][load_level] = []
            samples_load_level[load_level]=[]
            
            for ii in range(500):
                #500 samples are generaded on the basis of the trace
                rand_int    = random.randint(0, len(trace['k'])-1)
                woehler_params['k']         = trace['k'][rand_int]
                woehler_params['s_d']       = trace['s_d'][rand_int]
                woehler_params['n_e']       = trace['n_e'][rand_int]
                woehler_params['one_t_s']   = trace['one_t_s'][rand_int]
                outcome_sample, n_sample = self.generate_woehler_point( 
                    woehler_params, load_level)
                s_a = np.append(curve.data['loads'], load_level)
                n = np.append(curve.data['cycles'], n_sample)
                outcome = np.append(curve.data['outcome'], outcome_sample)
                samples['loads']  = np.append( samples['loads'],load_level)
                samples['cycles']  = np.append( samples['cycles'],n_sample)
                samples['outcome']  = np.append( samples['outcome'],outcome_sample)
                samples_load_level[load_level] = \
                    np.append(samples_load_level[load_level], n_sample)
                try:
                    curve_new = WoehlerCurve(s_a, n, outcome)
                except:
                    print(s_a)
                    print(n)
                    print(outcome)
                    time.sleep(222222222222)
                woehler_params_likeli = self.data_manager.transform_woehler_params(
                    curve_new.Mali_4p_result)
                distributions['k'][load_level] = np.append(distributions['k'][
                    load_level], woehler_params_likeli['k'])
                distributions['s_d'][load_level] = np.append(distributions['s_d'][
                    load_level], woehler_params_likeli['s_d'])
                distributions['n_e'][load_level] = np.append(distributions['n_e'][
                    load_level], woehler_params_likeli['n_e'])
                distributions['one_t_s'][load_level] = np.append(distributions[
                    'one_t_s'][load_level], woehler_params_likeli['one_t_s'])
            woehler_params = self.data_manager.transform_woehler_params(
                curve.Mali_4p_result)
            samples_load_level[load_level] = np.mean(samples_load_level[
                load_level])
            
        return distributions, samples, samples_load_level
    
    def generate_woehler_point(self, woehler_params, load_level):
        '''
        Generates a fictive experiment result on the given load level using the given parameters.

        Parameters
        ----------
        woehler_params : Dict
            A dictionary containing the four woehler parameter.
        load_level : Int
            The load level on which the experiment is to be generated.

        Returns
        -------
        outcome : String
            Outcome of the experiment.
        n_sample : Int
            n of the experiment.

        '''
        k = woehler_params['k']
        s_d_50 = woehler_params['s_d']
        n_e     = woehler_params['n_e']
        one_t_s = woehler_params['one_t_s']
        amp_50 = load_level
        p_failure = stats.norm.cdf(np.log10(amp_50/s_d_50), loc=np.log10(1), 
                                   scale=np.log10(one_t_s)/2.5631)
        #determin if failure or runout
        if random.random() < p_failure:
            outcome = 'failure'
            #if failure determine n
            n_sample_level = 10 ** np.random.normal(np.log10(n_e), 
                                                  np.log10(one_t_s**k)/2.5631,
                                                  1)
            n_sample      = 10 ** (np.log10(n_sample_level) +
                                 k * (np.log10(s_d_50) - np.log10(amp_50)))
            #N can never be bigger than MAX_N
            if n_sample >= 0.989 * self.MAX_N: #for some non transparent reasen the 4P MaxLikli in WoehlerParams does not work with a 1. However, the error is marginal.
                n_sample = self.MAX_N
                outcome = 'runout'
            elif n_sample < 1:
                n_sample = 1
        else:
            outcome = 'runout'
            n_sample = self.MAX_N
        return outcome, n_sample
    #%% uitility functions 
        
    def calculate_nmses(self, distributions, woehler_params, load_levels):
        '''
        Calculates the NMSEs for all load levels.

        Parameters
        ----------
        distributions : Dict
            A dictionary containing samples from the distribition of the parameters, that is expected after the next experiment.
        woehler_params : Dict
            A dictionary containing the four woehler parameter.
        load_levels : Array
            A one dimensional np array of the empircally determined,
            viable load levels.

        Returns
        -------
        all_nmses : Array
            A one dimensional np array containing the NMSE values computed for the load_levels.
        nmses : Dict
            A dictionary containing the NMSE values of the four parameters.

        '''
        mses = {}
        nmses = {}
        
        for param in self.PARAM_LIST:
            mses[param] = np.array([])
            nmses[param] = np.array([])
        for load_level in load_levels:
            for param in self.PARAM_LIST:
                mse = self.calculate_mse(distributions[param][load_level],
                                         woehler_params[param])
                mses[param] = np.append(mses[param], mse)
                
        for param in self.PARAM_LIST:
            nmses[param] = self.calculate_nmse(mses[param])
        all_nmses = np.ones(len(nmses['k']))
        for param in self.params_not_enough:
            all_nmses = np.multiply(all_nmses, nmses[param])
        return all_nmses, nmses
    def calculate_mbf_multidimensional(self, distributions, woehler_params,
                                       load_levels):
        '''
        Calculates the MBF for the load levels.

        Parameters
        ----------
        distributions : Dict
            A dictionary containing samples from the distribition of the parameters, that is expected after the next experiment.
        woehler_params : Dict
            A dictionary containing the four woehler parameter.
        load_levels : Array
            A one dimensional np array of the empircally determined,
            viable load levels.

        Returns
        -------
        mbfs : Array
            A 1-D n array of all MBFs.

        '''
        
        
        mbfs = []
        for load_level in load_levels:
            many_dis = np.array([])
            for ii in range(len(distributions['k'][load_level])):
                dis =np.array([0])
                for param in self.params_not_enough:
                    dis = dis + (distributions[param][load_level]-
                                 woehler_params[param])**2
                dis = np.sqrt(dis)
                many_dis = np.append(many_dis, dis)
            mbf = np.mean(many_dis)
            mbfs = np.append(mbfs, mbf)
        return mbfs
    def calculate_mse(self, theta_roof, theta):
        '''
        Calculates the MSE.

        Parameters
        ----------
        theta_roof : Array
            1-D np array of the parameter.
        theta : Int
            "True" value of theta.

        Returns
        -------
        mse : Int
            MSE value of the given data.

        '''
        
        mse = statistics.variance(theta_roof) + (statistics.mean(theta_roof)
                                                 - theta)**2
        return mse
    
    def calculate_nmse(self, mses):
        '''
        Calculate the NMSE. The MSE is normed to 1. While multiple viable definitions of the NMSE are possible this is the currently used one.

        Parameters
        ----------
        mses : Array
            1-D np array of the MSEs.

        Returns
        -------
        nmses : Array
            1-D np array of the NMSEs.

        '''
        nmses = mses / max(mses)
        return nmses
    def calculate_nmse_old(self, mses, load_levels ):
        '''
        Old methode of calculating the NMSE. The area under the NMSE is normed to 1.

        Parameters
        ----------
        mses : Array
            1-D np array of the MSEs.
        load_levels : Array
            A one dimensional np array of the empircally determined,
            viable load levels.

        Returns
        -------
        nmses : Array
            1-D np array of the NMSEs.

        '''
        area = 0
        for ii in range(len(mses)-1):
            area = area + ((mses[ii]+mses[ii+1])/2 * (load_levels[ii+1] 
                                                      - load_levels[ii]))
        h = area/(load_levels.max()-load_levels.min())
        nmses = mses / h
        return nmses
    
   
    def calculate_det_cov(self, distributions, load_levels):
        '''
        This function calculates the determinant of the covariance matrix for all load levels. See chapter "Volumen des Hyperellipsoids"

        Parameters
        ----------
        distributions : Dict
            A dictionary containing samples from the distribition of the parameters, that is expected after the next experiment.
        load_levels : Array
            A one dimensional np array of the empircally determined,
            viable load levels.

        Returns
        -------
        cov_dets : Array
            All determinants of the covariance matrix for each load level.

        '''
        covs = {}
        cov_dets = []
        for load_level in load_levels:
            m=np.array([])
            for param in self.params_not_enough:
                if m.shape[0]==0:
                    m = np.append(m,distributions[param][load_level])
                elif m.shape[0]>=10:
                    m = np.array([m,distributions[param][load_level]])
                elif m.shape[0]<=10:
                    m = np.c_[m.T,distributions[param][load_level]].T
            if m.shape==():
                covs[load_level] = 0
                cov_dets = np.append(cov_dets,0)
            else:
                covs[load_level] = np.cov(m)
                if covs[load_level].shape==():
                    cov_dets = np.append(cov_dets,covs[load_level])
                else:
                    cov_dets = np.append(cov_dets,np.linalg.det(covs[load_level]))

        return cov_dets
    def calculate_vars(self, distributions, load_levels):
        '''
        Calculates the variance for each parameter and load level.

        Parameters
        ----------
        distributions : Dict
            A dictionary containing samples from the distribition of the parameters, that is expected after the next experiment.
        load_levels : Array
            A one dimensional np array of the empircally determined,
            viable load levels.

        Returns
        -------
        all_variances : Dict
            Variance for each parameter and load level..

        '''
        variances = {}

        for param in self.PARAM_LIST:
            variances[param] = np.array([])
            
        for load_level in load_levels:
            for param in self.PARAM_LIST:
                variance = np.var(distributions[param][load_level])
                variances[param] = np.append(variances[param], variance)
                
        all_variances = np.ones(len(variances['k']))
        for param in self.params_not_enough:
            all_variances = np.multiply(all_variances, variances[param])
        
       
        
        return all_variances

    def calculate_prices(self, samples_load_level, load_levels):
        '''
        Calculates the expected price for all load levels.

        Parameters
        ----------
        samples_load_level : Dict
            All computed samples of possible future experiment results. The load level is saved additionally.
        load_levels : Array
            A one dimensional np array of the empircally determined,
            viable load levels.

        Returns
        -------
        prices : Array
            Np array of all expected prices.

        '''
        prices= []
        for load_level in load_levels:
            n = samples_load_level[load_level]
            a = 300
            b = 6* 10**(-5)
            price = a + b*n
            prices  =np.append(prices, price)
        return prices
    
    def calc_entropy(self, samples, k_nn = 15):
        '''
        Calculates the entropy of the given samples using a k nearest neighbour estimation.

        Parameters
        ----------
        samples : Array
            All samples for which the entropy is to be calculated.
        k_nn : Int, optional
            Number of nearest neighbours to be considered. The default is 15.

        Returns
        -------
        entropy : Int
            Entropy of the given samples.
       
        '''
        
        y = np.array([])
        x = np.sort(samples)
        
        for ii in range(len(x)):
            dis = abs(x-x[ii])
            dis = np.sort(dis)
            y = np.append(y,k_nn/(len(x)-1)* 1/(2*dis[k_nn]))
            
        entropy = -sum(np.log10(y))
        return entropy
   
if __name__ == '__main__':
    __spec__ = None
    freeze_support()
    
    
        
        