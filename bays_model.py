# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 18:13:07 2021

@author: Sven Mordeja
"""
__author__ = "Sven Mordeja"
import arviz as az
import help_functions
import numpy as np
import pickle
import pymc3 as pm
from multiprocessing import Process, freeze_support

class BaysModel:

    def __init__(self):
        '''
        This class's function is to compute the Bayes Model.
        It is a class for legacy reasons, but can be transformed
        into functions easily.
        Returns
        -------
        None.

        '''
        
    def calc_model(self, woehler_points, num_samples = 1000, prior_from_ki = None):
        '''
        This function computes the bayesian model. The samples
        are computed using PYMC3.

        Parameters
        ----------
        woehler_points : Dict
            A dictonary containing the measured data points. 
            (s_a, n, outcome)
        num_samples : Int, optional
            The number of samples drawn from the posterior.
            Note that this is the number of samples computed per 
            process. Thus, the actual number depends on the 
            number of cores used for mutiprocessing. 
            The default is 1000.
        prior_from_ki : Dict, optional
            This dictionary contains the prior knowledge. For
            every parameter the mean and the std deviation is 
            given. When None is given a uniform distribution is
            automatically choosen. The default is None.

        Returns
        -------
        woehler_params_bayes : Dict
            Return the woehler parameters estimated using the 
            mods of the posterior.
        trace : MultiTrace
            The MultiTrace object containing the samples from
            the posterior. The samples for a parameter can be 
            accessed similar to a dictionary by giving the parameter
            name as a key: trace['n_e']
        prior_distribution : Dict
            A dictionary containing the prior distribution used.
            As keys the Parameters are used. 

        '''
        #for failures and runouts the likelihood is computed differntly
        s_a_failure, n_failure, s_a_runout, n_runout = help_functions.get_runout_and_failure(woehler_points)
        
        #defining the Model
        with pm.Model() as four_p_model:
            if prior_from_ki == None:
                #deining the uniform distriputions
                one_t_s = pm.Uniform('one_t_s', lower=1, upper=3)
                s_d = pm.Uniform('s_d', lower=10, upper=1000)
                k = pm.Uniform('k', lower=0.5, upper=200)
                n_e = pm.Uniform('n_e', lower=1e+05, upper=9.496252e+08)
            else:
                #the Normal distribution is cut of for unrealistic values
                BoundedNormal = pm.Bound(pm.Normal, lower=1)
                one_t_s = BoundedNormal('one_t_s', mu = prior_from_ki['one_t_s'],
                                        sd = prior_from_ki['one_t_s_std'])
                BoundedNormal = pm.Bound(pm.Normal, lower=0)
                s_d = BoundedNormal('s_d', mu = prior_from_ki['s_d'], 
                                    sd = prior_from_ki['s_d_std'])
                k = BoundedNormal('k', mu = prior_from_ki['k'], 
                                  sd = prior_from_ki['k_std'])
                n_e = BoundedNormal('n_e', mu = prior_from_ki['n_e'],
                                    sd = prior_from_ki['n_e_std'])
            
            #making a dictionary for the priors.These are used by some plotting functions 
            prior_distribution ={}
            prior_distribution['one_t_s'] = one_t_s
            prior_distribution['s_d'] = s_d
            prior_distribution['k'] = k
            prior_distribution['n_e'] = n_e    
            
            
            #the Likelihood is computed in the following section
            #Depending if runouts or failures are availible the Likelihood is computed diffently to avoid running into errors.
            #See section "Vierparametriges Maximum-Likelihood-Modell" for the needed formulas.
            
            one_t_s_log = pm.Deterministic('one_t_s_log', np.log10(one_t_s))
            s_d_log = pm.Deterministic('s_d_log', np.log10(s_d))
            n_e_log = pm.Deterministic('n_e_log', np.log10(n_e))
            
            #N_E_log  = pm.Deterministic('N_E_log',np.log10(N_E))
            if s_a_failure.size > 0:
                amplitude_basq = pm.Deterministic(
                    'amplitude_basq',np.log10(n_failure) + k*(np.log10(s_a_failure)
                                                              - s_d_log))
                amplitude_dauer_norm = pm.Deterministic(
                    'amplitude_dauer_norm', np.log10(s_a_failure )- s_d_log)
            
            if s_a_runout.size > 0:
                amplitude_runout_norm = pm.Deterministic(
                    'amplitude_durch_norm', np.log10(s_a_runout)- s_d_log)
            
            s_basq = pm.Deterministic('s_basq', k*one_t_s_log/2.5631)
            s_ueberg = pm.Deterministic('s_ueberg', one_t_s_log/2.5631)

            if s_a_failure.size > 0:
                #Potentials are used for censored data in PYMC3
                #See: https://docs.pymc.io/en/v3/pymc-examples/examples/survival_analysis/censored_data.html
                ueberg = pm.Potential("ueberg", pm.Normal.dist(
                    mu = np.log10(1), sd = s_ueberg).logcdf(amplitude_dauer_norm))
                zeitf = pm.Normal('zeitf',mu = n_e_log, sd = s_basq, observed =
                                  amplitude_basq)
            
            if s_a_runout.size > 0:
                durchl = pm.Potential("durchl", pm.Normal.dist(
                    mu = np.log10(1), sd = s_ueberg).logcdf(np.log10(1)-(
                        amplitude_runout_norm-np.log10(1))))
            #The Maximum Aposteriori can be computed as follows.           
            #self.map = pm.find_MAP(model=four_p_model)
            
            #The samples are computed with 2000 tuning steps.
            #Depending on the current data points the target_accept might need to be changed,
            #to avoid divergences. Experience showed high numbers to be useful. If very few 
            #data points are availible divergences may not be avoidable.
            trace = pm.sample(num_samples, target_accept=0.99,
                              tune=2000, return_inferencedata=False)
           
            woehler_params_bayes = {}
        woehler_params_bayes['k'] = az.plots.plot_utils.calculate_point_estimate(
            'mode', trace['k'])
        
        woehler_params_bayes['s_d'] = az.plots.plot_utils.calculate_point_estimate(
            'mode', trace['s_d'])
        
        woehler_params_bayes['n_e'] = az.plots.plot_utils.calculate_point_estimate(
            'mode', trace['n_e'])
        
        woehler_params_bayes['one_t_s'] = az.plots.plot_utils.calculate_point_estimate(
            'mode', trace['one_t_s'])
  
        return woehler_params_bayes, trace, prior_distribution
       
    def save_trace(self, trace, woehler_points, num, index):
        '''
        If neccessairy the trace and woehler points can be saved.
        This might be helpful since the trace is computionally intensive.

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

        Returns
        -------
        None.

        '''
        name = 'trace_' + str(index) +'_' + str(num)
        with open(name + '.pkl', 'wb') as outp:
            pickle.dump(trace, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(woehler_points, outp, pickle.HIGHEST_PROTOCOL)

    def load_trace(self, num, index):
        '''
        The the trace and woehler points can be loaded.

        Parameters
        ----------
        num : Int
            Number of data points used.
        index : Integer
            Index of a dataset (colum number) eg. 0, 1, 2,... .

        Returns
        -------
        trace : MultiTrace
            The MultiTrace object containing the samples from
            the posterior. The samples for a parameter can be 
            accessed similar to a dictionary by giving the parameter
            name as a key: trace['n_e']
        woehler_points : Dict
            A dictonary containing the measured data points. 
            (s_a, n, outcome)

        '''
        name = 'trace_' + str(index) +'_' + str(num)
        with open(name + '.pkl', 'rb') as inp:
            trace = pickle.load(inp)
            woehler_points = pickle.load(inp)
        return trace, woehler_points


if __name__ == '__main__':
    __spec__ = None
    freeze_support()

    
    