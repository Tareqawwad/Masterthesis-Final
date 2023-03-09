
# Copyright (c) 2019 - for information on the respective copyright owner
# see the NOTICE file and/or the repository
# https://github.com/boschresearch/pylife
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Mustapha Kassem"
__maintainer__ = "Anton Kolyshkin"

import numpy as np
import time
import pandas as pd
import numpy.ma as ma
from numba import jit
from numba_stats import norm
from multiprocessing import Process, freeze_support
from scipy import stats, optimize
import mystic as my


class WoehlerCurve2:
    
    def __init__(self, S, N, outcome, infin_only=False, param_fix={}, param_estim={}):
        self.data ={'s_a': np.array(S,np.float64), 'n': np.array(N,np.float64), 'outcome': np.array(outcome)}
        self.__get_runout_and_failure()
        self.__sort()
        
        self.param_fix = param_fix
        self.param_estim = param_estim
        #self.__data_sort()
        # self.__deviation()

        if not infin_only:
            if len(self.s_a_runout) == 0:
                self.__slope()
                self.TS_pearl_chain()
                self.__maximum_like_procedure_k()

            elif len(np.unique(self.s_a_failure)) < 2:
                self.__calc_ld_endur_zones()
                self.__maximum_like_procedure_Sd()
            else:
                self.__calc_ld_endur_zones()
                self.__slope_NE()             
                self.__maximum_like_procedure_all_pars()
        else:
            self.__calc_ld_endur_zones()
            self.__maximum_like_procedure_Sd()
            
    def __get_runout_and_failure(self):
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
        s_a = self.data['s_a']
        n = self.data['n']
        outcome = self.data['outcome']
        
        self.n_failure = np.array([])
        self.n_runout = np.array([])
        self.s_a_failure = np.array([])
        self.s_a_runout = np.array([])
        for ii in range(len(outcome)):
               
            if outcome[ii] == 'failure':
                self.n_failure = np.append(self.n_failure, n[ii])
                self.s_a_failure = np.append(self.s_a_failure, s_a[ii])
                
            else:
                self.n_runout = np.append(self.n_runout, n[ii])
                self.s_a_runout = np.append(self.s_a_runout, s_a[ii])                                 


    def __sort(self):
        '''
        Computes the start value of the load endurance limit. This is done by searching for the lowest load
        level before the appearance of a runout data point, and the first load level where a runout appears.
        Then the median of the two load levels is the start value.
        '''
        
        
        
        if len(self.s_a_runout) > 0:
            highest_runout = np.max(self.s_a_runout)
            zf = self.s_a_failure[self.s_a_failure > highest_runout]
            self.zone_fin = zf
          
            if len(zf) == 0:
                self.zone_fin = self.s_a_runout
        else:
            self.zone_fin = self.s_a_failure             
               

        self.ld_lvls_fin = np.unique(self.zone_fin, return_counts=True)
        if len(self.ld_lvls_fin[0]) < 1:
            self.sa_min = np.nan
        else:
            self.sa_min = min(self.ld_lvls_fin[0])
            
            
            
    def __calc_ld_endur_zones(self):
                
                
        self.zone_inf_max = np.max(np.unique(self.s_a_runout))
        zone_fin_min = np.min(self.zone_fin)
        
        if zone_fin_min == 0 or np.isnan(zone_fin_min):
            self.fatg_lim = self.zone_inf_max
        else:
            self.fatg_lim = np.mean([zone_fin_min, self.zone_inf_max])
            
        print('self.fatg_lim', self.fatg_lim)
        index = np.where(self.s_a_failure <= self.fatg_lim)
        self.s_a_failure_inf_zone = self.s_a_failure[index]
        
    # Evaluation of the finite zone

    def __slope(self):
        '# Computes the slope of the finite zone with the help of a linear regression function'

        self.a_wl, self.b_wl, _, _, _ = stats.linregress(np.log10(self.s_a_failure),
                                                         np.log10(self.n_failure)
                                                         )

        '# Woehler Slope'
        self.k = -self.a_wl
        
    def __slope_NE(self):
        '# Computes the slope of the finite zone with the help of a linear regression function'

        self.a_wl, self.b_wl, _, _, _ = stats.linregress(np.log10(self.s_a_failure),
                                                         np.log10(self.n_failure)
                                                         )

        '# Woehler Slope'
        self.k = -self.a_wl
        
        self.N0 = 10**self.b_wl
        '# Load-cycle endurance start value relative to the load endurance start value'
        self.N_E = 10**(self.b_wl + self.a_wl*(np.log10(self.fatg_lim)))

    def TS_pearl_chain(self):
        '''
        Pearl chain method: consists of shifting the fractured data to a median load level.
        The shifted data points are assigned to a Rossow failure probability.The scatter in load-cycle
        direction can be computed from the probability net.
        '''
        # Mean load level:
        self.Sa_shift = np.mean(self.s_a_failure)

        # Shift probes to the mean load level
        self.N_shift = self.n_failure * ((self.Sa_shift/self.s_a_failure)**(-self.k))
        self.N_shift = np.sort(self.N_shift)

        self.fp = _rossow_fail_prob(self.N_shift)
        self.u = stats.norm.ppf(self.fp)

        self.a_pa, self.b_pa, _, _, _ = stats.linregress(np.log10(self.N_shift), self.u)

        # Scatter in load cycle direction
        self.TN_pearl_chain = 10**(2.5631031311*(1./self.a_pa))

        # Scatter in load direction
        '# Empirical method "following Koeder" to estimate the scatter in load direction '
        self.TS_pearl_chain = self.TN_pearl_chain**(1./self.k)

   
    def __maximum_like_procedure_all_pars(self):
        """
        Maximum likelihood is a method of estimating the parameters of a distribution model by maximizing
        a likelihood function, so that under the assumed statistical model the observed data is most probable.
        This procedure consists of estimating the Woehler curve parameters, where some of these paramters may
        be fixed by the user. The remaining paramters are then fitted to produce the best possible outcome.
        The procedure uses the function Optimize.fmin
        Optimize.fmin iterates over the function mali_sum_lolli values till it finds the minimum

        https://en.wikipedia.org/wiki/Maximum_likelihood_estimation

        Parameters
        ----------
        self.start_values: Start values of the Mali estimated parameters if none are fixed by the user.

        self.dict_bound: Boundary values of the Mali estimated parameters if none are fixed by the user.
        This forces the optimizer to search for a minimum solution within a given area.

        Returns
        -------
        self.Mali_5p_result: The estimated parameters computed using the optimizer.

        """

        if self.k > 0:
            self.dict_bound = {'SD_50': (self.fatg_lim*0.5, self.fatg_lim*1.5),
                               '1/TS': (0.5, 2),
                               'k_1': (self.k*0.3, self.k*5),
                               'ND_50': (self.N_E*0.15, self.n_runout.min() * 0.9)}
        else:
            self.k = 10
            self.dict_bound = {'SD_50': (self.fatg_lim*0.5, self.fatg_lim*1.5),
                               '1/TS': (0.5, 2),
                               'k_1': (1, 150),
                               'ND_50': (self.N_E*0.2, self.n_runout.min() * 0.9)}
        
        self.start_values = {'SD_50': self.fatg_lim,
                      '1/TS': 1.5,
                      'k_1': self.k,
                      'ND_50': self.N_E}

        for _k in self.param_fix:
            self.start_values.pop(_k)
            self.dict_bound.pop(_k)
            
        print('self.dict_bound', self.dict_bound)
        
        try:
            print('trying mystic')
            var_opt = my.scipy_optimize.fmin(_func_wrapper, [*self.start_values.values()],
                                             bounds=[*self.dict_bound.values()],
                                             args=([*self.start_values], self.param_fix,
                                                   self.s_a_failure, self.n_failure, self.s_a_runout, self.s_a_failure_inf_zone
                                                   ),
                                             disp=False,
                                             maxiter=1e3,
                                             maxfun=1e4,
                                             )
                                             
            self.Mali_4p_result = {}
            self.Mali_4p_result.update(self.param_fix)
            self.Mali_4p_result.update(zip([*self.start_values], var_opt))
                                             
			
        except:
            print('using scipy')
            var_opt = optimize.minimize(_func_wrapper, [*self.start_values.values()],
                                         args=([*self.start_values], self.param_fix,
                                               self.s_a_failure, self.n_failure, self.s_a_runout, self.s_a_failure_inf_zone,
                                               ),                                              
                                         method = 'L-BFGS-B',
                                         options={'maxiter' : 1e3,}                                         
                                         )


            self.Mali_4p_result = {}
            self.Mali_4p_result.update(self.param_fix)
            self.Mali_4p_result.update(zip([*self.start_values], var_opt.x))


        for key in self.Mali_4p_result.keys():
            if key in self.dict_bound.keys():
                if self.Mali_4p_result[key] in self.dict_bound[key]:
                    print('WARNING!!! The ', key, 'boundary was reached')
                
                

    def __maximum_like_procedure_Sd(self):
        ''' This maximum likelihood procedure estimates the load endurance limit SD50_mali_2_param and the
        scatter in load direction TS_mali_2_param.
        Moreover, the load cycle endurance is computed by the interesecting endurance limit line with the
        line of slope k_1
        '''
        SD_start = self.fatg_lim
        TS_start = 1.2

        var = optimize.fmin(Mali_SD_TS,
                            [SD_start, TS_start],
                            args=(self.s_a_failure_inf_zone, self.s_a_runout), disp=False)
        self.Mali_4p_result = {}
        self.Mali_4p_result = {'SD_50': var[0], '1/TS': var[1], 'k_1': None, 'ND_50': None}
        self.N_E = np.nan
        
        

    def __maximum_like_procedure_k(self):
        ''' This maximum likelihood procedure estimates the load endurance limit k_mali_2_param and the
        scatter in cycke direction TS_mali_2_param.
        Moreover, the load cycle endurance is computed by the interesecting endurance limit line with the
        line of slope k_1
        '''
        k_start = self.k
        TS_start = 1.2
        NE_fiktiv = 10**(self.b_wl + self.a_wl*(np.log10(self.sa_min)))
        

        var = optimize.fmin(Mali_k_TS, [k_start, TS_start],
                            args=(self.s_a_failure, self.sa_min, self.n_failure, NE_fiktiv),
                            disp=False)
        self.Mali_4p_result = {}
        self.Mali_4p_result = {'SD_50': None, '1/TS': var[1], 'k_1': var[0], 'ND_50': None}
        
@jit(nopython=True, cache=True)
def mali_sum_lolli(SD, TS, k, N_E, s_a_failure, n_failure, s_a_runout, s_a_failure_inf_zone):
    """
    Produces the likelihood functions that are needed to compute the parameters of the woehler curve.
    The likelihood functions are represented by probability and cummalative distribution functions.
    The likelihood function of a runout is 1-Li(fracture). The functions are added together, and the
    negative value is returned to the optimizer.

    Parameters
    ----------
    SD:
        Endurnace limit start value to be optimzed, unless the user fixed it.
    TS:
        The scatter in load direction 1/TS to be optimzed, unless the user fixed it.
    k:
        The slope k_1 to be optimzed, unless the user fixed it.
    N_E:
        Load-cycle endurance start value to be optimzed, unless the user fixed it.
    TN:
        The scatter in load-cycle direction 1/TN to be optimzed, unless the user fixed it.
    fractures:
        The data that our log-likelihood function takes in. This data represents the fractured data.
    zone_inf:
        The data that our log-likelihood function takes in. This data is found in the infinite zone.
    load_cycle_limit:
        The dependent variable that our model requires, in order to seperate the fractures from the
        runouts.

    Returns
    -------
    neg_sum_lolli :
        Sum of the log likelihoods. The negative value is taken since optimizers in statistical
        packages usually work by minimizing the result of a function. Performing the maximum likelihood
        estimate of a function is the same as minimizing the negative log likelihood of the function.

    """
    # Likelihood functions of the fractured data
    
    x_ZF = np.log10(n_failure * ((s_a_failure/SD)**(k)))
    Mu_ZF = np.log10(N_E)
    Sigma_ZF = np.log10(TS**k)/2.5631031311
    Li_ZF = norm.pdf(x_ZF, Mu_ZF, Sigma_ZF)
    LLi_ZF = np.sum(np.log(Li_ZF))

    # Likelihood functions of the data found in the infinite zone
    std_log = np.log10(TS)/2.5631031311
    
    Li_DF_failure = np.sum(np.log(norm.cdf(np.log10(s_a_failure_inf_zone/SD), np.log10(1), std_log)))
    Li_DF_runout = np.sum(np.log(1-norm.cdf(np.log10(s_a_runout/SD), np.log10(1), std_log)))
    #Li_DF_runout = np.sum(np.log(norm.cdf(-np.log10(s_a_runout/SD), np.log10(1), std_log)))

    neg_sum_lolli = - (Li_DF_runout + Li_DF_failure + LLi_ZF)

    return neg_sum_lolli
    '''
    x = np.arange(0,12,1)
print(len(x))
-(np.log(scipy.stats.norm.pdf(x,6.0,1.0)).sum()) 

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets) 
negative_loglikelihood(2, tfd.Normal(6,1)) 
'''

@jit(nopython=True, cache=True)
def Mali_SD_TS(variables, s_a_failure_inf_zone, s_a_runout):
    """
    Produces the likelihood functions that are needed to compute the endurance limit and the scatter
    in load direction. The likelihood functions are represented by a cummalative distribution function.
    The likelihood function of a runout is 1-Li(fracture).

    Parameters
    ----------
    variables:
        The start values to be optimized. (Endurance limit SD, Scatter in load direction 1/TS)
    zone_inf:
        The data that our log-likelihood function takes in. This data is found in the infinite zone.
    load_cycle_limit:
        The dependent variable that our model requires, in order to seperate the fractures from the
        runouts.

    Returns
    -------
    neg_sum_lolli :
        Sum of the log likelihoods. The negative value is taken since optimizers in statistical
        packages usually work by minimizing the result of a function. Performing the maximum likelihood
        estimate of a function is the same as minimizing the negative log likelihood of the function.

    """

    SD = variables[0]
    TS = variables[1]

    std_log = np.log10(TS)/2.5631031311
       
    Li_DF_failure = np.sum(np.log(norm.cdf(np.log10(s_a_failure_inf_zone/SD), np.log10(1), std_log)))
    #Li_DF_runout = np.sum(np.log(1-norm.cdf(np.log10(s_a_runout/SD), np.log10(1), std_log)))
    Li_DF_runout = np.sum(np.log(norm.cdf(-np.log10(s_a_runout/SD), np.log10(1), std_log)))

    neg_sum_lolli = -(Li_DF_runout + Li_DF_failure)

    return neg_sum_lolli

@jit(nopython=True, cache=True)
def Mali_k_TS(variables, s_a_failure, sa_min, n_failure, N_E):
    """
    Produces the likelihood functions that are needed to compute the slope and the scatter
    in the cycle direction. The likelihood functions are represented by a cummalative distribution function.
    The likelihood function of a runout is 1-Li(fracture).

    Parameters
    ----------
    variables:
        The start values to be optimized. (Endurance limit SD, Scatter in load direction 1/TS)
    zone_inf:
        The data that our log-likelihood function takes in. This data is found in the infinite zone.
    load_cycle_limit:
        The dependent variable that our model requires, in order to seperate the fractures from the
        runouts.

    Returns
    -------
    neg_sum_lolli :
        Sum of the log likelihoods. The negative value is taken since optimizers in statistical
        packages usually work by minimizing the result of a function. Performing the maximum likelihood
        estimate of a function is the same as minimizing the negative log likelihood of the function.

    """

    k = variables[0]
    TS = variables[1]

    # Likelihood functions of the fractured data
    x_ZF = np.log10(n_failure * ((s_a_failure/sa_min)**(k)))
    Mu_ZF = np.log10(N_E)
    Sigma_ZF = np.log10(TS**k)/2.5631031311
    Li_ZF = norm.pdf(x_ZF, Mu_ZF, Sigma_ZF)
    LLi_ZF = np.log(Li_ZF)

    sum_lolli = LLi_ZF.sum()
    neg_sum_lolli = -sum_lolli

    return neg_sum_lolli

@jit(nopython=True, cache=True)
def _rossow_fail_prob(x):
    """ Failure Probability estimation formula of Rossow

    'Statistics of Metal Fatigue in Engineering' page 16

    https://books.google.de/books?isbn=3752857722
    """
    i = np.arange(len(x))+1
    pa = (3.*(i)-1.)/(3.*len(x)+1.)

    return pa


def _func_wrapper(var_args, var_keys, fix_args, s_a_failure, n_failure, s_a_runout, s_a_failure_inf_zone):
    ''' 1) Finds the start values to be optimized. The rest of the paramters are fixed by the user.
        2) Calls function mali_sum_lolli to calculate the maximum likelihood of the current
        variable states.
    '''
    args = {}
    args.update(fix_args)
    args.update(zip(var_keys, var_args))

    
    return mali_sum_lolli(args['SD_50'], args['1/TS'], args['k_1'], 
                            args['ND_50'], s_a_failure, n_failure, s_a_runout, s_a_failure_inf_zone)
                          


if __name__ == "__main__":
    freeze_support()
    #example
    # import datetime
    # amplitude = np.array([165.00,165.00,138.00,138.00,110,110,83,83,31.70,32.40,34.50,32.40,34.50,41.40])
    # n = np.array([47500,77400,231400,186300,208900,344100,907500,1113300,100000000,37714000,89314000,34930000,8451000,100000000])
    # outcome = ['failure','failure','failure','failure','failure','failure','failure','failure','runout','failure','failure','failure','failure','runout']
    
    # amplitude_zeit = np.array([165.00,165.00,138.00,138.00,110,110,83,83])
    # amplitude_zeit_log  = np.log10(amplitude_zeit)
    # n_zeit = np.array([47500,77400,231400,186300,208900,344100,907500,1113300])
    # n_zeit_log  = np.log10(n_zeit)
    
    # amplitude_dauer = np.array([32.40,34.50,32.40,34.50])
    # amplitude_dauer_log  = np.log10(amplitude_dauer)
    # n_dauer = np.array([37714000,89314000,34930000,8451000])
    # n_dauer_log  = np.log10(n_dauer)
    
    # amplitude_durch = np.array([31.70,41.40])
    # amplitude_durch_log  = np.log10(amplitude_durch)
    # n_durch = np.array([100000000,100000000])
    # n_durch_log  = np.log10(n_durch)
    # curve = WoehlerCurve(amplitude, n, outcome)
    # begin_time = datetime.datetime.now()
    # for i in range(100):
    #     curve = WoehlerCurve(amplitude, n, outcome)
    #     woehler_params = curve.Mali_4p_result
    # print(datetime.datetime.now() - begin_time)
    