
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


class WoehlerCurve:
    
    def __init__(self, S, N, outcome, infin_only=False, param_fix={}, param_estim={}):
        self.data ={'loads': np.array(S,np.float64), 'cycles': np.array(N,np.float64), 'outcome': np.array(outcome)}
        self.param_fix = param_fix
        self.param_estim = param_estim
        self.__data_sort()
        self.__slope()
        # self.__deviation()

        if not infin_only:
            if len(self.runouts['loads']) == 0:
                self.__maximum_like_procedure_k()

            elif self.n_lvls_frac < 2:
                self.__maximum_like_procedure_Sd()
            else:
                self.__maximum_like_procedure_all_pars()
        else:
            self.__maximum_like_procedure_Sd()
         
    def __data_sort(self):

        self.loads_max = self.data['loads'].max()
        self.loads_min = self.data['loads'].min()

        self.cycles_max = self.data['cycles'].max()
        self.cycles_min = self.data['cycles'].min()
        fractures_boolean = self.data['outcome'] == np.array(['failure'])
        runouts_boolean = self.data['outcome'] == np.array(['runout'])
        self.fractures = {}
        self.runouts = {}
        for parameter in self.data:
            self.fractures[parameter] = self.data[parameter][fractures_boolean]
            self.runouts[parameter] = self.data[parameter][runouts_boolean]
            
        
        if len(self.runouts['loads']) > 0:
            zone_fin_boolean = self.fractures['loads'] > self.runouts['loads'].max()
            self.zone_fin = {}
            for parameter in self.data:
                self.zone_fin[parameter] = self.fractures[parameter][zone_fin_boolean]
            if len(self.zone_fin['loads']) == 0:
                self.zone_fin = self.runouts
        else:
            self.zone_fin = self.fractures

        self.__calc_ld_endur_zones()

        self.ld_lvls = np.unique(self.data['loads'], return_counts=True)
        self.ld_lvls_fin = np.unique(self.zone_fin['loads'], return_counts=True)
        if len(self.ld_lvls_fin[0]) < 1:
            self.sa_min = np.nan
        else:
            self.sa_min = min(self.ld_lvls_fin[0])
        self.ld_lvls_frac = np.unique(self.fractures['loads'], return_counts=True)
        self.n_lvls_frac = len(self.ld_lvls_frac[0])

    def __calc_ld_endur_zones(self):
        '''
        Computes the start value of the load endurance limit. This is done by searching for the lowest load
        level before the appearance of a runout data point, and the first load level where a runout appears.
        Then the median of the two load levels is the start value.
        '''
        zone_fin_min = self.zone_fin['loads'].min()
        
        if zone_fin_min == 0 or np.isnan(zone_fin_min):
            self.fatg_lim = self.runouts['loads'].max()
        else:
            self.fatg_lim = np.mean([zone_fin_min, self.runouts['loads'].max()])
        zone_inf_boolean = self.data['loads'] <= self.fatg_lim
        self.zone_inf = {}
        for parameter in self.data:
            self.zone_inf[parameter] = self.data[parameter][zone_inf_boolean]
        
    # Evaluation of the finite zone

    def __slope(self):
        '# Computes the slope of the finite zone with the help of a linear regression function'

        self.a_wl, self.b_wl, _, _, _ = stats.linregress(np.log10(self.fractures['loads']),
                                                         np.log10(self.fractures['cycles'])
                                                         )

        '# Woehler Slope'
        self.k = -self.a_wl
        '# Cycle for load = 1'
        self.N0 = 10**self.b_wl
        '# Load-cycle endurance start value relative to the load endurance start value'
        self.N_E = 10**(self.b_wl + self.a_wl*(np.log10(self.fatg_lim)))

    def __deviation(self):
        '''
        Pearl chain method: consists of shifting the fractured data to a median load level.
        The shifted data points are assigned to a Rossow failure probability.The scatter in load-cycle
        direction can be computed from the probability net.
        '''
        # Mean load level:
        self.Sa_shift = np.mean(self.fractures['loads'])

        # Shift probes to the mean load level
        self.N_shift = self.fractures['cycles'] * ((self.Sa_shift/self.fractures['loads'])**(-self.k))
        self.N_shift = np.sort(self.N_shift)

        self.fp = _rossow_fail_prob(self.N_shift)
        self.u = stats.norm.ppf(self.fp)

        self.a_pa, self.b_pa, _, _, _ = stats.linregress(np.log10(self.N_shift), self.u)

        # Scatter in load cycle direction
        self.TN = 10**(2.5631031311*(1./self.a_pa))

        # Scatter in load direction
        '# Empirical method "following Koeder" to estimate the scatter in load direction '
        self.TS = self.TN**(1./self.k)

   
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
        self.p_opt: Start values of the Mali estimated parameters if none are fixed by the user.

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
                               'ND_50': (self.N_E*0.15, self.runouts['cycles'].min() * 0.9)}
        else:
            self.k = 10
            self.dict_bound = {'SD_50': (self.fatg_lim*0.5, self.fatg_lim*1.5),
                               '1/TS': (0.5, 2),
                               'k_1': (1, 150),
                               'ND_50': (self.N_E*0.2, self.runouts['cycles'].min()*0.9)}
        self.p_opt = {'SD_50': self.fatg_lim,
                      '1/TS': 1.5,
                      'k_1': self.k,
                      'ND_50': self.N_E}

        for _k in self.param_fix:
            self.p_opt.pop(_k)
            self.dict_bound.pop(_k)
        
        var_opt = my.scipy_optimize.fmin(_func_wrapper, [*self.p_opt.values()],
                                         bounds=[*self.dict_bound.values()],
                                         args=([*self.p_opt], self.param_fix,
                                               self.fractures, self.data
                                               ),
                                         disp=False,
                                         maxiter=1e3,
                                         maxfun=1e4,
                                         )

        self.Mali_4p_result = {}
        self.Mali_4p_result.update(self.param_fix)
        self.Mali_4p_result.update(zip([*self.p_opt], var_opt))

        for key in self.Mali_4p_result.keys():
            if key in self.dict_bound[key]:
                print('WARNING!!! The ', key, 'boundary was reached')
    @jit(nopython=True, cache=True)
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
                            args=(self.zone_inf), disp=False)
        self.Mali_4p_result = {}
        self.Mali_4p_result = {'SD_50': var[0], '1/TS': var[1], 'k_1': None, 'ND_50': None}
        self.N_E = np.nan
    @jit(nopython=True, cache=True)
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
                            args=(self.fractures, self.sa_min, NE_fiktiv),
                            disp=False)
        self.Mali_4p_result = {}
        self.Mali_4p_result = {'SD_50': None, '1/TS': var[1], 'k_1': var[0], 'ND_50': None}
@jit(nopython=True, cache=True)
def mali_sum_lolli(SD, TS, k, N_E, fractures, alldata, alldata_outcome):
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
    
    x_ZF = np.log10(fractures[0,:] * ((fractures[1,:]/SD)**(k)))
    Mu_ZF = np.log10(N_E)
    Sigma_ZF = np.log10(TS**k)/2.5631031311
    Li_ZF = norm.pdf(x_ZF, Mu_ZF, Sigma_ZF)
    LLi_ZF = np.log(Li_ZF)

    # Likelihood functions of the data found in the infinite zone
    std_log = np.log10(TS)/2.5631031311
    
    #runouts = ma.masked_where(alldata[2,:] == 'failure', alldata[0,:])
    
    #t = runouts.mask.astype(int)
    if alldata_outcome[0] == 'failure':
        t1 = np.array([1])
    else:
        t1 = np.array([0])
    if alldata_outcome[1] == 'failure':
        t2 = np.array([1])
    else:
        t2 = np.array([0])   
    t = np.append(t1,t2)
    for ii in range(2,len(alldata_outcome)):
        if alldata_outcome[ii] == 'failure':
            t = np.append(t,np.array(1))
        else:
            t = np.append(t,np.array(0))
    Li_DF = norm.cdf(np.log10(alldata[1,:]/SD), np.log10(1), std_log)#norm.cdf(np.log10(alldata['loads']/SD), loc=np.log10(1), scale=std_log)
    LLi_DF = np.log(1-t-(1-2*t)*Li_DF).astype(np.float64)
    sum_lolli = 0.0
    for LLi_DFs in LLi_DF:
        sum_lolli = sum_lolli + LLi_DFs
    for LLi_ZFs in LLi_ZF:
        sum_lolli = sum_lolli + LLi_ZFs
    
    #sum_lolli = LLi_DF.sum() + LLi_ZF.sum()
    neg_sum_lolli = -sum_lolli

    return neg_sum_lolli
    '''
    x = np.arange(0,12,1)
print(len(x))
-(np.log(scipy.stats.norm.pdf(x,6.0,1.0)).sum()) 

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets) 
negative_loglikelihood(2, tfd.Normal(6,1)) 
'''

def Mali_SD_TS(variables, zone_inf):
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
    runouts = ma.masked_where(zone_inf.outcome == 'failures', zone_inf.cycles)
    t = runouts.mask.astype(int)
    Li_DF = norm.cdf(np.log10(zone_inf.loads/SD), np.log10(1), abs(std_log))
    LLi_DF = np.log(t+(1-2*t)*Li_DF)

    sum_lolli = LLi_DF.sum()
    neg_sum_lolli = -sum_lolli

    return neg_sum_lolli


def Mali_k_TS(variables, fractures, sa_min, N_E):
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
    x_ZF = np.log10(fractures.cycles * ((fractures.loads/sa_min)**(k)))
    Mu_ZF = np.log10(N_E)
    Sigma_ZF = np.log10(TS**k)/2.5631031311
    Li_ZF = norm.pdf(x_ZF, Mu_ZF, Sigma_ZF)
    LLi_ZF = np.log(Li_ZF)

    sum_lolli = LLi_ZF.sum()
    neg_sum_lolli = -sum_lolli

    return neg_sum_lolli


def _rossow_fail_prob(x):
    """ Failure Probability estimation formula of Rossow

    'Statistics of Metal Fatigue in Engineering' page 16

    https://books.google.de/books?isbn=3752857722
    """
    i = np.arange(len(x))+1
    pa = (3.*(i)-1.)/(3.*len(x)+1.)

    return pa


def _func_wrapper(var_args, var_keys, fix_args, fractures, alldata):
    ''' 1) Finds the start values to be optimized. The rest of the paramters are fixed by the user.
        2) Calls function mali_sum_lolli to calculate the maximum likelihood of the current
        variable states.
    '''
    args = {}
    args.update(fix_args)
    args.update(zip(var_keys, var_args))
    fractures = np.array([fractures['cycles'],fractures['loads']])
    alldata_outcome =alldata['outcome']
    alldata   = np.array([alldata['cycles'],alldata['loads']])
    
    return mali_sum_lolli(args['SD_50'].astype(np.float64), args['1/TS'].astype(np.float64), args['k_1'].astype(np.float64), args['ND_50'].astype(np.float64),
                          fractures, alldata, alldata_outcome)
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
    