
    #   ---------------------------------------------------------------------------------------
    
    # Copyright 2023 by John B Rundle, University of California, Davis, CA USA
    # 
    # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
    # documentation files (the     "Software"), to deal in the Software without restriction, including without 
    # limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
    # and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    # 
    # The above copyright notice and this permission notice shall be included in all copies or suSKLantial portions of the Software.
    # 
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
    # WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
    # COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
    # ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    #
    # Note that some part of these codes were written by the AI program chatGPT, a product of openai.com
    #
    #   ---------------------------------------------------------------------------------------
    ###############################################################################################
    ###############################################################################################

import os

import numpy as np
import random
import matplotlib.pyplot as plt
import math
from math import log10

    ###########################################################################
    ###########################################################################
            
def build_EMA_timeseries(eqs_list, time_list, NSteps, data_start_year, lambda_min_mult, lambda_max_mult):
            
    #   Define the fundamental list eqs_list (monthly number of events) from the year_array,
    #       which is the event catalog timeseries, converted from time index numbers to years
    
    #   First step is to convert event timeseries to a monthly timeseries using delta_time
    
    delta_time = 0.07692    # 1 month when catalog is converted to years by scaling the data
    
    #   Note that eqlist is the monthly number list of ETAS earthquakes 
                
    #   -------------------------------------------------------------
    
    #   Adjust earthquake time series to have a minimum and maximum monthly value if necessary
    
    adjust_timeseries = True        #   We can make this adjustable
    
    if adjust_timeseries:
    
        
        forecast_intervals = [0.]   #   This turns exclusions off
    
        excluded_months = int(forecast_intervals[0] * 13)
        
        eqs_list_excluded = eqs_list[:-excluded_months]
        
        if len(eqs_list_excluded) > 0:
    
            mean_eqs   = round(np.mean(eqs_list_excluded),3)
            stdev_eqs  = round(np.std(eqs_list_excluded),3)
            
        else:
        
            mean_eqs   = round(np.mean(eqs_list),3)
            stdev_eqs  = round(np.std(eqs_list),3)
    
        min_rate = lambda_min_mult * mean_eqs
        max_rate = lambda_max_mult * stdev_eqs + min_rate   #   1.0 works best so far
    
        for i in range(len(time_list)):
            if int(eqs_list[i]) <= min_rate:
                eqs_list[i] = min_rate
            if int(eqs_list[i]) >= max_rate:
                eqs_list[i] = max_rate
            
    #   -------------------------------------------------------------
    
    #   Apply Exponential Moving Average to eqs_list
    timeseries_EMA = timeseries_to_EMA(eqs_list, NSteps)
    
    #   Generate the monthly seismicity timeseries and require the timeseries data 
    #       to occur only after the data_start_year.  Number timeseries is computed/replaced
    #       by a logarithmic timeseries
    
    time_list_reduced, log_number_reduced, timeseries_EMA_reduced = \
                adjust_analysis_timeseries(time_list, timeseries_EMA, data_start_year, delta_time)
                
    #   -------------------------------------------------------------
        
#     number_data_points = 0
#     
#     for k in range(len(time_list)):
#         if time_list[k] >= data_start_year:
#             number_data_points += 1
#             
#     NN = number_data_points
    
    return timeseries_EMA_reduced, time_list_reduced, log_number_reduced, min_rate, max_rate
    
    ######################################################################
    
def adjust_analysis_timeseries(time_list, timeseries_EMA, data_start_year, delta_time):
        
        
#   This function creates a list of events after data_start_year and prepares it for analysis
#
    if data_start_year <= time_list[0]:
        data_start_year = time_list[0]
        
    number_data_points = 0
    
    for k in range(len(time_list)):
        if time_list[k] >= data_start_year:
            number_data_points += 1
            
    for i in range(len(time_list)):
        time_list[i] += 2.*delta_time        #    Adjust times to properly align the EMA times with the large EQ times
        
    last_index = len(time_list)-1
    time_list[last_index] += 2.*delta_time   #   adjustment to ensure correct last event time sequence
    
    log_number = [math.log(1.0+timeseries_EMA[i],10) for i in range(len(timeseries_EMA))]
    
    
    log_number_reduced          = log_number[- number_data_points:]
    time_list_reduced           = time_list[- number_data_points:]
    timeseries_EMA_reduced      = timeseries_EMA[- number_data_points:]
   
    return time_list_reduced, log_number_reduced, timeseries_EMA_reduced
      
    ###########################################################################
    
def timeseries_to_EMA(timeseries_orig, N_Steps):

    #   timeseries_orig is a list input.  Output is a list that is an Exponential Moving Average

    timeseries_EMA = []
        
    for i in range(1,len(timeseries_orig)+1):
        timeseries_raw = []
         
        for j in range(i):
            timeseries_raw.append(timeseries_orig[j])
             
        datapoint_EMA = EMA_weighted_time_series(timeseries_raw, N_Steps)
         
        timeseries_EMA.append(datapoint_EMA)
                  
    return timeseries_EMA
    
    ######################################################################
    
def EMA_weighted_time_series(time_series, NSteps):

    #   This method computes the Exponential Weighted Average of a list.  Last
    #       in the list elements are exponentially weighted the most

    N_events = len(time_series)
    
    weights = EMA_weights(N_events, NSteps)
    
    weights_reversed = list(reversed(weights))

    EMA_weighted_ts = []
    partial_weight_sum = 0.
    
    for i in range(N_events):
        partial_weight_sum += weights[i]
        weighted_ts = round(float(time_series[i])*weights_reversed[i],4)
        
        EMA_weighted_ts.append(weighted_ts)
        
    partial_weight_sum = round(partial_weight_sum,4)
    sum_value = sum(EMA_weighted_ts)
    
    if (float(partial_weight_sum)) <= 0.0:
        sum_value = 0.0001
        partial_weight_sum = 1.
    
    try:
        weighted_sum = float(sum_value)/float(partial_weight_sum)
    except:
        weighted_sum = 0.0
    
    return weighted_sum
    
    ######################################################################
    
def EMA_weights(N_events, N_Steps):

    #   This method computes the weights for the Exponential Weighted Average (EMA)

    alpha = 2./float((N_Steps+1))

    #   time_series_list is the time series of floating point values
    #       arranged in order of first element in list being earliest

    assert 0 < alpha <= 1
    
    weights = []
    
    #   Define the weights
    
    for i in range(0,N_events):
        weight_i = (1.0-alpha)**i
        weights.append(weight_i)
        
    sum_weights = sum(weights)
    weights =  [i/sum_weights for i in weights]
     
    return weights
    
    ######################################################################
    
def year_array_to_monthly_timeseries(year_events, month_interval):

    #   Get catalog location and grid box data.  Then get list of small earthquakes.
    #
    #   Next, discretize the earthquake list into time intervals of length delta_time.  
    #
    #   This discretized earthquake list is then the number of small earthquakes in time intervals
    #       of length delta_time.  
    #
    #   This method returns the list of discrete times (time_list) and the number of small earthquakes in each
    #       discrete time interval (eqs_list)
    
                                        
    year_index = 0
    max_year = max(year_events)
    min_year = min(year_events)
    
    number_year_bins = int( (max(year_events) - min(year_events))/month_interval)
    
    year_list               =   np.zeros(number_year_bins+1)        #   Same as time_list
    monthly_number_list     =   np.zeros(number_year_bins+1)        #   Same as eqs_list
    
    for i in range(len(year_list)):
        year_list[i] = min_year + float(i)*month_interval
    
    for i in range(len(year_events)):
        
        last_year_index = year_index
        year_index = int((year_events[i]-min_year)/month_interval)
        if year_index == last_year_index:
            monthly_number_list[year_index]+= 1
            
    log_monthly_number_list = [math.log10(1. + monthly_number_list[i]) for i in range(len(monthly_number_list))]
    
    return monthly_number_list, log_monthly_number_list, year_list
            
    ######################################################################
    