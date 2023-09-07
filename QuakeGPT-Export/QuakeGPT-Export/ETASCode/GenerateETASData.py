    #
    #   This code is based on the ETAS_ChatGPT_V2.13 code
    #
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
import sys

import numpy as np
import random
import math
from math import log10

import matplotlib.pyplot as plt
from matplotlib import gridspec


import time

import ETASCalcV5
import ETASFileWriter
import ETASFileMethods
import ETASPlotV3

import Compute_EMA_Timeseries

from datetime import datetime

import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.simplefilter("error")  #   Prints out location of error


    ###############################################################################################

# Set the parameters for the ETAS model
mu = 3.29                               #   Minimum magnitude
K = 1.0                                 #   Aftershock productivity parameter >>>>>>>>>>> Usually have K = 1
alpha = 1.0                             #   Exponent in the productivity relation, multiplies bval
qval = 1.5                              #   Spatial Omori exponent
sigma = 0.5                             #   Could be used in the spatial dependence part (not used in this code)
mag_large = 7.0                         #   Magnitude of large earthquakes for re-clustering or for stacked aftershock plots
rate_bg = 5.0                           #   Background rate
bval = 0.9                              #   GR b value, usually bval=0.9
pval = 1.2                              #   Omori p-value       >>>>>>>>> Usually pval = 1.2 or so
corr_length = 100                       #   Parameter in the spatial Omori relation (km)
corr_time   = 1                         #   Parameter in the temporal Omori relation (natural time)
dt_ratio_exp = 1.5                      #   Used in the equation that computes time to next event

t = 0.0                                 #   Initial time
kk=0                                    #   Counter for large earthquakes
time_main = 0.                          #   Used in the re-clustering
m  = mu                                 #   Initial magnitude is the minimum magnitude

scale_factor = 0.004                    #   Controls the (re)clustering, this is basically 2 x standard deviation - was 0.004
step_factor_aftrshk = 0.04              #   In degrees, controls the lat-lng steps for the random walk aftershocks

BathNumber = 1.2                        #   Bath's law value.  Note that this can be used to determine the
                                        #       the effective number of aftershocks for a given mainshock magnitude
                                        #       and thus the ratio of aftershocks to mainshocks
                                        #       Making this larger, say, 1.2, results in more large mainshocks, so it 
                                        #       trades off against a smaller b-value

scale_mag = 1.                          #   Scaling the aftershocks is from mu+scale_mag up to mag_large
mag_threshold = mu + BathNumber         #   Only events with mags larger than this can have aftershocks
plot_params =  True                     #   Used to display the params in the boxes on the plots
plot_USGS_data  = False                 #   If we want to plot real data, first converts the
                                        #       USGS files to the correct format and file name then runs the functions
                                        
pfit_lo = 80                            #   Low value of the parameters to fit a line to the Omori scaling plot
pfit_hi = 600                           #   High value of the parameters to fit a line to the Omori scaling plot

NSteps = 36                             #   Number of averaging steps used in the EMA

bval_std = 0.05                         #   Standard deviation of b-value  - use .3 with line fit
mag_bin_size  = 0.1                      #   Magnitude binning size
mu_thresh = mu                         #

download_USGS_catalog = False

#     ###############################################################################################
#     ###############################################################################################
#     
ntmax =int(sys.argv[1]) + 500               # Number of earthquakes requested
catalog_start_year = float(sys.argv[2])     # Decimal date on which catalog starts
catalog_end_year = float(sys.argv[3])       # Decimal date on which catalog ends

delta_years = catalog_end_year - catalog_start_year    

                ##################################################################

generate_and_write_new_catalog              = True
write_new_adjusted_file                     = False
write_transformer_test_train_files          = True
# 
    ###############################################################################################################################
    ###############################################################################################################################
                                                                                                                # READ ETAS DATA
    
    #   Note that the input hyperparameters are in file "location_input_data.csv"
    
input_file = open("./ETASCode/location_input_data.csv", "r")   #   At the moment, the four location choices are:
                                                        #   Los Angeles, Tokyo, Greece and Chile
    
catalog = 'ETAS'  #   Specifies the name of the catalog, the current region of interest.  Hyperparameter data is stored
                            #       in the file:   location_input_data.csv
    
for line in input_file:
    items = line.strip().split(',')

    if items[0] == catalog:
        Location = items[1]
        plot_start_year = float(items[2])
            
        center_lat = float(items[3])
        center_lng = float(items[4])
            
        delta_lat = float(items[5])     #   For the base catalog
        delta_lng = float(items[6])
            
        NELng = center_lng + delta_lng
        SWLng = center_lng - delta_lng
        NELat = center_lat + delta_lat
        SWLat = center_lat - delta_lat
            
            
        delta_deg = float(items[9]) #   For the regional catalog, which is a subset of base catalog
            
        delta_deg_lat = delta_deg
        delta_deg_lng = delta_deg
            
        NELng_local = center_lng + delta_deg
        SWLng_local = center_lng - delta_deg
        NELat_local = center_lat + delta_deg
        SWLat_local = center_lat - delta_deg

#         lambda_max_mult = float(items[13])  #   Defines the minimum monthly seismicity rate, improves skill
#         lambda_min_mult = float(items[14])  #   Defines a maximum rate to approximately compensate for the existence
#                                                 #       of a huge earthquake that overwhelms the other events.  It removes
#                                                 #       some of the small earthquakes and evens out the catalog
                                                
        NSteps = float(items[15])
            
        forecast_intervals = [float(items[16])]
            
input_file.close()

    ###############################################################################################################################
    ###############################################################################################################################
                                                                                                                # DEFINE VARIABLES

    #   Overridden Variables
    
lambda_max_mult = 1000000.  #   Defines the minimum monthly seismicity rate, improves skill
lambda_min_mult = 1.0       #   Defines the minimum monthly rate, adjustable parameter to improve skill - or use 1.2
min_mag = 3.29
min_map_mag = 3.29

params = [mu, K, pval, qval, sigma, ntmax, mag_large, rate_bg, t, kk, time_main, bval, mag_threshold, \
        BathNumber, step_factor_aftrshk, corr_length, corr_time, alpha, dt_ratio_exp, scale_factor, \
        plot_params, pfit_lo, pfit_hi, bval_std, mag_bin_size, mu_thresh]
        


    ###############################################################################################################################
    ###############################################################################################################################
                                                                                                            # DEFINE CALC METHODS


def define_USGS_parameters():

    #   Note that the input hyperparameters are in file "location_input_data.csv"
    
    input_file = open("location_input_data.csv", "r")   #   At the moment, the four location choices are:
                                                        #   Los Angeles, Tokyo, Greece and Chile
    
    catalog = 'LosAngeles'  #   Specifies the name of the catalog, the current region of interest.  Hyperparameter data is stored
                            #       in the file:   location_input_data.csv
    
    for line in input_file:
        items = line.strip().split(',')

        if items[0] == catalog:
            Location = items[1]
            plot_start_year = float(items[2])
            
            center_lat = float(items[3])
            center_lng = float(items[4])
            
            delta_lat = float(items[5])     #   For the base catalog
            delta_lng = float(items[6])
            
            NELng = center_lng + delta_lng
            SWLng = center_lng - delta_lng
            NELat = center_lat + delta_lat
            SWLat = center_lat - delta_lat
            
            max_depth = float(items[7])
            completeness_mag = float(items[8])
            
            delta_deg = float(items[9]) #   For the regional catalog, which is a subset of base catalog
            
            delta_deg_lat = delta_deg
            delta_deg_lng = delta_deg
            
            NELng_local = center_lng + delta_deg
            SWLng_local = center_lng - delta_deg
            NELat_local = center_lat + delta_deg
            SWLat_local = center_lat - delta_deg
            
            grid_size = float(items[10])    #   Spatial grid box size in Degrees
            
            mag_large = float(items[11])        #   Defines the magnitude used for the ROC and Precision tests
            min_mag = float(items[12])        #   Defines the magnitude of the small earthquakes used
            
#             lambda_max_mult = float(items[13])  #   Defines the minimum monthly seismicity rate, improves skill
#             lambda_min_mult = float(items[14])  #   Defines a maximum rate to approximately compensate for the existence
                                                #       of a huge earthquake that overwhelms the other events.  It removes
                                                #       some of the small earthquakes and evens out the catalog
                                                
#             NSteps = float(items[15])
            
            forecast_intervals = [float(items[16])]
            
    input_file.close()

    return NELat, NELng, SWLat, SWLng, completeness_mag, plot_start_year, \
            NELat_local, NELng_local, SWLat_local, SWLng_local, min_mag, max_depth
    
    #   .............................................................
    
def write_output_files(date_file, time_file, year_file, x_events, y_events, z_events, mags, aftshk_bg_list,\
        text_file_name):

    nowcast_output=open(text_file_name, 'w')

    for i in range(len(year_file)):
        print(date_file[i], time_file[i], year_file[i], round(x_events[i],4),round(y_events[i],4),mags[i], \
                round(z_events[i],4), aftshk_bg_list[i], file = nowcast_output)

    nowcast_output.close()
    
    return

    #   .............................................................
    
def recluster_times(mu, mag_large, year_events, mags, scale_factor):

    input_file_name = './ETASCode/Output_Files/ETAS_Output.txt'

    date_events, time_events, year_events, x_events, y_events, mags, depth_events, label_events = \
            read_existing_catalog(input_file_name)
    
    cluster = 'scale_invariant'
    
    mu = float(mu)
#     mag_large = float(mag_large)

    mag_large = max(mags) - BathNumber       #   Largest aftershock possible
    mag_small_temp = mu + scale_mag     #   Smallest mainshock that can have aftershocks 
    
    range_of_mags = int( (mag_large - mag_small_temp)/0.1  )
    print('range_of_mags', range_of_mags)
    
    year_adjusted = year_events
    duration_flag = True
    
    print('len(year_events)', len(year_events))
    
    while mag_small_temp < mag_large:
    
        print('Computing clustered events for mag_large = ', round(mag_small_temp,2))
    
        number_mainshocks = 0
        for j in range(len(mags)):
            if(mags[j] >= mag_small_temp):
                number_mainshocks += 1
        
        cycle_list, label_cycle_list = ETASCalcV5.mainshock_cycles(mags, year_adjusted, label_events, mag_small_temp)
        
        year_adjusted, aftershock_list = ETASCalcV5.omori_recluster(cycle_list, label_cycle_list, year_adjusted, mags, scale_factor)
        
        mag_small_temp += 0.1

    date_adjusted                       = []
    time_adjusted                       = []
    
# 
    for i in range(len(year_adjusted)):
        decimal_year = catalog_start_year + delta_years*year_adjusted[i]/(max(year_adjusted))
        date_of_event, time_of_day = ETASCalcV5.decimal_year_to_date(decimal_year)
        date_adjusted.append(date_of_event)
        time_adjusted.append(time_of_day)
        
    dt_avg = (max(year_adjusted) - min(year_adjusted) )/float(len(year_adjusted))
    
#     print('dt_avg from recluster_times: ', dt_avg)
    
#     
    return dt_avg, date_adjusted, time_adjusted, year_adjusted, aftershock_list
    
    ###############################################################################################

def read_existing_catalog(input_file_name):

    mags                    =   []
    date_events             =   []
    time_events             =   []
    year_events             =   []
    depth_events            =   []
    y_events                =   []
    x_events                =   []
    label_events            =   []

    data_file = open(input_file_name,"r")
    
    for line in data_file:
        items = line.strip().split()
        
        date_events.append(items[0])
        time_events.append(items[1])
        year_events.append(float(items[2]))
        x_events.append(float(items[3]))
        y_events.append(float(items[4]))
        mags.append(float(items[5]))
        depth_events.append(float(items[6]))
        label_events.append(items[7])
        
    data_file.close()  
    
    return date_events, time_events, year_events, x_events, y_events, mags, depth_events, label_events
    
    ###############################################################################################
    
def year_fraction(date):

    start_date = datetime(date.year, 1, 1)
    end_date = datetime(date.year + 1, 1, 1)
    year_duration = (end_date - start_date).total_seconds()
    current_duration = (date - start_date).total_seconds()
    decimal_years = date.year + current_duration / year_duration
    
    return decimal_years
    
    ###############################################################################################
    
def calc_test_train_data(year_events):

    month_interval          = 0.07692   #   We use a time interval of "months" 
                                        #       where 1 month = 4 weeks = "lunar month", 52 weeks/year
                                        
                                        
    monthly_number_list, log_monthly_number_list, year_list = \
            Compute_EMA_Timeseries.year_array_to_monthly_timeseries(year_events, month_interval)
            
    monthly_number_list_raw = monthly_number_list
    
    #   ---------------------------------------------------
    
    data_start_year = year_list[0]
    year_array = year_list
    
    eqs_list = monthly_number_list.copy()
    
    timeseries_EMA_reduced, time_list_reduced, log_number_reduced, min_rate, max_rate = \
            Compute_EMA_Timeseries.build_EMA_timeseries(eqs_list, year_list, NSteps, \
            data_start_year, lambda_min_mult, lambda_max_mult)
            
    return monthly_number_list_raw, year_list, timeseries_EMA_reduced, time_list_reduced, log_number_reduced
    
    ###############################################################################################
        
def calc_omori_plot_data():

    input_file_name =  './ETASCode/Output_Files/ETAS_Scale_Invariant_Output.txt'

    date_events, time_events, year_events, x_events, y_events, mags, depth_events, label_events = \
            read_existing_catalog(input_file_name)
            
    plot_data_start_year = 1970.

    mag_large_omori = 5.5
    print()
    print('Magnitude for Re-clustering is: ', mag_large_omori)
    print()
    
    number_mainshocks = 0
    for i in range(len(mags)):
        if mags[i] >= mag_large_omori:
            number_mainshocks += 1
            
    print('>>>>  number_mainshocks', number_mainshocks)
    
    cycle_list, label_cycle_list = ETASCalcV5.mainshock_cycles(mags, year_events,label_events, mag_large_omori)
    
    temp_scale_factor = 0.0  #   We just need the cycle_list for the adjusted catalog to get the aftershock_list
                             #       So set the scale_factor temporarily = 0
    
    year_adjusted, aftershock_list = ETASCalcV5.omori_recluster(cycle_list, label_cycle_list, year_events, mags, temp_scale_factor)
    
    dt_avg = (max(year_adjusted)-min(year_adjusted))/float(len(year_adjusted))
    
    number_aftershocks = 0
    
    for i in range(len(label_events)):
        if label_events[i] == 'a':
            number_aftershocks += 1

    print()
    fraction_aftershocks = round( 100.0*float(number_aftershocks)/float(len(label_events)), 2)
    print('Fraction Aftershocks: ', str(fraction_aftershocks) + '%' )
    print()

    return aftershock_list, label_cycle_list, number_mainshocks, mag_large_omori, dt_avg, scale_factor, params
    
    ###############################################################################################################################
    ###############################################################################################################################
                                                                                                                        # PLOTS

def simple_data_plot(time_list_reduced, log_number_reduced):

    input_file_name =  './ETASCode/Output_Files/ETAS_Scale_Invariant_Output.txt'

    date_events, time_events, year_events, x_events, y_events, mags, depth_events, label_events = \
            read_existing_catalog(input_file_name)
            
    plot_data_start_year = 1970.
    
    max_value = max(log_number_reduced)
    
    index_1970 = 0
    while time_list_reduced[index_1970] < plot_data_start_year:
        index_1970 += 1
    
    log_number_reduced_reversed = [max_value - log_number_reduced[i] for i in range(len(log_number_reduced))]
    
    fig, ax = plt.subplots()
         
    ax.plot(time_list_reduced[index_1970:], log_number_reduced_reversed[index_1970:], '-', lw=0.75, \
            color='blue', zorder=4, label=sys.argv[4])
    
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    min_plot_line = [ymin for i in range(len(time_list_reduced[index_1970:]))]
    
    ax.fill_between(time_list_reduced[index_1970:] , min_plot_line, log_number_reduced_reversed[index_1970:], \
            color='c', alpha=0.1, zorder=0)
            
    for i in range(len(year_events)):

        if float(mags[i]) >= 6.0 and float(mags[i]) < 7.0  and float(year_events[i]) >= plot_start_year:
            x_eq = [year_events[i], year_events[i]]
            y_eq = [ymin,ymax]
            
            ax.plot(x_eq, y_eq, linestyle='dotted', color='k', lw=0.75, zorder=2)
            
    ax.plot([],[], linestyle='dotted', color='k', lw=0.7, zorder=2, label = '7.0 $>$ M $\geq$ 6.0')
            
    for i in range(len(year_events)):

        if float(mags[i]) >= 7.0 and float(year_events[i]) >= plot_start_year:
            x_eq = [year_events[i], year_events[i]]
            y_eq = [ymin,ymax]
            
            ax.plot(x_eq, y_eq, linestyle='--', color='r', lw=0.75, zorder=2)
            
    ax.plot([],[], linestyle='--', color='r', lw=0.7, zorder=2, label='M $\geq$ 7.0')
    
    max_plot_line = [ymax for i in range(len(time_list_reduced[index_1970:]))]
    
    ax.fill_between(time_list_reduced[index_1970:] , log_number_reduced_reversed[index_1970:], max_plot_line, \
            color='white', alpha=1.0, zorder=3)
            
    title_text = 'Data Set'
    leg = ax.legend(loc = 'upper right', title=title_text, fontsize=8)
    leg.set_title(title_text,prop={'size':10})   #   Set the title text font size
    
    SupTitle_text = 'ETAS Earthquake Potential State $\Theta(t)$ vs. Time: ' 

    plt.suptitle(SupTitle_text, fontsize=12)
    
    Title_text = 'Within ' + str(round(delta_deg_lat,2)) + '$^o$ Latitude and ' + \
            str(round(delta_deg_lng,2)) + '$^o$ Longitude of Simulated Los Angeles'
            
    plt.title(Title_text, fontsize=9)
    
    plt.ylabel('Large Earthquake Potential State $\Theta(t)$', fontsize = 11)
    plt.xlabel('Time (Year)', fontsize = 11)
    
    data_string_title = 'EMA' + '_NSTP' + str(NSteps) + '_MM' + str(min_mag) + '_CF' + str(lambda_min_mult) 

    figure_name = './ETASCode/Data/SEISR_' + data_string_title + '_' + str(plot_data_start_year) + '_'+ sys.argv[4] + '.png'
    plt.savefig(figure_name,dpi=300)
    
    plt.show()
    
    plt.close('all')

    return
    
    ###############################################################################################
    
def plot_GR(mags):

    mag_bins, freq_mag_bins_pdf, freq_mag_bins_sdf, log_freq_mag_bins = ETASCalcV5.freq_mag(mags)
    
    freq_mag_bins = [10**(log_freq_mag_bins[i]) for i in range(len(log_freq_mag_bins))]
    
    fig, ax = plt.subplots()

    ax.plot(mag_bins, freq_mag_bins, marker = 'o', color ='blue', markersize = 4, lw = 0)
    
    #   Compute b-value
    mag_lo = mu                 #   log10(10)
    mag_hi = 6.5          #   log10(100)

    mag_line_data, freq_line_data, slope = ETASCalcV5.calc_b_value(log_freq_mag_bins, mag_bins, mag_lo, mag_hi)
    
    bfit = - slope

#   Plot the best fitting line
    ax.plot(mag_line_data, freq_line_data, '--', color = 'red', markersize = 0, lw = 1.5)
    
    ax.set_yscale('log')
    
    textstr1 =   ' Minimum Magnitude: ' + str(mu) +\
                '\n K: ' + str(K) +\
                '\n Alpha: ' + str(K) +\
                '\n Omori p-value: ' + str(pval)+\
                '\n q-value: ' + str(qval)+\
                '\n b-value: ' + str(bval)+\
                '\n Baths Law Number: ' + str(BathNumber)+\
                '\n Aftershock Step: ' + str(step_factor_aftrshk)+ '$^o$'+\
                '\n Correlation Length: ' + str(corr_length)+ ' $Km$'\
                '\n Correlation Time: ' + str(corr_time) + ' Year' +\
                '\n Cluster Factor: ' + str(scale_factor) +\
                '\n Rate Ratio Exp:' + str(dt_ratio_exp)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.75)

    # place a text box at lower left
    if plot_params:
        ax.text(0.015, 0.02, textstr1, transform=ax.transAxes, fontsize=5,\
            verticalalignment='bottom', horizontalalignment = 'left', bbox=props, linespacing = 1.8)
        
    textstr2 =      ' b-value (from fit): ' + str(round(bfit,2))
                    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.75)
    ax.text(0.85, 0.975, textstr2, transform=ax.transAxes, fontsize=8,\
        verticalalignment='top', horizontalalignment = 'center', bbox=props, linespacing = 1.8)
 
    SupTitle_text = 'Magnitude-Frequency Diagram for ' + str(len(mags)) +  ' ETAS Earthquakes'
    plt.suptitle(SupTitle_text, fontsize=10)
    
    if plot_params:
        Title_text = 'Hyper-Parameter Values Displayed in Legend'
        plt.title(Title_text, fontsize=8)

    plt.xlabel('Magnitude of ETAS Earthquakes')
    plt.ylabel('$Log_{10}$(Number)')
    
    FigureName = './ETASCode/Data/GR_Mag_Freq.png'
    plt.savefig(FigureName, dpi=200)

    plt.show()
    
    plt.close()
    
    return 
    
    ###############################################################################################################################
    ###############################################################################################################################
                                                                                                                # WRITE FILES
    
if generate_and_write_new_catalog:

    t_time, dt_avg, mags, year_events, x_events, y_events, z_events, aftshk_bg_list = ETASCalcV5.generate_catalog(params)
    
    nowcast_output_catalog          = []
    date_file                       = []
    year_file                       = []
    time_file                       = []
    
    # Get today's date
    today = datetime.today()

    # Calculate decimal years
    todays_decimal_year = year_fraction(today)
    
    print()
    print('Today, todays_decimal_year: ', today, todays_decimal_year)
    print()
# 
    for i in range(len(year_events)):
    
        decimal_year = catalog_start_year + delta_years * year_events[i]/(max(year_events))
        year_file.append(round(decimal_year,8))
        date_of_event, time_of_day = ETASCalcV5.decimal_year_to_date(decimal_year)
        date_file.append(date_of_event)
        time_file.append(time_of_day)
    
    text_file_name = './ETASCode/Output_Files/ETAS_Output.txt'
    
    write_output_files(date_file, time_file, year_file, x_events, y_events, z_events, mags, aftshk_bg_list,\
            text_file_name)
    
    write_new_adjusted_file  = True     #   Also, write a new adjusted file
    
    #   print ratio of compute time to real time
    
    compute_time = t_time
    catalog_time = delta_years
    
    print()
    print('catalog_time, compute_time, catalog_time/compute_time: ', \
                round(catalog_time,4), round(compute_time,4), round(catalog_time/compute_time,4) )
    print()
#     
    #   .............................................................
    
if write_new_adjusted_file:

    input_file_name = './ETASCode/Output_Files/ETAS_Output.txt'
    
    date_events, time_events, year_events, x_events, y_events, mags, z_events, label_events = \
            read_existing_catalog(input_file_name)
    
    year_events = [float(year_events[i]) for i in range(len(year_events))]
    mags        = [float(mags[i]) for i in range(len(mags))]
    
    number_mainshocks = 0
    for i in range(len(mags)):
        if mags[i] >= mag_large:
            number_mainshocks += 1
            
    print('>>>>  number_mainshocks', number_mainshocks)
    
    mu = min(mags)
    
    #   The following does the scale-invariant reclustering
    dt_avg, date_adjusted, time_adjusted, year_adjusted, aftershock_list = recluster_times(mu, mag_large, year_events, mags, scale_factor)
    
    #   These two function calls below only do a single reclustering
    
    date_file                       = []
    year_file                       = []
    time_file                       = []
    
    for i in range(len(year_adjusted)):
        decimal_year = year_adjusted[i]
        year_file.append(round(decimal_year,8))
        date_of_event, time_of_day = ETASCalcV5.decimal_year_to_date(decimal_year)
        date_file.append(date_of_event)
        time_file.append(time_of_day)
        
    txt_file_name   = './ETASCode/Output_Files/ETAS_Scale_Invariant_Output.txt'
    
    print()

    write_output_files(date_file, time_file, year_file, x_events, y_events, z_events, mags, label_events, \
            txt_file_name)
    
    #   Find and print the large events
    
    mag_large_events    =   []
    year_large_events   =   []
    
    for i in range(len(year_file)):
        if mags[i] >= mag_large:
            mag_large_events.append(mags[i])
            year_large_events.append(year_file[i])
            print('For Large Events, Year: ', round(year_file[i],3), ' Magnitude: ', round(mags[i],2),\
                    ' Longitude: ', round(x_events[i], 4), ' Latitude: ', round(y_events[i], 4), ' Time Step: ', i)
            
    #   .............................................................
    
if write_transformer_test_train_files:

    input_file_name = './ETASCode/Output_Files/ETAS_Scale_Invariant_Output.txt'

    date_events, time_events, year_events, x_events, y_events, mags, z_events, label_events = \
            read_existing_catalog(input_file_name)
            
    monthly_number_list, year_list, timeseries_EMA_reduced, time_list_reduced, log_number_reduced = \
         calc_test_train_data(year_events)
       
    output_file = open('./ETASCode/Output_Files/test_train_data.txt', 'w')
    
    max_value = max(log_number_reduced)
    log_number_reduced_reversed = [max_value - log_number_reduced[i] for i in range(len(log_number_reduced))]
    
    #   Do we want to rescale the log_number_reduced_reversed data here to lie between [0,1]?
    
    for i in range(len(monthly_number_list)):
        print(monthly_number_list[i], log_number_reduced_reversed[i], time_list_reduced[i], file=output_file)
    
    output_file.close()
    
    #   .............................................................
    #
    #   Plots
    
    simple_data_plot(time_list_reduced, log_number_reduced)
    
    plot_GR(mags)
    
    ETASPlotV3.plot_mag_timeseries_clustered(year_events, mags, params)
    
    aftershock_list, label_cycle_list, number_mainshocks, mag_large_omori, dt_avg, scale_factor, params = \
        calc_omori_plot_data()

    ETASPlotV3.plot_omori_aftershock_decay(aftershock_list, label_cycle_list, \
            number_mainshocks, mag_large_omori, dt_avg, scale_factor, params)
            
    #   .............................................................


    
    
