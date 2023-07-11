
 
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
    # Note that some part of these codes were written by the AI program chatGPT, a product of openAI
    #
    #   ---------------------------------------------------------------------------------------
    ###############################################################################################
    ###############################################################################################

import numpy as np
import random
import matplotlib.pyplot as plt
import math
from math import log10

import time

import ETASCalcV5
import ETASPlotV3
import ETASFileWriter

from datetime import datetime


    ###############################################################################################

# Set the parameters for the ETAS model
mu = 3.0                                #   Minimum magnitude
K = 1.                                  #   Omori p-value
qval = 1.5                              #   Spatial Omori exponent
sigma = 0.5                             #   Used in the spatial dependence part
ntmax =32500                           #   Number of earthquakes requested
# ntmax = 1500                          #   Number of earthquakes requested
mag_large = 6.5                         #   Magnitude of large earthquakes for re-clustering - 6.25 is good
rate_bg = 5.0                           #   Background rate
bval = 0.95                             #   GR b value
pval = 1.0                              #   Omori p-value

t = 0.0                                 #   Initial time
kk=0                                    #   Counter for large earthquakes
time_main = 0.                          #   Used in the re-clustering
m  = mu                                 #   Initial magnitude is the minimum magnitude

scale_factor = 0.999                    #   Controls the (re)clustering
step_factor_aftrshk = 0.06              #   In degrees, controls the lat-lng steps for the random walk aftershocks
BathNumber = 1.0                        #   From Bath's law
scale_mag = 2.                          #   Scaling the aftershocks is from mu+scale_mag up to mag_large
mag_threshold = mu + BathNumber      #   Only events with mags larger than this can have aftershocks

params = [mu, K, pval, qval, sigma, ntmax, mag_large, rate_bg, t, kk, time_main, bval, mag_threshold, BathNumber, step_factor_aftrshk]


#     ###############################################################################################
#     ###############################################################################################
#     

generate_and_write_new_catalog              = True
write_new_adjusted_file                     = False

plot_magnitude_time                         = True
plot_magnitude_time_clustered               = True

plot_magnitude_frequency                    = True

plot_scale_invariant_omori_plot             = True
plot_number_events_vs_time                  = False

plot_xy_positions                           = False
plot_LA_lat_lng_positions                   = False
plot_LA_seismicity_map                      = True
# 
#     ###############################################################################################
#     ###############################################################################################
    
    #   BEGIN LOCATION INPUTS
    
    ###############################################################################################
    ###############################################################################################
    
    #   Note that the input hyperparameters are in file "location_input_data.csv"
    
input_file = open("location_input_data.csv", "r")   #   At the moment, the four location choices are:
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
        min_mag = float(items[12])          #   Defines the magnitude of the small earthquakes used
            
        lambda_max_mult = float(items[13])  #   Defines the minimum monthly seismicity rate, improves skill
        lambda_min_mult = float(items[14])  #   Defines a maximum rate to approximately compensate for the existence
                                                #       of a huge earthquake that overwhelms the other events.  It removes
                                                #       some of the small earthquakes and evens out the catalog
                                                
        NSteps = float(items[15])
            
        forecast_intervals = [float(items[16])]
            
input_file.close()

    #################################################################
    #################################################################
    
    #   Overridden Variables

min_mag = 3.29

    #################################################################
    #################################################################
def write_output_files(date_file, time_file, year_file, x_events, y_events, z_events, mags, text_file_name, csv_file_name):

    nowcast_output=open(text_file_name, 'w')

    for i in range(len(year_file)):
        print(date_file[i], time_file[i], year_file[i], round(x_events[i],4),round(y_events[i],4),mags[i], \
                round(z_events[i],4), file = nowcast_output)

    nowcast_output.close()
    
    input_file = text_file_name

    output_file = csv_file_name
    
    ETASFileWriter.text_to_csv(input_file, output_file)
    
    return

    #   .............................................................
    
def recluster_times(mu, mag_large, year_events, mags, scale_factor):

    input_file_name = './Output_Files/ETAS_Output.txt'

    date_events, time_events, year_events, x_events, y_events, mags, depth_events = read_existing_catalog(input_file_name)
    
#     print('year_events', year_events)
#     print()
#     print()
    
    cluster = 'scale_invariant'
    
    mu = float(mu)
    mag_large = float(mag_large)

    mag_large_temp = mu + scale_mag
    
    range_of_mags = int( (mag_large - mu - scale_mag)/0.1  )
    print('range_of_mags', range_of_mags)
    
    year_adjusted = year_events
    duration_flag = True
    
    print('len(year_events)', len(year_events))
    
    while mag_large_temp < mag_large:
    
        print('Computing clustered events for mag_large = ', mag_large_temp)
    
        number_mainshocks = 0
        for j in range(len(mags)):
            if(mags[j] >= mag_large_temp):
                number_mainshocks += 1
        
        cycle_list = ETASCalcV5.mainshock_cycles(mags, year_adjusted, mag_large_temp)
        
        year_adjusted, aftershock_list = ETASCalcV5.omori_recluster(cycle_list, year_adjusted, mags, scale_factor)
        
        mag_large_temp += 0.1

     
#     cycle_list = ETASCalcV5.mainshock_cycles(mags, year_events, mag_large_temp)
#         
#     year_adjusted, aftershock_list = ETASCalcV5.omori_recluster(cycle_list, year_adjusted, mags, scale_factor)
#         
        
    date_adjusted                       = []
    time_adjusted                       = []
    
# 
    for i in range(len(year_adjusted)):
        decimal_year = 1960. + 63.369*year_adjusted[i]/(max(year_adjusted))
        date_of_event, time_of_day = ETASCalcV5.decimal_year_to_date(decimal_year)
        date_adjusted.append(date_of_event)
        time_adjusted.append(time_of_day)
        
    dt_avg = (max(year_adjusted) - min(year_adjusted) )/float(len(year_adjusted))
    
    print('dt_avg from recluster_times: ', dt_avg)
    
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
        
    data_file.close()  
    
    return date_events, time_events, year_events, x_events, y_events, mags, depth_events
    
    ###############################################################################################
    
def xy_to_lat_lng_LA(x_events, y_events):

    #   Put spatial center on Los Angeles, and other events within 5 degrees of that location
    
    LA_lat = 34.0522
    LA_lng = -118.2437
    
    x_mean = np.mean(x_events)
    y_mean = np.mean(y_events)
    
    delta_x = [(x_events[i]-x_mean) for i in range(len(x_events)) ] 
    delta_y = [(y_events[i]-y_mean) for i in range(len(y_events)) ] 
    
    abs_delta_x = [np.abs(delta_x[i]) for i in range(len(delta_x))]
    abs_delta_y = [np.abs(delta_y[i]) for i in range(len(delta_y))]
    
    max_x = max(abs_delta_x)
    max_y = max(abs_delta_y)
    
    adjusted_lat = [LA_lat + 5.*(delta_y[i])/max_y for i in range(len(delta_y))]
    adjusted_lng = [LA_lng + 5.*(delta_x[i])/max_x for i in range(len(delta_x))]

    return adjusted_lng, adjusted_lat
    
    ###############################################################################################
    
def decimal_year(date):

    start_date = datetime(date.year, 1, 1)
    end_date = datetime(date.year + 1, 1, 1)
    year_duration = (end_date - start_date).total_seconds()
    current_duration = (date - start_date).total_seconds()
    decimal_years = date.year + current_duration / year_duration
    
    return decimal_years

    ###############################################################################################
    ###############################################################################################
    
if generate_and_write_new_catalog:

    dt_avg, mags, year_events, x_events, y_events, z_events = ETASCalcV5.generate_catalog(params)
    
    nowcast_output_catalog          = []
    date_file                       = []
    year_file                       = []
    time_file                       = []
    
    # Get today's date
    today = datetime.today()

    # Calculate decimal years
    todays_decimal_year = decimal_year(today)
    
    print()
    print('Today, todays_decimal_year: ', today, todays_decimal_year)
    print()
# 
    for i in range(len(year_events)):
    
#         decimal_year = 1960. + 63.369*year_events[i]/(max(year_events))
        decimal_year = 1960. + (todays_decimal_year - 1960.0)*year_events[i]/(max(year_events))
        year_file.append(round(decimal_year,8))
        date_of_event, time_of_day = ETASCalcV5.decimal_year_to_date(decimal_year)
        date_file.append(date_of_event)
        time_file.append(time_of_day)
    
    text_file_name = './Output_Files/ETAS_Output.txt'
    csv_file_name  = './Output_Files/ETAS_Output.csv'
    
    write_output_files(date_file, time_file, year_file, x_events, y_events, z_events, mags, text_file_name, csv_file_name)
    
    write_new_adjusted_file  = True     #   Also, write a new adjusted file
#     
    #   .............................................................
    
if write_new_adjusted_file:

    input_file_name = './Output_Files/ETAS_Output.txt'
    
    date_events, time_events, year_events, x_events, y_events, mags, z_events = read_existing_catalog(input_file_name)
    

    year_events = [float(year_events[i]) for i in range(len(year_events))]
    mags        = [float(mags[i]) for i in range(len(mags))]
    
#     print()
#     print('Enter Max magnitude for clustering, which must be <= ', max(mags))
#     print()
#     mag_large = float(input('Enter mag_large: '))
    
    number_mainshocks = 0
    for i in range(len(mags)):
        if mags[i] >= mag_large:
            number_mainshocks += 1
            
    print('>>>>  number_mainshocks', number_mainshocks)
    
    mu = min(mags)
    
    #   The following does the scale-invariant reclustering
    dt_avg, date_adjusted, time_adjusted, year_adjusted, aftershock_list = recluster_times(mu, mag_large, year_events, mags, scale_factor)
    
    #   These two function calls below only do a single reclustering
    
#     cycle_list = ETASCalcV5.mainshock_cycles(mags, year_events, mag_large)
#     
#     year_adjusted, aftershock_list = ETASCalcV5.omori_recluster(cycle_list, year_events, mags, scale_factor)
    
    date_file                       = []
    year_file                       = []
    time_file                       = []
    
    for i in range(len(year_adjusted)):
        decimal_year = year_adjusted[i]
        year_file.append(round(decimal_year,8))
        date_of_event, time_of_day = ETASCalcV5.decimal_year_to_date(decimal_year)
        date_file.append(date_of_event)
        time_file.append(time_of_day)
        
    txt_file_name   = './Output_Files/ETAS_Scale_Invariant_Output.txt'
    csv_file_name   = './Output_Files/ETAS_Scale_Invariant_Output.csv'
    
    print()

    write_output_files(date_file, time_file, year_file, x_events, y_events, z_events, mags, txt_file_name, csv_file_name)
    
    #   Find and print the large events
    
    mag_large_events    =   []
    year_large_events   =   []
    
    for i in range(len(year_file)):
        if mags[i] >= mag_large:
            mag_large_events.append(mags[i])
            year_large_events.append(year_file[i])
            print('For Large Events, Year: ', round(year_file[i],3), ' Magnitude: ', round(mags[i],2))
            
#     
    #   .............................................................
    
if plot_magnitude_time:

    input_file_name = './Output_Files/ETAS_Output.txt'

    date_events, time_events, year_events, x_events, y_events, mags, z_events = read_existing_catalog(input_file_name)

    ETASPlotV3.plot_mag_timeseries_raw(year_events, mags)
    
    #   .............................................................
    
if plot_magnitude_time_clustered:

    input_file_name = './Output_Files/ETAS_Scale_Invariant_Output.txt'

    date_events, time_events, year_events, x_events, y_events, mags, z_events = read_existing_catalog(input_file_name)

    ETASPlotV3.plot_mag_timeseries_clustered(year_events, mags, mu, mag_large, scale_factor, scale_mag)
    
    #   .............................................................

if plot_magnitude_frequency:

    input_file_name = './Output_Files/ETAS_Output.txt'
#     input_file_name   = './Output_Files/ETAS_Scale_Invariant_Output.txt'
    
    date_events, time_events, year_events, x_events, y_events, mags, depth_events = read_existing_catalog(input_file_name)

    ETASPlotV3.plot_GR(mags)
    
       #   .............................................................
       
if plot_xy_positions:

    input_file_name = './Output_Files/ETAS_Output.txt'                      #   Original XY positions
    
    date_events, time_events, year_events, x_events, y_events, mags, z_events = read_existing_catalog(input_file_name)
    
    x_events = [float(x_events[i]) for i in range(len(x_events))]
    y_events = [float(y_events[i]) for i in range(len(y_events))]
    mags        = [float(mags[i]) for i in range(len(mags))]
    
    mu = min(mags)

    ETASPlotV3.plot_xy_locations(x_events,y_events, mags, mu) 
    
    #   .............................................................
    
if plot_LA_lat_lng_positions:

    input_file_name   = './Output_Files/ETAS_Scale_Invariant_Output.txt'    #   Fake LA positions

    date_events, time_events, year_events, x_events, y_events, mags, z_events = read_existing_catalog(input_file_name)
    
    x_events = [float(x_events[i]) for i in range(len(x_events))]
    y_events = [float(y_events[i]) for i in range(len(y_events))]
    mags        = [float(mags[i]) for i in range(len(mags))]
    
    mu = min(mags)

    ETASPlotV3.plot_lat_lng_locations(x_events,y_events, mags, mu) 
    
    #   .............................................................
    
if plot_scale_invariant_omori_plot:

       #  Get the magnitude and year files

    input_file_name   = './Output_Files/ETAS_Scale_Invariant_Output.txt'

    date_events, time_events, year_events, x_events, y_events, mags, z_events = read_existing_catalog(input_file_name)
    
    
    year_events = [float(year_events[i]) for i in range(len(year_events))]
    mags        = [float(mags[i]) for i in range(len(mags))]
    
#     print()
#     print('Enter Max magnitude for clustering, which must be <= ', max(mags))
#     print()
#     mag_large = float(input('Enter mag_large: '))

    print()
    print('Magnitude for Re-clustering is: ', mag_large)
    print()
    
    number_mainshocks = 0
    for i in range(len(mags)):
        if mags[i] >= mag_large:
            number_mainshocks += 1
            
    print('>>>>  number_mainshocks', number_mainshocks)
    
    cycle_list = ETASCalcV5.mainshock_cycles(mags, year_events, mag_large)
    
    temp_scale_factor = 1.0  #   We just need the cycle_list for the adjusted catalog to get the aftershock_list
                             #       So set the scale_factor temporarily = 1
    
    year_adjusted, aftershock_list = ETASCalcV5.omori_recluster(cycle_list, year_events, mags, temp_scale_factor)
    
    dt_avg = (max(year_adjusted)-min(year_adjusted))/float(len(year_adjusted))
    
    #   Below we use the original scale factor
    ETASPlotV3.plot_scale_invariant_aftershock_decay(aftershock_list, number_mainshocks, mag_large, dt_avg, scale_factor)
    
    date_events, time_events, year_events, x_events, y_events, mags, z_events = read_existing_catalog(input_file_name)
    
    ETASPlotV3.plot_mag_timeseries_clustered(year_events, mags, mu, mag_large, scale_factor, scale_mag)

    #   .............................................................

if plot_LA_seismicity_map:

    input_file_name   = './Output_Files/ETAS_Scale_Invariant_Output.txt'    #   Fake LA positions

    date_events, time_events, year_events, x_events, y_events, mags, z_events = read_existing_catalog(input_file_name)
    
    x_events = [float(x_events[i]) for i in range(len(x_events))]
    y_events = [float(y_events[i]) for i in range(len(y_events))]
    mags        = [float(mags[i]) for i in range(len(mags))]
    
    mu = min(mags)

    ETASPlotV3.map_seismicity(NELat_local, NELng_local, SWLat_local, SWLng_local, plot_start_year, \
        Location, catalog, mag_large, min_mag, mu, forecast_intervals, \
        date_events, time_events, year_events, x_events, y_events, mags, z_events) 
        
    #   .............................................................


