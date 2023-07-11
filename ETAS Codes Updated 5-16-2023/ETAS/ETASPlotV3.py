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

import numpy as np
import random
import math
from math import log10

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature, LAND, COASTLINE

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.patches import Rectangle

from matplotlib.offsetbox import AnchoredText
from matplotlib.image import imread
import matplotlib.ticker as mticker

import ETASCalcV5

    ###############################################################################################

def plot_mag_timeseries_raw(year_events, mags):

    mag_list = []
    year_list = []
    
    
    fig, ax = plt.subplots()
    
#     for i in range(int(0.9*len(time_events)), len(time_events)):
    for i in range(int(len(year_events))):
        mag_list.append(mags[i])
        year_list.append(year_events[i])

    ax.plot(year_list, mag_list, marker = 'o', color ='blue', markersize = 3, lw = 0.0)
    
#     plt.plot(time_events, mags, marker = 'o', color ='blue', markersize = 4, lw = 0.2)
    # plt.scatter(time_events, mags, markersize = 0.1)

    Title_text = 'Magnitude-Time Diagram for ' + str(len(mags)) +  ' ETAS Earthquakes (Raw Data)'
    plt.title(Title_text, fontsize=10)

    plt.xlabel('Time Index')
    plt.ylabel('ETAS Earthquake Magnitude')
    
    FigureName = './Figures/mag_time_raw.png'
          
    plt.savefig(FigureName, dpi=200)
    
    #plt.show()
    
    plt.close()
    
    return
    
    ###############################################################################################
    
def plot_mag_timeseries_clustered(year_events, mags, mu, mag_large, scale_factor, scale_mag):

    mag_list = []
    year_list = []
    
    fig, ax = plt.subplots()
    
#     for i in range(int(0.9*len(time_events)), len(time_events)):
    for i in range(len(year_events)):
        mag_list.append(mags[i])
        year_list.append(year_events[i])
    
    ax.plot(year_list, mag_list, marker = 'o', color ='blue', markersize = 3, lw = 0.0)
    
#     plt.plot(time_events, mags, marker = 'o', color ='blue', markersize = 4, lw = 0.2)
    # plt.scatter(time_events, mags, markersize = 0.1)
    
    SupTitle_text = 'Magnitude-Time Diagram for ' + str(len(mags)) +  ' ETAS Earthquakes'
    plt.suptitle(SupTitle_text, fontsize = 12)
        
    Title_text = 'Scale Invariant Clustering for M' + str(mu+scale_mag) + ' to M' + str(mag_large) \
                + ' Events, Scale Factor ' + str(scale_factor)
    plt.title(Title_text, fontsize = 10)

    plt.xlabel('Time Index')
    plt.ylabel('ETAS Earthquake Magnitude')
    
    FigureName = './Figures/mag_time_clustered.png'
          
    plt.savefig(FigureName, dpi=200)
    
    #plt.show()
    
    plt.close()
    
    return
    
    ###############################################################################################
    
def plot_GR(mags):

    mag_bins, freq_mag_bins_pdf, freq_mag_bins_sdf, log_freq_mag_bins = ETASCalcV5.freq_mag(mags)
    plt.plot(mag_bins, log_freq_mag_bins, marker = 'o', color ='blue', markersize = 4, lw = 0)

    Title = 'Magnitude-Frequency Diagram for ' + str(len(mags)) +  ' ETAS Earthquakes'
    plt.title(Title, fontsize=10)

    plt.xlabel('Magnitude of ETAS Earthquakes')
    plt.ylabel('$Log_{10}$(Number)')
    
    FigureName = './Figures/GR_mag_freq.png'
    plt.savefig(FigureName, dpi=200)

    #plt.show()
    
    plt.close()
    
    return
    
    ###############################################################################################
    
def plot_xy_locations(x_events,y_events, mags, mu):

    x_events = [x_events[i] - x_events[0] for i in range(len(x_events))]
    y_events = [y_events[i] - y_events[0] for i in range(len(y_events))]

#     plt.plot(x_events[:], y_events[:], '-', color='black', lw = 0.3, zorder=1)

    fig, ax = plt.subplots()

    for i in range(len(mags)):

         if mags[i]>=3.0 and mags[i] < 4.:
             ax.plot(x_events[i], y_events[i], marker = '.', color = 'black', ms=1, zorder=1)
            
         if mags[i]>=4. and mags[i] < 5.:
             ax.plot(x_events[i], y_events[i], marker = 'o', color = 'green', ms=2,  zorder=2)
#         
         if mags[i]>=5. and mags[i] < 6.:
             ax.plot(x_events[i], y_events[i], marker = 'o', color= 'orange',  ms=4, zorder=3)

         if mags[i]>=6. and mags[i] < 7:
             ax.plot(x_events[i], y_events[i], marker = 'o', mfc = 'red', mec='red', ms=7,  zorder = 4)
# 
         if mags[i]>=7.:
             ax.plot(x_events[i], y_events[i], marker = '*', mfc = 'yellow', mec='black', mew = 0.5, ms=14, zorder = 5)
#             
    ax.plot(x_events[0], y_events[0], marker = '+', color = 'cyan', mew = 1.5, ms = 16, zorder = 6)
        
    ax.plot([], [], marker = '.', color = 'black', ms=1, lw = 0, label='${4}>M\geq{3.5}$')
    ax.plot([], [], marker = 'o', color = 'green', ms=2, lw = 0, label='${5}>M\geq{4}$')
#    
    ax.plot([], [], marker = 'o', color='orange', ms=4, lw = 0, label='${6}>M\geq{5}$')
#     
    ax.plot([], [], marker = 'o', mfc = 'red', mec='red', ms=7, lw = 0, label='${7}>M\geq{6}$')
    ax.plot([], [], marker = '*', mfc = 'yellow', mec='black', mew = 0.5, ms=14, lw = 0, label='$M\geq{7}$')
   
    ax.legend(loc = 'upper right', fontsize=9)

    SupTitle_text = 'Locations for ' + str(len(x_events)) +  ' ETAS Earthquakes '
    plt.suptitle(SupTitle_text, fontsize=12)
    
    Title_text = 'For a Model Having $M_{min}=$' + str(mu)
    plt.title(Title_text, fontsize=10)

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    FigureName = './Figures/XY_Positions.png'

    plt.savefig(FigureName, dpi=200)
#     
 #    plt.show()
#     
    plt.close()
# 
    return
    
#     ###############################################################################################

def plot_lat_lng_locations(x_events,y_events, mags, mu):

    fig, ax = plt.subplots()

    for i in range(len(mags)):

         if mags[i]>=3.0 and mags[i] < 4.:
             ax.plot(x_events[i], y_events[i], marker = '.', color = 'black', ms=1, zorder=1)
            
         if mags[i]>=4. and mags[i] < 5.:
             ax.plot(x_events[i], y_events[i], marker = 'o', color = 'green', ms=2,  zorder=2)
#         
         if mags[i]>=5. and mags[i] < 6.:
             ax.plot(x_events[i], y_events[i], marker = 'o', color= 'orange',  ms=4, zorder=3)

         if mags[i]>=6. and mags[i] < 7:
             ax.plot(x_events[i], y_events[i], marker = 'o', mfc = 'red', mec='red', ms=7,  zorder = 4)
# 
         if mags[i]>=7.:
             ax.plot(x_events[i], y_events[i], marker = '*', mfc = 'yellow', mec='black', mew = 0.5, ms=14, zorder = 5)
#             
    ax.plot(x_events[0], y_events[0], marker = '+', color = 'cyan', mew = 1.5, ms = 16, zorder = 6)
        
    ax.plot([], [], marker = '.', color = 'black', ms=1, lw = 0, label='${4}>M\geq{3.5}$')
    ax.plot([], [], marker = 'o', color = 'green', ms=2, lw = 0, label='${5}>M\geq{4}$')
#    
    ax.plot([], [], marker = 'o', color='orange', ms=4, lw = 0, label='${6}>M\geq{5}$')
#     
    ax.plot([], [], marker = 'o', mfc = 'red', mec='red', ms=7, lw = 0, label='${7}>M\geq{6}$')
    ax.plot([], [], marker = '*', mfc = 'yellow', mec='black', mew = 0.5, ms=14, lw = 0, label='$M\geq{7}$')
   
    ax.legend(loc = 'upper right', fontsize=9)

    SupTitle_text = 'Locations for ' + str(len(x_events)) +  ' ETAS Earthquakes '
    plt.suptitle(SupTitle_text, fontsize=12)
    
    Title_text = 'For Fake Los Angeles ETAS Data Having $M_{min}=$' + str(mu)
    plt.title(Title_text, fontsize=10)

    plt.xlabel('Fake Longitude ($^o$)')
    plt.ylabel('Fake Latitude ($^o$)')
    
    FigureName = './Figures/Lat_Lng_Positions.png'

    plt.savefig(FigureName, dpi=200)
#     
 #    plt.show()
#     
    plt.close()
# 
    return
    
#     ###############################################################################################

    
def plot_scale_invariant_aftershock_decay(aftershock_list, number_mainshocks, mag_large, dt_avg, scale_factor):

    bin_length = dt_avg * 40        #   25 is interesting
    bin_length = dt_avg * 20        #   25 is interesting
    bin_length = dt_avg * 10        #   25 is interesting

    fig, ax = plt.subplots()
    
#     bin_length = .0005
    
#     print('bin_length', bin_length)
    
#     time_bins = np.arange(bin_length, 200, bin_length)  #   200 is interestng
    time_bins = np.arange(bin_length, 80, 1)  #   200 is interestng  -- Is 80* time_bins < min time between M6.5 events??
    
#     print('len(time_bins)', len(time_bins))
    
    aftershock_number_bins = np.ones(len(time_bins))

    for i in range(len(aftershock_list)):
    
        aftershock_sum = ETASCalcV5.sum_aftershock_intrvl(aftershock_list[i])
        
        duration = len(aftershock_list[i])
        
        if aftershock_sum > 0.1:
        
            for j in range(1,len(aftershock_list[i])):

                try:
                    bin_index = int( (aftershock_list[i][j])/bin_length ) + 1
                except:
                    bin_index = 1
                
                
                if bin_index < duration:
#                 print(mag_large, bin_index)
                    try:
                        aftershock_number_bins[bin_index] += 1
                    except:
                        pass
    
    print('number_mainshocks', number_mainshocks)    
    
    time_bins = [math.log10(time_bins[k]) for k in range(len(time_bins)) ]
    aftershock_number_bins = [math.log10(aftershock_number_bins[k]) for k in range(len(aftershock_number_bins)) ]
    
#     for i in range(len(time_bins)):
#         print('i, time_bins[i], aftershock_number_bins[i]', i, time_bins[i], aftershock_number_bins[i])
#     
    time_bins_to_plot               =   []
    aftershock_number_bins_to_plot  =   []
    
    plot_counter = 0
    for i in range(len(time_bins)):
        j = len(time_bins) - i -1
        if aftershock_number_bins[j] > 0.0:     #   With this catalog, we need to stop at about 80 - maybe the minimum time
                                                #       between mag_large events
            plot_counter += 1

    time_bins_to_plot               = time_bins[:plot_counter]
    aftershock_number_bins_to_plot  = aftershock_number_bins[:plot_counter]
    
    time_bins_to_plot               = time_bins_to_plot[4:]
    aftershock_number_bins_to_plot  = aftershock_number_bins_to_plot[4:]
    
    ax.plot(time_bins_to_plot, aftershock_number_bins_to_plot, marker = 'o', color = 'blue', markersize = 4, lw = 0)
    
    SupTitle_text = 'Aftershock Decay for ' + str(number_mainshocks) +  ' ETAS Mainshocks with M > ' + str(mag_large)
    plt.suptitle(SupTitle_text, fontsize=12)
    
    Title_text = 'Scale Invariant Clustering with Scale Factor ' + str(scale_factor)
    plt.title(Title_text, fontsize=10)

    plt.xlabel('$Log_{10}$(Time After Mainshock)')
    plt.ylabel('$Log_{10}$(Aftershock Number)')
    
    FigureName = './Figures/scale_invariant_omori_plot.png'

    plt.savefig(FigureName, dpi=200)
    
    #plt.show()
    
    plt.close()
    
    return
    
    ###############################################################################################
    
def plot_single_event_aftershock_decay(aftershock_list, number_mainshocks, mag_large, dt_avg, scale_factor):

    bin_length = dt_avg
    
    time_bins = np.arange(bin_length, 400, bin_length)
    
    aftershock_number_bins = np.ones(len(time_bins))
    
    for i in range(len(aftershock_list)):
    
        aftershock_sum = ETASCalcV5.sum_aftershock_intrvl(aftershock_list[i])
        
        if aftershock_sum > 0.1:
        
            duration = len(aftershock_list[i])
            
            for j in range(1,len(aftershock_list[i])):

                try:
                    bin_index = int( (aftershock_list[i][j])/bin_length ) + 1
                except:
                    bin_index = 1
                
                
                if bin_index < duration:
                    aftershock_number_bins[bin_index] += 1
    
    time_bins = [math.log10(time_bins[k]) for k in range(len(time_bins)) ]
    aftershock_number_bins = [math.log10(aftershock_number_bins[k]) for k in range(len(aftershock_number_bins)) ]
    
    time_bins_to_plot               =   []
    aftershock_number_bins_to_plot  =   []
    
    plot_counter = 0
    for i in range(len(time_bins)):
        j = len(time_bins) - i -1
        if aftershock_number_bins[j] > 0.0:
            plot_counter += 1

    time_bins_to_plot               = time_bins[:plot_counter]
    aftershock_number_bins_to_plot  = aftershock_number_bins[:plot_counter]
    
    time_bins_to_plot               = time_bins_to_plot[4:]
    aftershock_number_bins_to_plot  = aftershock_number_bins_to_plot[4:]

    plt.plot(time_bins_to_plot, aftershock_number_bins_to_plot, marker = 'o', color = 'blue', markersize = 4, lw = 0)
    
    #   Save the file

    SupTitle_text = 'Aftershock Decay for ' + str(number_mainshocks) +  ' ETAS Mainshocks with M > ' + str(mag_large)
    plt.suptitle(SupTitle_text, fontsize=12)
    
    Title_text = 'Single Scale Clustering with Scale Factor ' + str(scale_factor)
    plt.title(Title_text, fontsize=10)

    plt.xlabel('$Log_{10}$(Time After Mainshock)')
    plt.ylabel('$Log_{10}$(Aftershock Number)')
    
    FigureName = './Figures/single_scale_omori_plot.png'
    plt.savefig(FigureName, dpi=200)
    
    #plt.show()
    
    plt.close()
    
    return
    
    ###############################################################################################
    
def map_seismicity(NELat_local, NELng_local, SWLat_local, SWLng_local, plot_start_year, \
        Location, catalog,mag_large, min_mag, mu, forecast_intervals, \
        date_array, time_array, year_array, lng_array, lat_array, mag_array, depth_array):


    #   Note:  This uses the new Cartopy interface
    #
    #   -----------------------------------------
    #
    #   Define plot map
    
    dateline_crossing = False
    
    #   Define coordinates
    left_long   = SWLng_local
    right_long  = NELng_local
    top_lat     = SWLat_local
    bottom_lat  = NELat_local
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
    
    longitude_labels = [left_long, right_long]
    longitude_labels_dateline = [left_long, 180, right_long, 360]   #   If the map crosses the dateline
    
    central_long_value = 0
    if dateline_crossing:
        central_long_value = 180
        
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_long_value))
    ax.set_extent([left_long, right_long, bottom_lat, top_lat])

    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='face',
                                        facecolor='coral')
                                        

                                        
    ocean_10m_3000 = cfeature.NaturalEarthFeature('physical', 'bathymetry_H_3000', '10m',
#                                         edgecolor='black',
                                        facecolor='#0000FF',
                                        alpha = 0.3)
                                        

                                        
    lakes_10m = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
#                                        edgecolor='black',
                                        facecolor='blue',
                                        alpha = 0.75)
                                        
    rivers_and_lakes = cfeature.NaturalEarthFeature('physical', 'rivers_lakes_centerlines', '10m',
#                                        edgecolor='aqua',
                                        facecolor='blue',
                                        alpha = 0.75)

    ax.add_feature(ocean_10m_3000)

    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.95)
    ax.add_feature(cfeature.RIVERS, linewidth= 0.5)
    ax.add_feature(cfeature.STATES, edgecolor='gray',linewidth= 0.5)
#     ax.add_feature(states_provinces, edgecolor='gray')
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    
#     stamen_terrain = cimgt.StamenTerrain()
    stamen_terrain = cimgt.Stamen('terrain-background')
    #   Zoom level should not be set to higher than about 6
    ax.add_image(stamen_terrain, 6)

    if dateline_crossing == False:
        gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=True,
                   linewidth=0.25, color='black', alpha=0.5, linestyle='dotted')
                   
    if dateline_crossing == True:
        gl = ax.gridlines(xlocs=longitude_labels_dateline, draw_labels=True,
                   linewidth=1.0, color='white', alpha=0.5, linestyle='--')

    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = True
    gl.ylines = True

    if catalog == 'LosAngeles':
        gl.xlocator = mticker.FixedLocator([-112,-114,-116, -118, -120, -122])
    
    if catalog == 'Tokyo':
        gl.xlocator = mticker.FixedLocator([132,134,136, 138, 140, 142, 144, 146])

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    gl.xlabel_style = {'size': 12, 'color': 'black'}
    gl.ylabel_style = {'size': 12, 'color': 'black'}
    
    #   -----------------------------------------
    #   Put california faults on the map
    
    input_file_name = './California_Faults.txt'
    input_file  =   open(input_file_name, 'r')
    
    for line in input_file:
        items = line.strip().split()
        number_points = int(len(items)/2)
        
        for i in range(number_points-1):
            x = [float(items[2*i]),float(items[2*i+2])]
            y = [float(items[2*i+1]), float(items[2*i+3])]
            ax.plot(x,y,'-', color='darkgreen',lw=0.55, zorder=2)
    
    input_file.close()
    #    
    #   -----------------------------------------
    #
    #   Plot the data
    
#     mag_array, date_array, time_array, year_array, depth_array, lat_array, lng_array = \
#             SEISRFileMethods.read_regional_catalog(min_mag)
            
    for i in range(len(mag_array)): #   First plot them all as black dots
        if float(mag_array[i]) >= min_mag and float(year_array[i]) >= plot_start_year:
            ax.plot(float(lng_array[i]), float(lat_array[i]), '.k', ms=0.5, zorder=1)
        
        if float(mag_array[i]) >= 6.0 and float(mag_array[i]) < 6.89999  and float(year_array[i]) >= plot_start_year:
 #            ax.plot(float(lng_array[i]), float(lat_array[i]), 'g*', ms=11, zorder=2)
            ax.plot(float(lng_array[i]), float(lat_array[i]), 'o', mec='b', mfc='None', mew=1.25, \
                ms=6, zorder=2)
            
        if float(mag_array[i]) >= 6.89999 and float(year_array[i]) >= plot_start_year:
#             ax.plot(float(lng_array[i]), float(lat_array[i]), 'y*', ms=15, zorder=2)
            ax.plot(float(lng_array[i]), float(lat_array[i]), 'o', mec='r', mfc='None', mew=1.25,\
                ms=12, zorder=2)
# 
    lower_mag = min_mag

#     ax.plot(float(lng_array[i])+ 1000., float(lat_array[i])+1000., 'o', mec='k', mfc='None', mew=0.5, ms = 2, \
#                     zorder=4, label= str(4.9) + '$ > M \geq $'+str(lower_mag))
                    
 #    ax.plot(float(lng_array[i])+ 1000., float(lat_array[i])+1000., '.k', ms=1, \
#                     zorder=4, label='$5.9 > M \geq $'+str(mu))
                    
    ax.plot(float(lng_array[i])+ 1000., float(lat_array[i])+1000., '.k', ms=1, \
                    zorder=4, label='$5.9 > M \geq $'+str(lower_mag) )
                    
    ax.plot(float(lng_array[i])+1000., float(lat_array[i])+1000., 'o', mec='b', mfc='None', mew=1.25, \
                    ms=6, zorder=4, label='\n$6.9 > M \geq 6.0$')
                    
    ax.plot(float(lng_array[i])+1000., float(lat_array[i])+1000., 'o', mec='r', mfc='None', mew=1.25,\
                    ms=12, zorder=4, label='\n$M \geq 6.9$')
                    
    title_text = 'Magnitude Key'
    leg = ax.legend(loc = 'lower left', title=title_text, fontsize=6, fancybox=True, framealpha=0.5)
    leg.set_title(title_text,prop={'size':8})   #   Set the title text font size

    SupTitle_text = 'Seismicity for $M\geq$' + str(min_mag)  + ' after ' + str(plot_start_year)
    plt.suptitle(SupTitle_text, fontsize=14, y=0.98)
#     
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location 
    plt.title(Title_text, fontsize=10)
    
    #   -------------------------------------------------------------

    figure_name = './Figures/Seismicity_Map_' + Location + '_' + str(plot_start_year) + '.png'
    plt.savefig(figure_name,dpi=600)

#     plt.show()
#     
    plt.close()

    #   -------------------------------------------------------------
    
    return None

    ######################################################################
    
