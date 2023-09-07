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
    
import numpy as np
import os
import sys

import datetime
from datetime import datetime

import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.simplefilter("error")  #   Prints out location of error

    ###############################################################################################
    
def year_fraction(date):

    print()
    print('called decimal year function')

    catalog_start_year = datetime(date.year, 1, 1)
    catalog_end_year = datetime(date.year + 1, 1, 1)
    year_duration = (catalog_end_year - catalog_start_year).total_seconds()
    current_duration = (date - catalog_start_year).total_seconds()
    decimal_years = date.year + current_duration / year_duration
    
    return decimal_years
    
    ###############################################################################################
    
with open(r"./ETASCode/USGS_regional.catalog", 'r') as fp:
    num_USGS_data = sum(1 for line in fp)

print()

# Get today's date
today = datetime.today()
todays_decimal_year = year_fraction(today)
catalog_end_year = str(todays_decimal_year)

    ###############################################################################################

textstring = 'There are ' + str(num_USGS_data) + ' events in the USGS data set for Los Angeles region having M>3.29 since 1960.  \n'\
    + 'The number of test data will be equal to this.  You should enter a value for number of training data \n'\
    + 'at least a factor of 5-10 larger than this.'
    
print(textstring)
print()

    ###############################################################################################

#   Ask for number of training and test data points

print()
print('Choose: Create training data only (choice 1); Create test data only (choice 2); Or create both (choice 3)')

resp = input()

if resp == '1':
    
    num_points = input('Enter Number of Training Data Points (> ' + str(num_USGS_data) + ') : ')
    num_points = int(num_points)
    
    delta_years = todays_decimal_year - 1960.
    catalog_start_year = str(1960. - (float(num_points-num_USGS_data)/float(num_USGS_data)) * delta_years)
    
    print('catalog_start_year', catalog_start_year)

    # Generate time series data for ETAS model and store it in the training_event_file.txt:
    os.system('python ./ETASCode/GenerateETASData.py ' + str(num_points) + ' ' + catalog_start_year + ' ' + catalog_end_year + ' Train')
    
    os.system('mv ./ETASCode/Output_Files/test_train_data.txt ./ETASCode/Output_Files/train_data_values.txt')
    
    os.system('mv ./ETASCode/Output_Files/ETAS_Scale_Invariant_Output.txt ./ETASCode/Output_Files/ETAS_Train_Catalog.txt')
    
elif resp == '2':

    catalog_start_year = str(1960.0)

    # Generate time series data for ETAS model and store it in the test_event_file.txt:
    os.system('python ./ETASCode/GenerateETASData.py ' + str(num_USGS_data) + ' ' + catalog_start_year + ' ' + catalog_end_year + ' Test')
    
    os.system('mv ./ETASCode/Output_Files/test_train_data.txt ./ETASCode/Output_Files/test_data_values.txt')
    
    os.system('mv ./ETASCode/Output_Files/ETAS_Scale_Invariant_Output.txt ./ETASCode/Output_Files/ETAS_Test_Catalog.txt')
    
    #   Now call SEISR codes to write the monthly data using same method with argv

else:   #   Choice 3

    num_points = input('Enter Number of Training Data Points (> ' + str(num_USGS_data) + ') : ')
    num_points = int(num_points)
    
    delta_years = todays_decimal_year - 1960.
    catalog_start_year = str(1960. - (float(num_points-num_USGS_data)/float(num_USGS_data)) * delta_years)
    
    # Generate time series data for ETAS model and store it in the training_event_file.txt:
    os.system('python ./ETASCode/GenerateETASData.py ' + str(num_points) + ' ' + catalog_start_year + ' ' + catalog_end_year + ' Train')
    
    os.system('mv ./ETASCode/Output_Files/test_train_data.txt ./ETASCode/Output_Files/train_data_values.txt')
    
    os.system('mv ./ETASCode/Output_Files/ETAS_Scale_Invariant_Output.txt ./ETASCode/Output_Files/ETAS_Train_Catalog.txt')
    
    #   -----------------------------------------
    
    catalog_start_year = str(1960.0)
    
    # Generate time series data for ETAS model and store it in the test_event_file.txt:
    os.system('python ./ETASCode/GenerateETASData.py ' + str(num_USGS_data) + ' ' + catalog_start_year + ' ' + catalog_end_year + ' Test')
    
    os.system('mv ./ETASCode/Output_Files/test_train_data.txt ./ETASCode/Output_Files/test_data_values.txt')
    
    os.system('mv ./ETASCode/Output_Files/ETAS_Scale_Invariant_Output.txt ./ETASCode/Output_Files/ETAS_Test_Catalog.txt')
    
    #   -----------------------------------------
    #   -----------------------------------------
    
    