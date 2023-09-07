 
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


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import math

import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.simplefilter("error")  #   Prints out location of error


    ###############################################################################################
    
   
    #   Define values for if statements
    
train_new_model                             =   False           #   Trains a new model
continue_existing_model_training            =   False            #   Continues the training of an existing model
plot_test_predictions_existing_model        =   True           #   Makes a plot of the test data
predict_beyond_test_data                    =   False            #   Extends the previous plot beyond the test data
create_test_prediction_movie_slides         =   False           #   Makes movie slides of the training process
create_loss_training_movie                  =   False           #   Makes movie slides of the training process
    ###############################################################################################
    
# Instantiate the model, set hyperparameters
window_size =24                           # Number of previous data used to make a prediction.  Needs to be as long or
                                            #   longer than the typical lengths characterizing variations in the data
future_window_size = 12                      # For future predictions
input_dim = 3*window_size-3                 # Number of features in the input
output_dim = 3*future_window_size-3         # Predict a sequence in the future window
output_dim = future_window_size
hidden_dim = 32                             # Hidden dimension of the Transformer model
num_layers = 2                              # Number of layers in the Transformer model
num_heads = 4                               # Number of attention heads in the Transformer model
dropout = 0.2                               # Dropout rate

num_new_train_epochs    =   1               # Number of cycles in the training loop
num_improvement_epochs  =   10              # Number of epochs in an interation
improvement_cycles      =   400             # Number of improvement interations
                                            #   Total training epochs = 
                                            #   num_new_train_epochs + num_improvement_epochs + improvement_cycles
                                            #   
total_epochs =  num_new_train_epochs + num_improvement_epochs * improvement_cycles
                                    
num_predictions     = 150        # Number of future predictions

month_interval          = 0.07692   #   We use a time interval of "months" 
                                        #       where 1 month = 4 weeks = "lunar month", 52 weeks/year
                                        
np.random.seed(42)
                                        
plot_lower_limit = 1970.
plot_upper_limit = 2030

delta_threshold = 1.0
prediction_threshold_lower = 0.90
prediction_threshold_upper = prediction_threshold_lower + delta_threshold

params = [window_size, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout, num_new_train_epochs, \
            num_improvement_epochs, improvement_cycles, total_epochs, num_predictions, month_interval, plot_lower_limit]

    #############################################################################################################################
    #############################################################################################################################
                                                                                                                         #  MODEL
                                                                                                                         
    #   -----------------------------------------------------------------------
    
    #   Define the Transformer MODEL
    
    #   -----------------------------------------------------------------------
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        output = self.transformer(x, x)
        output = output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)
        output = self.fc(output[:, -1, :])  # Predict the last timestep
        return output

#     #######################################################
    
# Function to save the model data
def save_sliding_window_model(model, path):
    model_data = {
        'state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'dropout': dropout
    }
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
        
    return
        
    #######################################################

# Function to load model data
def load_sliding_window_model(path):
    with open(path, 'rb') as f:
        model_data = pickle.load(f)
    
    loaded_model = TransformerModel(
        model_data['input_dim'],
        model_data['hidden_dim'],
        model_data['output_dim'],
        model_data['num_layers'],
        model_data['num_heads'],
        model_data['dropout']
    )
    loaded_model.load_state_dict(model_data['state_dict'])
    
    return loaded_model

    
    #######################################################

def create_sliding_state_vectors(data, time_value, window_size, future_window_size):

    inputs, targets, time_inputs, time_targets = [], [], [], []
    for i in range(len(data) - window_size - future_window_size + 1):
    
        inputs_list         = data[i:i+window_size]
        time_inputs_list    = time_value[i:i+window_size]  
        
        first_diff = list(np.diff(inputs_list))
        for j in range(len(first_diff)):
            inputs_list.append(first_diff[j])
        second_diff = list(np.diff(first_diff))
        for j in range(len(second_diff)):
            inputs_list.append(second_diff[j])
            
        inputs.append(inputs_list)
        time_inputs.append(time_inputs_list)
        
        targets_list        = data[i+window_size:i+window_size+future_window_size]
        time_targets_list   = time_value[i+window_size:i+window_size+future_window_size] 
        
        targets.append(targets_list)
        time_targets.append(time_targets_list)
        
    return np.array(inputs), np.array(targets), time_inputs, time_targets

    #######################################################
    
def create_sliding_state_vectors_prediction(data, time_value, window_size, future_window_size):

    inputs = []
    for i in range(len(data) - window_size):        #   Goes right up to the end of the data
    
        inputs_list         = data[i:i+window_size]
#         
        first_diff = list(np.diff(inputs_list))
        for j in range(len(first_diff)):
            inputs_list.append(first_diff[j])
        second_diff = list(np.diff(first_diff))
        for j in range(len(second_diff)):
            inputs_list.append(second_diff[j])
            
        inputs.append(inputs_list)

    time_enlarged = time_value.copy()
    for j in range(future_window_size):
        time_enlarged.append(time_value[-1:][0] + float(j+1)*month_interval)
        
    time_targets = []
    for i in range(len(time_enlarged) - future_window_size):
        time_targets_list   = time_enlarged[i+window_size:i+window_size+future_window_size] 
        time_targets.append(time_targets_list)
        
    return np.array(inputs), time_targets

    #######################################################
    
def train_model_new_data(input_file_name):

    # Read training data file
    train_data, train_data_scalar, time_value = read_data_file('./ETASCode/Output_Files/train_data_values.txt')
    train_data = rescale_data(train_data)
    train_inputs, train_targets, time_inputs_train, time_targets_train = \
            create_sliding_state_vectors(train_data, time_value, window_size, future_window_size)
    
#     # Convert data to tensors
    train_inputs = torch.from_numpy(train_inputs).float()
    train_targets = torch.from_numpy(train_targets).unsqueeze(1).float()

 
    # Create the sliding window model
    sliding_window_model = TransformerModel(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout)
    #     model = TransformerModel(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(sliding_window_model.parameters(), lr=0.001)
    
    # Training loop with sliding window model
    num_epochs = 1
    for epoch in range(num_epochs):
        sliding_window_model.train()
        optimizer.zero_grad()
    
        # Forward pass
        outputs = sliding_window_model(train_inputs.unsqueeze(1))
    #     
        outputs_future_window = outputs
        if future_window_size > 1:
            outputs_future_window = outputs.squeeze(1)
            
        # Compute loss
        loss = criterion(outputs_future_window, train_targets.squeeze(1))

        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss = float(loss) / float(len(train_data))
        print(f"Epoch {epoch+1}/{total_epochs}, Loss: {epoch_loss}")
        print()

    # Save the sliding window model data
    save_sliding_window_model(sliding_window_model, 'sliding_window_model_with_sequence.pkl')

    return 

    #####################################################
    
def improve_existing_model(input_file_name, epoch_number):

    # Read training data file
    train_data, train_data_scalar, time_value = read_data_file('./ETASCode/Output_Files/train_data_values.txt')
    train_data = rescale_data(train_data)
    train_inputs, train_targets, time_inputs_train, time_targets_train = \
            create_sliding_state_vectors(train_data, time_value, window_size, future_window_size)
    
    # Convert data to tensors
    train_inputs = torch.from_numpy(train_inputs).float()
    train_targets = torch.from_numpy(train_targets).float()

    # Load the existing sliding window model
    loaded_sliding_window_model = load_sliding_window_model('sliding_window_model_with_sequence.pkl')

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(loaded_sliding_window_model.parameters(), lr=0.001)

    # Training loop
    num_epochs = num_improvement_epochs
    for epoch in range(num_epochs):
        loaded_sliding_window_model.train()
        optimizer.zero_grad()
    
        # Forward pass
        outputs = loaded_sliding_window_model(train_inputs.unsqueeze(1))
    #     
    
        outputs_future_window = outputs
        train_targets_future_window = train_targets
        if future_window_size > 1:
            outputs_future_window = outputs.squeeze(1)
            train_targets_future_window = train_targets.squeeze(1)
            
        # Compute loss
        loss = criterion(outputs_future_window, train_targets_future_window)

        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_number += 1
    
        epoch_loss = float(loss) / float(len(train_data))
        print(f"Epoch {epoch_number}/{total_epochs}, Loss: {epoch_loss}")
        print()
        
#     # Save the sliding window model data
    save_sliding_window_model(loaded_sliding_window_model, 'sliding_window_model_with_sequence.pkl')
#     
    return epoch_number

    #######################################################

def test_the_model(model_to_load):

    #   Here we test to see whether, given the previous "window_size" set of data, we can predict the values in the next
    #       "future_window_size" set of data

    # Read testing data file
    test_data, test_data_scalar, time_value = read_data_file('./ETASCode/Output_Files/test_data_values.txt')
    
    
    test_data = rescale_data(test_data)
    test_inputs, test_targets, time_inputs_test, time_targets_test = \
            create_sliding_state_vectors(test_data, time_value, window_size, future_window_size)
#     
    # Convert data to tensors
    test_inputs = torch.from_numpy(test_inputs).float()
    test_targets = torch.from_numpy(test_targets).float()
    
    # Load the existing sliding window model
    loaded_sliding_window_model = load_sliding_window_model(model_to_load)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Perform sliding window prediction for test data
    sliding_window_outputs = []
    with torch.no_grad():
        for i in range(len(test_inputs)):
            sliding_window_input = test_inputs[i:i+window_size].unsqueeze(1)
            sliding_window_output = loaded_sliding_window_model(sliding_window_input)
            sliding_window_outputs.append(sliding_window_output[0].tolist())
    
    outputs_tensor = torch.tensor(sliding_window_outputs)

    # Compute loss
    loss = criterion(outputs_tensor, test_targets)

    # Produce the output list
    predictions_list = []
    for i in range(len(sliding_window_outputs)):
        working_list = []
        for j in range(future_window_size):
            working_list = (sliding_window_outputs[i])
        predictions_list.append(working_list)
#     
    return  predictions_list, time_targets_test, loss


    #######################################################
    
def future_predictions_beyond_test_data(model_to_load):
#     
    # Generate future predictions beyond the test data
    #   Here we test to see whether, given the previous "window_size" set of data, we can predict the values in the next
    #       "future_window_size" set of data

    # Read testing data file
    test_data, test_data_scalar, time_value = read_data_file('./ETASCode/Output_Files/test_data_values.txt')
    
    test_data = rescale_data(test_data)
    prediction_inputs, time_predictions_list = \
            create_sliding_state_vectors_prediction(test_data, time_value, window_size, future_window_size)
#     
    # Convert data to tensors
    prediction_inputs = torch.from_numpy(prediction_inputs).float()
    
    # Load the existing sliding window model
    loaded_sliding_window_model = load_sliding_window_model('sliding_window_model_with_sequence.pkl')
    
    # Perform sliding window prediction for test data
    sliding_window_outputs = []
    with torch.no_grad():
        for i in range(len(prediction_inputs)):
            sliding_window_input = prediction_inputs[i:i+window_size].unsqueeze(1)
            sliding_window_output = loaded_sliding_window_model(sliding_window_input)
            sliding_window_outputs.append(sliding_window_output[0].tolist())
            
    predictions_list = []
    for i in range(len(sliding_window_outputs)):
        working_list = []
        for j in range(future_window_size):
            working_list = (sliding_window_outputs[i])
        predictions_list.append(working_list)
#     
    return  predictions_list, time_predictions_list
# 
    #############################################################################################################################
    #############################################################################################################################
                                                                                                                     # UTILITIES
                                                                                                                     
    #   -----------------------------------------------------------------------
    
    #   Define the Transformer UTILITIES (read files, write files, calculations, etc.)
    
    #   -----------------------------------------------------------------------
#         
def read_data_file(input_file_name):

    minimum_time = plot_lower_limit - float(window_size) * month_interval
    
#   Load the data as a list:  data_value is a list of the data points

    input_file = open(input_file_name, 'r')  #   input file should have 3 entries: events data; data value; and time (or index)

    events_value    =   []
    data_value      =   []
    time_value      =   []     #   We don't really use this for training the data, perhaps only for plotting

    for line in input_file:
        items = line.strip().split()
        
        events_value.append(float(items[0]))
        data_value.append(float(items[1]))
        time_value.append(float(items[2]))
            
    input_file.close()
    
    num_points = len(data_value)
    
    data_value_scalar = data_value  #   Used in the plotting routine
    
    data_value_scalar = rescale_data(data_value_scalar)
    data_value = rescale_data(data_value_scalar)
    
    # Change data to a 1 dim feature vector
    data = np.reshape(data_value,(num_points,))
    
    return data, data_value_scalar, time_value
    
    #######################################################
    
def read_test_catalog():

    mags                    =   []
    date_events             =   []
    time_events             =   []
    year_events             =   []
    depth_events            =   []
    y_events                =   []
    x_events                =   []
    label_events            =   []

    data_file = open('./ETASCode/Output_Files/ETAS_Test_Catalog.txt',"r")
    
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

    #######################################################

def rescale_data(data_values_unscaled):     #   This function rescales the data to values between [0,1]

    #   Find min and max
     
    min_data_value = min(data_values_unscaled)
    max_data_value = max(data_values_unscaled)
    
    delta_data_values = max_data_value - min_data_value
    
    data_values_rescaled = []
    
    #   Rescale to lie in the interval [0,1]
    for i in range(len(data_values_unscaled)):
        rescaled_value = (data_values_unscaled[i]-  min_data_value)/delta_data_values
        data_values_rescaled.append(rescaled_value)

    return data_values_rescaled
    
      
    #############################################################################################################################
    #############################################################################################################################
                                                                                                                          # PLOTS

    #   -----------------------------------------------------------------------

    #   Define the Transformer Model PLOT Methods
    
    #   -----------------------------------------------------------------------

def plot_model_test_predictions(predicted_values, time_targets_test, test_data_scalar, time_value, loss, total_epochs):

    print(len(time_targets_test), len(predicted_values))
#     
    #   ---------------------------------------------------
    
    #   Read existing test catalog
    date_events, time_events, year_events, x_events, y_events, mags, depth_events, label_events = \
            read_test_catalog()
    
    #   ---------------------------------------------------

    data_to_plot, time_to_plot = [], []
    for i in range(len(time_value)):
        if time_value[i] >= plot_lower_limit and time_value[i] < plot_upper_limit:
            time_to_plot.append(time_value[i])
            data_to_plot.append(test_data_scalar[i])
            
    fig, ax = plt.subplots()
    
    ax.plot(time_to_plot, data_to_plot, '-', color='black', lw=0.75, zorder=10, label='Test Data\nLoss = ' + str(round(loss,4)))
    
    for i in range(len(predicted_values)):
        if min(time_targets_test[i]) > plot_lower_limit and max(time_targets_test[i]) < plot_upper_limit:
            predicted_data_working  =   predicted_values[i]
            time_data_working       =   time_targets_test[i] 

            ax.plot(time_data_working, predicted_data_working, \
                '-', color='lightskyblue', lw=0.75, zorder = 5)
            
            ax.plot(time_data_working[0], predicted_data_working[0], \
                marker=  'o', color='green', markersize=2.0, lw=0.0, zorder = 6)
            
            ax.plot(time_data_working[future_window_size-1], predicted_data_working[future_window_size-1], \
                marker=  'o', color='red', markersize=2.0, lw=0.0, zorder = 6)
            
    ax.plot([],[],'-', color='blue', lw=0.75, zorder = 5, label='Prediction Trajectory')
    
    #   ---------------------------------------------------
            
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    min_plot_line = [ymin for i in range(len(time_to_plot))]
    
#     
    ax.fill_between(time_to_plot, min_plot_line, data_to_plot, color='c', alpha=0.1, zorder=0)
            
    for i in range(len(year_events)):

        if float(mags[i]) >= 6.0 and float(mags[i]) < 7.0  and float(year_events[i]) >= plot_lower_limit:
            x_eq = [year_events[i], year_events[i]]
            y_eq = [ymin,ymax]
            
            ax.plot(x_eq, y_eq, linestyle='dotted', color='k', lw=0.75, zorder=1)
            
    ax.plot([],[], linestyle='dotted', color='k', lw=0.7, zorder=0, label = '7.0 $>$ M $\geq$ 6.0')
            
    for i in range(len(year_events)):

        if float(mags[i]) >= 7.0 and float(year_events[i]) >= plot_lower_limit:
            x_eq = [year_events[i], year_events[i]]
            y_eq = [ymin,ymax]
            
            ax.plot(x_eq, y_eq, linestyle='--', color='r', lw=0.75, zorder=1)
            
    ax.plot([],[], linestyle='--', color='r', lw=0.7, zorder=2, label='M $\geq$ 7.0')
    
    max_plot_line = [ymax for i in range(len(time_to_plot))]
    
    ax.fill_between(time_to_plot , data_to_plot, max_plot_line, color='white', alpha=1.0, zorder=3)    
    
    #   ---------------------------------------------------
#         
    leg = ax.legend(loc='lower left', fancybox=True, fontsize = 6, framealpha = 0.95)

    SupTitle_text = 'Time Series Prediction by Deep Learning Transformer with Attention'
    plt.suptitle(SupTitle_text, fontsize=10)
    
    Title_text = 'Predicted vs. Test. Total Training Epochs = ' + str(total_epochs) + \
             '\nPast Window Size = '+ str(window_size) + ' Months.' + ' Future Window Size = '+ str(future_window_size) + ' Months.'
    plt.title(Title_text, fontsize=8)
    
    plt.ylabel('Scaled Nowcast')
    plt.xlabel('Time(Years)')

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.show()

    return

#     #######################################################

def plot_model_beyond_test_data(predicted_values, time_predictions_test, test_data_scalar, time_value, loss, total_epochs):

#     
    #   ---------------------------------------------------
    
    #   Read existing test catalog
    date_events, time_events, year_events, x_events, y_events, mags, depth_events, label_events = \
            read_test_catalog()
    
    #   ---------------------------------------------------
    
#     print('time_value from plot model beyond test',time_value)
#     print()

    data_to_plot, time_to_plot = [], []
    for i in range(len(time_value)):
        if time_value[i] >= plot_lower_limit:
            time_to_plot.append(time_value[i])
            data_to_plot.append(test_data_scalar[i])
            
    fig, ax = plt.subplots()
    
    ax.plot(time_to_plot, data_to_plot, '-', color='black', lw=0.75, zorder=10, label='Test Data\nLoss = ' + str(round(loss,4)))
    
    for i in range(len(predicted_values)):
        if min(time_predictions_test[i]) > plot_lower_limit:
            predicted_data_working  =   predicted_values[i]
            time_data_working       =   time_predictions_test[i] 

            ax.plot(time_data_working, predicted_data_working, \
                '-', color='lightskyblue', lw=0.75, zorder = 5)
            
            ax.plot(time_data_working[0], predicted_data_working[0], \
                marker=  'o', color='green', markersize=2.0, lw=0.0, zorder = 6)
            
            ax.plot(time_data_working[future_window_size-1], predicted_data_working[future_window_size-1], \
                marker=  'o', color='red', markersize=2.0, lw=0.0, zorder = 6)
            
    ax.plot([],[],'-', color='blue', lw=0.75, zorder = 5, label='Prediction Trajectory')
    
    #   ---------------------------------------------------
            
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    min_plot_line = [ymin for i in range(len(time_to_plot))]
    
#     
    ax.fill_between(time_to_plot, min_plot_line, data_to_plot, color='c', alpha=0.1, zorder=0)
            
    for i in range(len(year_events)):

        if float(mags[i]) >= 6.0 and float(mags[i]) < 7.0  and float(year_events[i]) >= plot_lower_limit:
            x_eq = [year_events[i], year_events[i]]
            y_eq = [ymin,ymax]
            
            ax.plot(x_eq, y_eq, linestyle='dotted', color='k', lw=0.75, zorder=1)
            
    ax.plot([],[], linestyle='dotted', color='k', lw=0.7, zorder=0, label = '7.0 $>$ M $\geq$ 6.0')
            
    for i in range(len(year_events)):

        if float(mags[i]) >= 7.0 and float(year_events[i]) >= plot_lower_limit:
            x_eq = [year_events[i], year_events[i]]
            y_eq = [ymin,ymax]
            
            ax.plot(x_eq, y_eq, linestyle='--', color='r', lw=0.75, zorder=1)
            
    ax.plot([],[], linestyle='--', color='r', lw=0.7, zorder=2, label='M $\geq$ 7.0')
    
    max_plot_line = [ymax for i in range(len(time_to_plot))]
    
#     ax.fill_between(time_data , test_data, max_plot_line, color='white', alpha=1.0, zorder=3)    
    ax.fill_between(time_to_plot , data_to_plot, max_plot_line, color='white', alpha=1.0, zorder=3)    
    
    #   ---------------------------------------------------
#         
    leg = ax.legend(loc='lower left', fancybox=True, fontsize = 6, framealpha = 0.95)

    SupTitle_text = 'Time Series Prediction by Deep Learning Transformer with Attention'
    plt.suptitle(SupTitle_text, fontsize=10)
    
    Title_text = 'Predicted vs. Test Beyond Test Data. Total Training Epochs = ' + str(total_epochs) + \
             '\nPast Window Size = '+ str(window_size) + ' Months.' + ' Future Window Size = '+ str(future_window_size) + ' Months.'
    plt.title(Title_text, fontsize=8)
    
    plt.ylabel('Scaled Nowcast')
    plt.xlabel('Time(Years)')

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.show()

    return

#     #######################################################

def plot_prediction_movie_slides(predicted_values, time_targets_test, test_data_scalar, time_value, \
        loss, total_epochs, num_movie_data, slide_index):

    #   Note:  Use Quicktime app to make the movie.  Go to File -> Open Image Sequence, 
    #   then for Choose Media, select the folder
    #   with the movie slides in it.  Then convert the .mov file to .mp4 using   
    #   https://www.freeconvert.com/video-converter
    
    print('Creating slide: ', slide_index, ' of ', num_movie_data, ' total slides')
    print()
    
#     
    #   ---------------------------------------------------
    
    #   Read existing test catalog
    date_events, time_events, year_events, x_events, y_events, mags, depth_events, label_events = \
            read_test_catalog()
    
    #   ---------------------------------------------------

    data_to_plot, time_to_plot = [], []
    for i in range(len(time_value)):
        if time_value[i] >= plot_lower_limit and time_value[i] < plot_upper_limit:
            time_to_plot.append(time_value[i])
            data_to_plot.append(test_data_scalar[i])
            
    fig, ax = plt.subplots()

    ax.set_xlim(plot_lower_limit - (future_window_size + 12)*month_interval, time_value[-1:][0] + (future_window_size+12)*month_interval)
    
    ax.plot(time_to_plot, data_to_plot, '-', color='black', lw=0.75, zorder=10, label='Test Data\nLoss = ' + str(round(loss,4)))
    
    for i in range(len(predicted_values)):
        if min(time_targets_test[i]) > plot_lower_limit and max(time_targets_test[i]) < plot_upper_limit:
            predicted_data_working  =   predicted_values[i]
            time_data_working       =   time_targets_test[i] 

            ax.plot(time_data_working, predicted_data_working, \
                '-', color='lightskyblue', lw=0.75, zorder = 5)
            
            ax.plot(time_data_working[0], predicted_data_working[0], \
                marker=  'o', color='green', markersize=1.5, lw=0.0, zorder = 6)
            
            ax.plot(time_data_working[future_window_size-1], predicted_data_working[future_window_size-1], \
                marker=  'o', color='red', markersize=1.5, lw=0.0, zorder = 6)
            
    ax.plot([],[],'-', color='lightskyblue', lw=0.75, zorder = 5, label='Prediction Trajectory')
    ax.plot([],[], marker=  'o', color='green', markersize=1.5, lw=0.0, zorder = 5, label='Prediction[1]')
    ax.plot([],[], marker=  'o', color='red', markersize=1.5, lw=0.0, zorder = 5, label='Prediction['+ str(future_window_size) + ']')
    
    #   ---------------------------------------------------
            
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    min_plot_line = [ymin for i in range(len(time_to_plot))]
    
#     
    ax.fill_between(time_to_plot, min_plot_line, data_to_plot, color='c', alpha=0.2, zorder=0)
            
    for i in range(len(year_events)):

        if float(mags[i]) >= 6.0 and float(mags[i]) < 7.0  and float(year_events[i]) >= plot_lower_limit:
            x_eq = [year_events[i], year_events[i]]
            y_eq = [ymin,ymax]
            
            ax.plot(x_eq, y_eq, linestyle='dotted', color='k', lw=0.75, zorder=1)
            
    ax.plot([],[], linestyle='dotted', color='k', lw=0.7, zorder=0, label = '7.0 $>$ M $\geq$ 6.0')
            
    for i in range(len(year_events)):

        if float(mags[i]) >= 7.0 and float(year_events[i]) >= plot_lower_limit:
            x_eq = [year_events[i], year_events[i]]
            y_eq = [ymin,ymax]
            
            ax.plot(x_eq, y_eq, linestyle='--', color='r', lw=0.75, zorder=1)
            
    ax.plot([],[], linestyle='--', color='r', lw=0.7, zorder=2, label='M $\geq$ 7.0')
    
    max_plot_line = [ymax for i in range(len(time_to_plot))]
    
    ax.fill_between(time_to_plot , data_to_plot, max_plot_line, color='white', alpha=1.0, zorder=3)    
    
    #   ---------------------------------------------------
#         
    leg = ax.legend(loc='lower left', fancybox=True, fontsize = 6, framealpha = 0.95)

    SupTitle_text = 'Time Series Prediction by Deep Learning Transformer with Attention'
    plt.suptitle(SupTitle_text, fontsize=10)
    
    Title_text = 'Predicted vs. Test \nTotal Training Epochs = ' + str(total_epochs) + \
             '\nPast Window Size = '+ str(window_size) + ' Months.' + ' Future Window Size = '+ str(future_window_size) + ' Months.'
    plt.title(Title_text, fontsize=8)
    
    plt.ylabel('Scaled Nowcast')
    plt.xlabel('Time(Years)')

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    FigureName = './DataTransformer/Time_Series_Prediction_Transformer_000' + str(slide_index) + '.png'
    plt.savefig(FigureName, dpi=200)
    
    plt.close()
    
    return
    
    #######################################################
    
def plot_model_loss_movie(test_data_scalar, time_value, predicted_values_movie, time_values_movie, \
                num_cycles, running_loss, min_loss, min_loss_index, slide_index, caption):

    figure, axis = plt.subplots(1, 2)

    ax1 = axis[0]
    ax2 = axis[1]

    #   ---------------------------------------------------
    
    #   Read existing test catalog
    date_events, time_events, year_events, x_events, y_events, mags, depth_events, label_events = \
            read_test_catalog()
    
    #   ---------------------------------------------------
    
    time_data = []
    test_data = []
    
    for i in range(len(time_value)):
        if time_value[i] > plot_lower_limit and time_value[i] < plot_upper_limit:
            time_data.append(time_value[i])
            test_data.append(test_data_scalar[i])
            
    ax1.plot(time_data, test_data, '-', color='blue', lw=0.5, zorder=4, label='Test Data')
    
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    
    ax1.set_ylim(min(test_data)-0.05,max(test_data)+0.05)
    
    ax1.plot(time_values_movie, predicted_values_movie, \
            marker=  'o', color='green', markersize=1.5, lw=0.0, zorder = 5)
            
    ax1.plot([], [],  marker=  'o', color='green', markersize=1.5, lw=0.0, zorder = 5, 
        label='Predicted[0] Value')
    
    #   ---------------------------------------------------
            
    min_plot_line = [ymin for i in range(len(time_data))]
    
    ax1.fill_between(time_data, min_plot_line, test_data, color='c', alpha=0.1, zorder=10)
            
    for i in range(len(year_events)):

        if float(mags[i]) >= 6.0 and float(mags[i]) < 7.0  and float(year_events[i]) >= plot_lower_limit:
            x_eq = [year_events[i], year_events[i]]
            y_eq = [ymin,ymax]
            
            ax1.plot(x_eq, y_eq, linestyle='dotted', color='k', lw=0.75, zorder=2)
            
    ax1.plot([],[], linestyle='dotted', color='k', lw=0.7, zorder=2, label = '7.0 $>$ M $\geq$ 6.0')
            
    for i in range(len(year_events)):

        if float(mags[i]) >= 7.0 and float(year_events[i]) >= plot_lower_limit:
            x_eq = [year_events[i], year_events[i]]
            y_eq = [ymin,ymax]
            
            ax1.plot(x_eq, y_eq, linestyle='--', color='r', lw=0.75, zorder=2)
            
    ax1.plot([],[], linestyle='--', color='r', lw=0.7, zorder=2, label='M $\geq$ 7.0')
    
    max_plot_line = [ymax for i in range(len(time_data))]
    
    ax1.fill_between(time_data , test_data, max_plot_line, color='white', alpha=1.0, zorder=3)    
    
    Loss = loss.item()
    if caption == 'Optimal Model':
        Loss = min_loss
    
    label_text = 'Loss: ' + str(round(min_loss,6)) 
    ax1.plot([], [],'', color = 'None', label =  label_text)
    
    label_text = 'Past Window (Months): ' + str(window_size)
    ax1.plot([], [],'', color = 'None', label =  label_text)
    
    label_text = 'Future Window (Months): ' + str(future_window_size) 
    ax1.plot([], [],'', color = 'None', label =  label_text)
    

    label_text = 'Training Cycle: ' + str(slide_index) + ' of ' + str(improvement_cycles)
    if caption == 'Optimal Model':
        label_text = 'Training Cycle: ' + str(min_loss_index) + ' of ' + str(improvement_cycles)
        
    ax1.plot([], [],'', color = 'None', label =  label_text)
    
    #   ---------------------------------------------------
#         
    leg = ax1.legend(loc='lower left', fancybox=True, fontsize = 5, framealpha = 0.975, edgecolor = 'black')
    
    Title_text = 'Predicted vs. Test'

    ax1.set_title(Title_text, fontsize=8)
    
    ax1.set_ylabel('Scaled Nowcast', fontsize = 7)
    ax1.set_xlabel('Time(Years)', fontsize = 7)
    
    ax1.tick_params(axis = 'x', labelsize = 7 )
    ax1.tick_params(axis = 'y', labelsize = 7 )
    #   -------------------------------------------------------------------------------------------
    #   -------------------------------------------------------------------------------------------
                                                                                    # SECOND PLOT
    # Second Plot
    
    if caption == 'Optimal Model':
        running_loss = running_loss[:-1]
    
    x = np.arange(len(running_loss))
    
    ax2.plot(x,running_loss, '-', marker = 'o', markersize = 3, color='blue', lw = 0.0)

    ymin = 1.e-3
    ymax = 1.e-1
    
    ax2.set_ylim(ymin,ymax)
    ax2.set_xlim(-1,num_cycles)
    
    ax2.semilogy(ymin, ymax)
    
    ax2.set_title("Running Loss", fontsize=8)
    
    ax2.set_ylabel('Loss (Mean Square Error)', fontsize = 7)
    ax2.set_xlabel('Improvement Cycle Number', fontsize = 7)
    
    ax2.tick_params(axis = 'x', labelsize = 7 )
    ax2.tick_params(axis = 'y', labelsize = 7 )
    
    #   Vertical line
    ax2.plot([min_loss_index,min_loss_index],[1.1*ymin, 0.8*ymax],'--',lw = 0.75, color = 'red', \
                label = 'Minimum Loss:\n'+ str(round(min_loss,6))+ '\n@ Cycle Number:' + str(min_loss_index))
                
    #   Horiz line                                       
    ax2.plot([2,num_cycles-2],[min_loss, min_loss],'--',lw = 0.75, color = 'red')                
        
    leg = ax2.legend(loc='upper right', fancybox=True, fontsize = 8, framealpha = 1.0, edgecolor = 'black')
    
    SupTitle_text = 'Prediction vs. Test by Deep Learning Transformer with Attention\n\n'
    
    if caption == 'Optimal Model':          #   Occurs with when calculations are ended
        SupTitle_text = 'Optimal Model\n\n'
    
    plt.suptitle(SupTitle_text, fontsize=10, y = 0.875)
    
    figure.tight_layout(pad=1.5)

    
    #   ---------------------------------------------------
    #   ---------------------------------------------------
    
    
    FigureName = './DataLoss/Time_Series_Loss_Transformer_000' + str(2*slide_index) + '.png'
    plt.savefig(FigureName, dpi=200)

    FigureName = './DataLoss/Time_Series_Loss_Transformer_000' + str(2*slide_index+1) + '.png'
    plt.savefig(FigureName, dpi=200)

    FigureName = './DataLoss/Time_Series_Loss_Transformer_000' + str(2*slide_index+2) + '.png'
    plt.savefig(FigureName, dpi=200)
    
    if caption == 'Optimal Model':
        FigureName = './DataLoss/Time_Series_Loss_Transformer_000' + str(2*slide_index+3) + '.png'
        plt.savefig(FigureName, dpi=200)
        
    plt.close()
    
    return
    
    #############################################################################################################################
    #############################################################################################################################
                                                                                                                 # RUN THE MODEL
   
    #   -----------------------------------------------------------------------
   
   #    RUN The Transformer Model
   
    #   -----------------------------------------------------------------------
   
if train_new_model:

    os.system('rm transformer_model.pkl')

    model_to_load = './sliding_window_model_with_sequence.pkl'
    running_loss = []
    min_loss = 1.0

    input_file_name = './ETASCode/Output_Files/train_data_values.txt'

    train_model_new_data(input_file_name)
    
    epoch_number = num_new_train_epochs

    for i in range(improvement_cycles):
    
        epoch_number = improve_existing_model(input_file_name, epoch_number)
        
        #   Save the optimal model
        
        predicted_values, time_targets_test, loss = test_the_model(model_to_load)
        
        running_loss.append(loss.item())
        
        print()
        print('Current Test Data Loss: ', loss.item(), ' For Improvement Cycle: ', i)
        print()
        
        
        if loss.item() < min_loss:
            min_cycle_index = i
            min_loss = min(running_loss)
            
    print()
    print('Minimum Loss Model:')
    print('Minimum Loss from Test Data, Min_Cycle_Index:', min_loss, min_cycle_index)

    #######################################################
    
if continue_existing_model_training:

    running_loss = []
    min_loss = 1.0

    input_file_name = './ETASCode/Output_Files/train_data_values.txt'

    epoch_number = num_new_train_epochs
    
    for i in range(improvement_cycles):
    
        epoch_number = improve_existing_model(input_file_name, epoch_number)
        
        #   Test the model
        
        model_to_load = './sliding_window_model_with_sequence.pkl'
        predicted_values, time_targets_test, loss = test_the_model(model_to_load)
        
        running_loss.append(loss.item())
        
        print()
        print('Current Test Data Loss: ', loss.item(), ' For Improvement Cycle: ', i)
        print()
        
        
        if loss.item() < min_loss:
        
            # Load the current model
            loaded_sliding_window_model = load_sliding_window_model('./sliding_window_model_with_sequence.pkl')
            
            #   Save the minimum loss model
            save_sliding_window_model(loaded_sliding_window_model, './sliding_window_model_with_sequence_optimal.pkl')
            min_cycle_index = i
            min_loss = min(running_loss)
            
    print()
    print('Minimum Loss Model:')
    print('Minimum Loss from Test Data, Min_Cycle_Index:', min_loss.item(), min_cycle_index)
    
    
    #######################################################

if plot_test_predictions_existing_model:

    test_data, test_data_scalar, time_value = read_data_file('./ETASCode/Output_Files/test_data_values.txt')
    
    model_to_load = './sliding_window_model_with_sequence.pkl'
    resp = input('Plot the Optimal Model? (Y/N)')
    if resp == 'Y':
        model_to_load = './sliding_window_model_with_sequence_optimal.pkl'
    print()
    
    resp = input('Enter a new plot start date? (Y/N)')
    if resp == 'Y':
        plot_lower_limit = float(input('Enter Value (> 1970)'))
    resp = input('Enter a new plot end date? (Y/N)')
    if resp == 'Y':
        plot_upper_limit = float(input('Enter Value (< 2024)'))
        
    predicted_values, time_targets_test, loss = test_the_model(model_to_load)
    
    plot_model_test_predictions(predicted_values, time_targets_test, test_data_scalar, time_value, loss, total_epochs)

    #######################################################

if predict_beyond_test_data:

    test_data, test_data_scalar, time_value = read_data_file('./ETASCode/Output_Files/test_data_values.txt')
    
    model_to_load = './sliding_window_model_with_sequence.pkl'
    resp = input('Plot the Optimal Model? (Y/N)')
    if resp == 'Y':
        model_to_load = './sliding_window_model_with_sequence_optimal.pkl'
    print()
    
    
    resp = input('Enter a new plot start date? (Y/N)')
    if resp == 'Y':
        plot_lower_limit = float(input('Enter Value (> 1970)'))
    resp = input('Enter a new plot end date? (Y/N)')
    if resp == 'Y':
        plot_upper_limit = float(input('Enter Value (< 2024)'))
    
    predicted_values_test, time_targets_test, loss = test_the_model(model_to_load)  #   Need this for the loss computation
    
    predicted_values, time_predictions_test = future_predictions_beyond_test_data(model_to_load)
    
    plot_model_beyond_test_data(predicted_values, time_predictions_test, test_data_scalar, time_value, loss, total_epochs)
        

    #######################################################
    
if create_test_prediction_movie_slides:

    #   Note:  Use Quicktime app to make the movie.  Go to File -> Open Image Sequence, 
    #   then for Choose Media, select the folder
    #   with the movie slides in it.  Then convert the .mov file to .mp4 using   
    #   https://www.freeconvert.com/video-converter
    
    test_data, test_data_scalar, time_value = read_data_file('./ETASCode/Output_Files/test_data_values.txt')
    
#     print()
#     model_to_load = './sliding_window_model_with_sequence.pkl'
#     resp = input('Plot the Optimal Model? (Y/N)')
#     if resp == 'Y':
    model_to_load = './sliding_window_model_with_sequence_optimal.pkl'
    print()
    
    predicted_values_test, time_targets_test, loss = test_the_model(model_to_load)  #   Need this for the loss computation
    
    predicted_values, time_predictions_test = future_predictions_beyond_test_data(model_to_load)
    
    print(predicted_values)
    
    #   ---------------------------------------------------    
    
    num_movie_data          = len(predicted_values) 
    slide_flag = True
    
    for slide_index in range(num_movie_data):
    
        predicted_values_movie      = []
        time_values_movie           = []
        
        if slide_index <= len(predicted_values):
            for j in range(slide_index+1):
                if min(time_predictions_test[j]) > plot_lower_limit and max(time_predictions_test[j]) < plot_upper_limit:
                    predicted_values_movie.append(predicted_values[j])
                    time_values_movie.append(time_predictions_test[j])
                
        if len(time_values_movie) > 0 and slide_flag:
            slide_index_min = slide_index
            slide_flag = False
            num_movie_data -= slide_index_min

        if len(time_values_movie) > 0 and slide_flag == False:
            slide_index_movie = slide_index - slide_index_min
            plot_prediction_movie_slides(predicted_values_movie, time_values_movie, \
                test_data_scalar, time_value, loss, total_epochs, num_movie_data, slide_index_movie)
                
    #######################################################
    
if create_loss_training_movie:

    #   Note:  Use Quicktime app to make the movie.  Go to File -> Open Image Sequence, 
    #   then for Choose Media, select the folder
    #   with the movie slides in it.  Then convert the .mov file to .mp4 using   
    #   https://www.freeconvert.com/video-converter
    
    os.system('rm ./sliding_window_model_with_sequence.pkl')
    
    test_data, test_data_scalar, time_value = read_data_file('./ETASCode/Output_Files/test_data_values.txt')
    
    test_data_scalar_copy   = test_data_scalar.copy()
    time_value_copy         = time_value.copy()

    input_file_name = './ETASCode/Output_Files/train_data_values.txt'

    train_model_new_data(input_file_name)
    
    model_to_load = './sliding_window_model_with_sequence.pkl'
    predicted_values, time_predictions_test, loss = test_the_model(model_to_load)
    
#     print('predicted_values', predicted_values)
    
    running_loss = [loss.item()]
    cycle_number = 0
    num_cycles = improvement_cycles+1
    caption = ''
    min_loss = 1.0
    min_loss_index = 0
    
    epoch_number = num_new_train_epochs
    
    predicted_values_movie      = []
    time_values_movie           = []
    
    for j in range(len(predicted_values)):
        if min(time_predictions_test[j]) > plot_lower_limit and max(time_predictions_test[j]) < plot_upper_limit:
            predicted_values_movie.append(predicted_values[j])
            time_values_movie.append(time_predictions_test[j])
    
    # create an initial slide here
    plot_model_loss_movie(test_data_scalar_copy, time_value_copy, predicted_values_movie, time_values_movie, \
        num_cycles, running_loss, min_loss, min_loss_index, cycle_number, caption)

    for slide_index in range(1, num_cycles):       # slide_index is the cycle number
    
        epoch_number = improve_existing_model(input_file_name, epoch_number)
        
        predicted_values, time_predictions_test, loss = test_the_model(model_to_load)
        
        running_loss.append(loss.item())
        
        print()
        print('Current Test Data Loss: ', loss.item(), ' For Improvement Cycle: ', slide_index)
        print()
        
        if loss.item() < min_loss:
        
            min_loss = loss.item()
            
            min_loss_index = minpos = running_loss.index(min(running_loss))
        
            # Load the current model
            loaded_sliding_window_model = load_sliding_window_model('./sliding_window_model_with_sequence.pkl')
            
            #   Save the minimum loss model
            save_sliding_window_model(loaded_sliding_window_model, './sliding_window_model_with_sequence_optimal.pkl')
            
            min_slide_index = slide_index
            
            predicted_values_min = predicted_values.copy()
            
            optimal_flag = True     # Turn the optimal flag on
        
        predicted_values_movie      = []
        time_values_movie           = []
        
        for j in range(len(predicted_values)):
            if min(time_predictions_test[j]) > plot_lower_limit and max(time_predictions_test[j]) < plot_upper_limit:
                predicted_values_movie.append(predicted_values[j][0])
                time_values_movie.append(time_predictions_test[j][0])
                

        plot_model_loss_movie(test_data_scalar_copy, time_value, predicted_values_movie, time_values_movie, \
                num_cycles, running_loss, min_loss, min_loss_index, slide_index, caption)
                
        optimal_flag = False     # Turn the optimal flag off
            
    #Add a final slide to the stack for the optimal model
    
    caption = 'Optimal Model'
    
    predicted_values_movie      = []
    time_values_movie           = []
    
    for j in range(len(predicted_values)):
        if min(time_predictions_test[j]) > plot_lower_limit and max(time_predictions_test[j]) < plot_upper_limit:
            predicted_values_movie.append(predicted_values_min[j][0])
            time_values_movie.append(time_predictions_test[j][0])
            
    
    plot_model_loss_movie(test_data_scalar_copy, time_value, predicted_values_movie, time_values_movie, \
        num_cycles, running_loss, min_loss, min_loss_index, slide_index, caption)
        
    #     #######################################################

