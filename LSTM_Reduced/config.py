import os
import numpy as np

class Config:
    sequence_length = 5
    input_size = 1
    num_units = 10
    learning_rate = 0.01

    activation = "tanh"
    init = "xavier"
    output_activation = "identity" # or "tanh" or "sigmoid" when bounded input or "identity"
    type = "stateless" # "statefull" or "stateless"
    dtype = np.float32
    
    if (output_activation == "identity"):
        scaler = "standart" # standart (with "identity") or max with "tanh" or "sigmoid"
    else:
        scaler = "minmax" # standart (with "identity") or max with "tanh" or "sigmoid"

    if (type == "stateless"):
        prop_horizon = 0
        batch_size = 200
        pred_time = 1 # MUST
    if (type == "statefull"):
        prop_horizon = 100
        batch_size = 1 # MUST
        pred_time = 1 # MUST

    lambda_loss_amount = 0

    EULER = False


    max_epoch = 100
    
    gpu_en = 0
    gpu_memory_fraction = 0.1
    
    train_val_ratio = 0.5

    stacking_input_data = "S2" # S1 if [1,2,3],[4,5,6] S2 if [1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8] etc.


# PREDICTION:
    number_init_conditions = 1
    dt = 0.01 # for computation of next state (iterative prediction)
    T = 1
    pert_A = 0
    N1 = 1


    sigma = 10
    rho = 28
    beta = 8./3
  
    
    cwd = os.getcwd()

    characteristic_string = type+"_SL_"+str(sequence_length)+"_NU_"+str(num_units)+"_PH_"+str(prop_horizon)+"_LR_"+str(int(learning_rate*1e6))+"_"+activation+"_"+init+"_"+output_activation+"_"+scaler

    central_path = "."

    saving_model_path = central_path + "/Saved_LSTM_Models/lstm_model_"+characteristic_string

    train_result_path = central_path + "/Training_Results/train_data_lstm_"+characteristic_string+".pickle"
    loading_model_path = "" # for retraining
    train_data_path = "./Training_Data/lorenz_training_data_reduced.pickle"
    training_figure_path = central_path + "/Training_Figures/train_figure_lstm_"+characteristic_string
    training_figure_name = "Training and validation error"

    prediction_loading_saved_model_path = central_path + "/Saved_LSTM_Models/lstm_model_"+characteristic_string

    prediction_results_path = central_path + "/Prediction_Results/prediction_results_lstm_"+characteristic_string+".pickle"



    if EULER == False:
        lstm_module_path = "/Users/pantelisvlachas/Documents/PhD/PROJECT_LSTM/lstm_module"
    else:
        lstm_module_path = "/cluster/home/pvlachas/project_lstm/lstm_module"







