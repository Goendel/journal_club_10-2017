import tensorflow as tf
import numpy as np
from config import Config as conf
import sys
sys.path.insert(0, conf.lstm_module_path)
from lstm_module import *
import time
import sys
import os
import pickle
import random as rand
from scipy.integrate import ode
from functions import *

def prediction(conf):
    
    start_time = time.time()
    if not os.path.exists(conf.central_path + "/Prediction_Results"):
        os.makedirs(conf.central_path + "/Prediction_Results")

    if not os.path.exists(conf.central_path + "/Prediction_Figures"):
        os.makedirs(conf.central_path + "/Prediction_Figures")


    # fix random seed for reproducibility
    np.random.seed(8)

    print("############### LOADING TRAINING FILE ###############")

    train_result_path = conf.train_result_path

    with open(train_result_path, "rb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        data = pickle.load(file)
        n_samples = data["n_samples"]
        total_parameters = data["total_parameters"]
        conf = data["conf"]
        train_loss_vec = data["train_loss_vec"]
        val_loss_vec = data["val_loss_vec"]
        epoch_smallest_val_error = data["epoch_smallest_val_error"]
        loss_weights = data["loss_weights"]
        scaler_input = data["scaler_input"]
        scaler_output = data["scaler_output"]
        del data

    print("TRAINING FILE: "+train_result_path+" LOADED")




    # define the model
    lstm_model = lstm()
    saver = tf.train.Saver()




    # Loading initial conditions
    with open(conf.train_data_path, "rb") as file:
        data = pickle.load(file)
        input_initial_conditions = data["input_initial_conditions"]
        del data


    # shape of initial conditions
    n_ic = np.shape(input_initial_conditions)[0]
    ic_idx = np.random.permutation(n_ic)[:conf.number_init_conditions]
    input_initial_conditions = input_initial_conditions[ic_idx,:]
    n_ic = conf.number_init_conditions

    print("Number of initial conditions: " + str(conf.number_init_conditions))

    dt = conf.dt
    T = conf.T

    number_hotstarting_steps = conf.prop_horizon//2

    number_steps = np.int(T/dt)
    number_total_steps = number_steps + number_hotstarting_steps

    print("Perturbation: "+str(conf.pert_A))
    print("Ensemble size: "+str(conf.N1))
    print("############ PREDICTION LOOP ############")

    predicted_evolution_en = np.zeros((number_steps+1, conf.N1, conf.input_size, n_ic))
    true_evolution_en = np.zeros((number_steps+1, conf.N1, conf.input_size, n_ic))

    predicted_evolution_en_mean = np.zeros((number_steps+1, conf.input_size, n_ic))
    true_evolution = np.zeros((number_steps+1, conf.input_size, n_ic))

    input_sequence = np.zeros((conf.sequence_length, conf.input_size))

    integrator = ode(lorenz)


    saver = tf.train.Saver()
    if(conf.gpu_en==0):
        gpu_options = tf.GPUOptions()
        print("# Running on CPU... ")

    if(conf.gpu_en==1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
        print("# Running on GPU with pre process memory fraction: "+str(config.gpu_memory_fraction))


    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
        path_str = conf.prediction_loading_saved_model_path+"_E"+str(epoch_smallest_val_error)+".ckpt"
        saver.restore(sess, path_str)
        print("MODEL FILE: " + path_str)
        print("MODEL RESTORED.")

        for ic in range(n_ic):
            
            input0 = np.array(input_initial_conditions[ic,:]) # current t
            t0 = 0
            integrator.set_initial_value(input0, t0).set_f_params(conf.sigma, conf.rho, conf.beta)
            integrator.integrate(integrator.t+dt)
            u0 = integrator.y
            input_sequence[0,:] = u0[0]
            
            for t in range(conf.sequence_length-1):
                integrator.integrate(integrator.t+dt)
                input_sequence[t+1,:] = integrator.y[0]

            # Scaling
            input_sequence_scaled = scaler_input.transform(input_sequence)

            # initialize with input_t_scaled the first conf.sequence_length states
            input_test_sequence = np.reshape(input_sequence_scaled, (1, input_sequence_scaled.shape[0], input_sequence_scaled.shape[1]))
            
            # Initialize LSTM states
            c_state_t = np.zeros((1,conf.num_units))
            h_state_t = np.zeros((1,conf.num_units))


            # ITERATIVE PREDICTION
            input_t = input_sequence[-1,:]
            input_t = np.reshape(input_t, (1, input_t.shape[0]))

            # initalize the prediction matrices
            predicted_evolution_en_ic = np.zeros((number_steps+1,conf.N1,conf.input_size))

            true_evolution_en_ic = np.zeros((number_steps+1,conf.N1,conf.input_size))
            true_evolution_ic = np.zeros((number_steps+1,conf.input_size))

            for t in range(number_total_steps+1):
                if t<number_hotstarting_steps:
                    # Calculating true ensemble using L96 equations
                    print("\r## Initial Condition: "+str(ic)+" ## Time instant: "+str(t)+10*" ", end="")
                    
                    integrator.integrate(integrator.t+dt)
                    input_t = integrator.y[0]
                    
                    if conf.type == "statefull":
                        _, new_states = sess.run([lstm_model.prediction, lstm_model.states], feed_dict={lstm_model.input:input_test_sequence, lstm_model.c_state:c_state_t, lstm_model.h_state:h_state_t, lstm_model.loss_weights:loss_weights})
                    if conf.type == "stateless":
                        _, new_states = sess.run([lstm_model.prediction, lstm_model.states], feed_dict={lstm_model.input:input_test_sequence, lstm_model.loss_weights:loss_weights})



                    if conf.type == "statefull":
                        # updating the LSTM state
                        c_state_t = new_states[0] # np.zeros((1,conf.num_units)) # new_states[0]
                        h_state_t = new_states[1] # np.zeros((1,conf.num_units)) # new_states[1]

                    predicted_evolution_en_meanscaled = scaler_input.transform(input_t)
                    predicted_evolution_en_meanscaled = np.reshape(predicted_evolution_en_meanscaled, (1, predicted_evolution_en_meanscaled.shape[0], predicted_evolution_en_meanscaled.shape[1]))
                    
                    input_test_sequence = np.concatenate((input_test_sequence, predicted_evolution_en_meanscaled),axis=1) # now we have conf.sequence_length+conf.pred_time samples
                    
                    input_test_sequence = input_test_sequence[:,conf.pred_time:conf.sequence_length+conf.pred_time,:]
            
                elif t==number_hotstarting_steps:
                    
                    input_t_en = np.matlib.repmat(input_t, conf.N1, 1) + conf.pert_A*np.random.randn(conf.N1,conf.input_size)

                    true_evolution_en_ic[t-number_hotstarting_steps,:,:] = input_t_en.copy()
                    predicted_evolution_en_ic[t-number_hotstarting_steps,:,:] = input_t_en.copy()
                    
                    input_test_sequence_en = np.tile(np.reshape(input_test_sequence, (input_test_sequence.shape[0],input_test_sequence.shape[1],input_test_sequence.shape[2],1)), (1,1,1,conf.N1))
                    
                    # Recording true value
                    true_evolution_ic[t-number_hotstarting_steps,:] = input_t
                    
                    for n in range(conf.N1):
                        input_t_en_n = input_t_en[n,:]
                        input_t_en_n = np.reshape(input_t_en_n,(1,conf.input_size))
                        predicted_evolution_en_n = input_t_en_n
                        predicted_evolution_en_meanscaled_en_n = scaler_input.transform(predicted_evolution_en_n)
                        predicted_evolution_en_meanscaled_en_n = np.reshape(predicted_evolution_en_meanscaled_en_n, (1, predicted_evolution_en_meanscaled_en_n.shape[0], predicted_evolution_en_meanscaled_en_n.shape[1]))
                        input_test_sequence_en[0,-1,:,n] = predicted_evolution_en_meanscaled_en_n
                    
                    if conf.type == "statefull":
                        c_state_t_en = np.tile(np.reshape(c_state_t, (c_state_t.shape[0],c_state_t.shape[1],1)), (1,1,conf.N1))
                        h_state_t_en = np.tile(np.reshape(h_state_t, (h_state_t.shape[0],h_state_t.shape[1],1)), (1,1,conf.N1))


                elif t>number_hotstarting_steps:
                    print("\r## Initial Condition: "+str(ic)+" ## Time instant: "+str(t)+10*" ", end="")

                    integrator.integrate(integrator.t+dt)
                    input_t = integrator.y[0]

                    # Recording true value
                    true_evolution_ic[t-number_hotstarting_steps,:] = input_t
            
                    for n in range(conf.N1):


                        if conf.type == "statefull":
                            c_state_t = c_state_t_en[:,:,n].copy()
                            h_state_t = h_state_t_en[:,:,n].copy()
                            predicted_derivative_t, new_states = sess.run([lstm_model.prediction, lstm_model.states], feed_dict={lstm_model.input:input_test_sequence_en[:,:,:,n], lstm_model.c_state:c_state_t, lstm_model.h_state:h_state_t, lstm_model.loss_weights:loss_weights})
                        if conf.type == "stateless":
                            predicted_derivative_t, new_states = sess.run([lstm_model.prediction, lstm_model.states], feed_dict={lstm_model.input:input_test_sequence_en[:,:,:,n], lstm_model.loss_weights:loss_weights})

                        if conf.type == "statefull":
                            # updating the LSTM state
                            if((t+1)%(conf.prop_horizonumber_steps+1)==0):
                                    c_state_t_en[:,:,n] = np.zeros((1,conf.num_units)) # np.zeros((1,conf.num_units)) # new_states[0]
                                    h_state_t_en[:,:,n] = np.zeros((1,conf.num_units)) # np.zeros((1,conf.num_units)) # new_states[1]

                            else:
                                    c_state_t_en[:,:,n] = new_states[0] # np.zeros((1,conf.num_units)) # new_states[0]
                                    h_state_t_en[:,:,n] = new_states[1] # np.zeros((1,conf.num_units)) # new_states[1]

                        predicted_derivative_t_reshaped = predicted_derivative_t[0,:,:]
                        Dinput_t = scaler_output.inverse_transform(predicted_derivative_t_reshaped)
                        
                        input_t = np.reshape(predicted_evolution_en_ic[t-number_hotstarting_steps-1,n,:].copy(), (1,conf.input_size))

                        input_t += Dinput_t*dt

                        predicted_evolution_en_ic[t-number_hotstarting_steps,n,:] = input_t.copy()

                        predicted_evolution_en_meanscaled = scaler_input.transform(input_t)
                        predicted_evolution_en_meanscaled = np.reshape(predicted_evolution_en_meanscaled, (1, predicted_evolution_en_meanscaled.shape[0], predicted_evolution_en_meanscaled.shape[1]))

                        input_test_sequence_temp = np.concatenate((input_test_sequence_en[:,:,:,n], predicted_evolution_en_meanscaled),axis=1) # now we have conf.sequence_length+conf.pred_time samples
                        input_test_sequence_en[:,:,:,n] = input_test_sequence_temp[:,conf.pred_time:conf.sequence_length+conf.pred_time,:]

            # record LSTM prediction results

            predicted_evolution_en[:,:,:,ic] = predicted_evolution_en_ic
        #    true_evolution_en[:,:,:,ic] = true_evolution_ic_en

            # record LSTM prediction results
            predicted_evolution_en_mean[:,:,ic] = np.mean(predicted_evolution_en_ic, axis=1)
            true_evolution[:,:,ic] = true_evolution_ic

    print("\n")

    # Calculate RMSE

    err = true_evolution - predicted_evolution_en_mean
    err_temp = np.multiply(err, err)
    RMSE_input_t_unscaled = np.sqrt(np.mean(err_temp, 2))


    print("CALCULATING TESTING TIME")
    time_end = time.time()
    testing_time = time_end - start_time
    testing_hours= int(testing_time//3600)
    testing_minutes = int((testing_time-testing_hours*3600)//60)
    testing_seconds = int((testing_time-testing_hours*3600-testing_minutes*60))
    print("testing time : {:02.0f}:{:02.0f}:{:02.0f}".format(testing_hours, testing_minutes, testing_seconds))


    # Saving prediction data

    data = {
        "testing_time": testing_time,
        "testing_hours": testing_hours,
        "testing_minutes": testing_minutes,
        "testing_seconds": testing_seconds,
        "conf": conf,
        "predicted_evolution_en": predicted_evolution_en,
        "true_evolution_en": true_evolution_en,
        "predicted_evolution_en_mean": predicted_evolution_en_mean,
        "true_evolution": true_evolution,
        "RMSE_input_t_unscaled": RMSE_input_t_unscaled,
        "n_ic": n_ic,
        "number_steps": number_steps,
        "dt": dt,
        "T": T,
        "epoch_smallest_val_error": epoch_smallest_val_error,
        "scaler_input": scaler_input,
        "scaler_output": scaler_output,
    }

    print("SAVING PREDICTION DATA")

    with open(conf.prediction_results_path, "wb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    print("Prediction data saved in " + conf.prediction_results_path)

    return predicted_evolution_en_mean

def main():
    return prediction(conf)




