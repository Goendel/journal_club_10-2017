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
import sklearn.preprocessing as sk
from sklearn.metrics import mean_squared_error


def assertCorrectInputSize(input_sequence, target_sequence, input_size):
    if any([input_size!=np.shape(input_sequence)[1], input_size!=np.shape(target_sequence)[1]]):
        raise ValueError("ERROR: INCORECT INPUT SIZE!")

def scaleData(X, scaler_str):
    if scaler_str=="standart":
        # Standardization (zero mean, unit variance data)
        scaler = sk.StandardScaler()
        scaler = scaler.fit(X)
        X_scaled = scaler.transform(X)
        # Checking reconstruction
        X_rec = scaler.inverse_transform(X_scaled)
        print("############### STANDART SCALING ###############")
        print("Mean of the standarized data is (MUST BE 0): {:f}".format(X_scaled.mean()))
        print("Stddev of the standarized data is (MUST BE 1): {:f}".format(X_scaled.var()))
        print("Mean squared error of rescaling (MUST BE 0): {:f}".format(mean_squared_error(X_rec, X)))
        print("##########################################")
        del X_rec
    elif scaler_str=="minmax":
        # MaxAbsScaler to scale the OUTPUT values in range [-1,1]
        scaler = sk.MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(X)
        X_scaled = scaler.transform(X)
        # Checking reconstruction
        X_rec = scaler.inverse_transform(X_scaled)
        print("############### MIN MAX SCALING ###############")
        print("Max of max abs scaled data (MUST BE 1): {:f}".format(X_scaled.max()))
        print("Min of max abs scaled data (MUST BE -1): {:f}".format(X_scaled.min()))
        print("Mean squared error of rescaling (MUST BE 0): {:f}".format(mean_squared_error(X_rec, X)))
        print("##########################################")
        del X_rec
    return X_scaled, scaler


def divideData(data):
    n_samples = np.shape(data)[0]
    n_train = int(n_samples*conf.train_val_ratio)
    data_train = data[n_train: , :]
    data_val = data[:n_train, :]
    return data_train, data_val

def createTrainingDataBatches(input_train, target_train, batch_size):
    n_samples = np.shape(input_train)[0]
    input_train_batches = []
    target_train_batches = []
    n_batches = int(n_samples/batch_size)
    for i in range(n_batches):
        input_train_batches.append(input_train[batch_size*i:batch_size*i+batch_size])
        target_train_batches.append(target_train[batch_size*i:batch_size*i+batch_size])
    return input_train_batches, target_train_batches, n_batches


def createStackedDerivativeData(data_input, data_output, sequence_length, pred_time=1):
    # data_input:[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8] ...
    # data_output:[d3],[d4],[d5],[d6],[d7],[d8] ...
    data_input_stacked, data_output_stacked = [], []
    for i in range(np.shape(data_input)[0]-sequence_length-pred_time+1):
        tempx = data_input[i:(i+sequence_length), :]
        tempy = data_output[(i+sequence_length-1):(i+sequence_length+pred_time-1),:]
        data_input_stacked.append(tempx)
        data_output_stacked.append(tempy)
    return np.array(data_input_stacked), np.array(data_output_stacked)



def scaleDatasets(input_sequence_train, input_sequence_val, target_sequence_train, target_sequence_val, scaler_str):
    input_sequence_train_scaled, scaler_input = scaleData(input_sequence_train, scaler_str)
    target_sequence_train_scaled, scaler_output = scaleData(target_sequence_train, scaler_str)
    input_sequence_val_scaled = scaler_input.transform(input_sequence_val)
    target_sequence_val_scaled = scaler_output.transform(target_sequence_val)
    input_train, target_train = createStackedDerivativeData(input_sequence_train_scaled, target_sequence_train_scaled, conf.sequence_length, conf.pred_time)
    input_val, target_val = createStackedDerivativeData(input_sequence_val_scaled, target_sequence_val_scaled, conf.sequence_length, conf.pred_time)
    return input_train, input_val, target_train, target_val, scaler_input, scaler_output



time_start = time.time()

# fix random seed for reproducibility
np.random.seed(8)

train_data_path = conf.train_data_path
with open(train_data_path, "rb") as file:
    data = pickle.load(file)
    input_sequence = data["input_sequence"]
    target_sequence = data["target_sequence"]
    loss_weights = data["loss_weights"]
    del data

# Checking
assertCorrectInputSize(input_sequence, target_sequence, conf.input_size)

# divide the training samples to train and validation sets
input_sequence_train, input_sequence_val = divideData(input_sequence)
target_sequence_train, target_sequence_val = divideData(target_sequence)


input_train, input_val, target_train, target_val, scaler_input, scaler_output = scaleDatasets(input_sequence_train, input_sequence_val, target_sequence_train, target_sequence_val, conf.scaler)

#########################

if (conf.type == "stateless"):
    shuffle_order = np.arange(np.shape(input_train)[0])
    np.random.shuffle(shuffle_order)
    input_train = input_train[shuffle_order, :, :]
    target_train = target_train[shuffle_order, :, :]

input_train_batches, target_train_batches, n_batches = createTrainingDataBatches(input_train, target_train, conf.batch_size)

if (conf.type == "statefull"):
    # Define the states for saving and updating during training
    C_states = np.zeros((input_train.shape[0], conf.num_units)) # saving the cell states for use in next instances
    H_states = np.zeros((input_train.shape[0], conf.num_units)) # saving the cell states for use in next instances
    C_states_val = np.zeros((input_val.shape[0], conf.num_units)) # saving the cell states for use in next instances
    H_states_val = np.zeros((input_val.shape[0], conf.num_units)) # saving the cell states for use in next instances
    input_val_batches, target_val_batches, n_batches = createTrainingDataBatches(input_val, target_val, conf.batch_size)
    n_train_per_epoch_approx = input_train.shape[0]/4.0
    print("STATEFULL LSTM, approximate number of training steps per epoch {:d}".format(int(n_train_per_epoch_approx)))


# define the model
lstm_model = lstm()

lstm_trainer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(lstm_model.loss, var_list=lstm_model.trainable_variables)

saver = tf.train.Saver(max_to_keep=1000)

start_time = time.time()
if not os.path.exists(conf.central_path + "/Saved_LSTM_Models"):
    os.makedirs(conf.central_path + "/Saved_LSTM_Models")

if not os.path.exists(conf.central_path + "/Training_Results"):
    os.makedirs(conf.central_path + "/Training_Results")




saver = tf.train.Saver()
if(conf.gpu_en==0):
    gpu_options = tf.GPUOptions()
    print("# Running on CPU... ")

if(conf.gpu_en==1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
    print("# Running on GPU with pre process memory fraction: "+str(config.gpu_memory_fraction))


n_samples = np.shape(input_train)[0]
n_data = n_samples*np.shape(input_train)[2]
print("############### TOTAL PARAMETERS ###############")
total_parameters = lstm_model.total_model_parameters
print("# number of parameters: {:}".format(total_parameters))
print("# number of samples per parameter: {:}".format(n_data/total_parameters))

train_loss_vec = []
val_loss_vec = []
min_loss_val = 1e6



print("############### TRAINING ###############")
sys.stdout.write(" ")
sys.stdout.flush()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    
    if conf.loading_model_path == "":
        sess.run(tf.global_variables_initializer()) # initializing all variables
    else:
        saver.restore(sess, conf.loading_model_path)

    counter_trained_before = 0 # for statefull lstm
    n_training_steps_epoch = 0 # for statefull lstm
    k = 1 # for statefull lstm
    
    
    # Losses before starting to optimize
    if (conf.type=="stateless"):
        loss_train_epoch = sess.run(lstm_model.loss, feed_dict={lstm_model.input:input_train, lstm_model.target:target_train, lstm_model.loss_weights:loss_weights})
        loss_val_epoch = sess.run(lstm_model.loss, feed_dict={lstm_model.input:input_val, lstm_model.target:target_val, lstm_model.loss_weights:loss_weights})
    if (conf.type=="statefull"):
        loss_train_epoch = sess.run(lstm_model.loss, feed_dict={lstm_model.input:input_train, lstm_model.target:target_train, lstm_model.loss_weights:loss_weights, lstm_model.c_state:C_states, lstm_model.h_state:H_states})
        loss_val_epoch = sess.run(lstm_model.loss, feed_dict={lstm_model.input:input_val, lstm_model.target:target_val, lstm_model.loss_weights:loss_weights, lstm_model.c_state:C_states_val, lstm_model.h_state:H_states_val})
    train_loss_vec.append(loss_train_epoch)
    val_loss_vec.append(loss_val_epoch)

    output_string = "\r### TRAINING START ###"
    output_string = output_string +"\nTRAIN LOSS = {:.6f}".format(loss_train_epoch)
    output_string = output_string + "\nVALIDATION LOSS = {:.6f}".format(loss_val_epoch)
    sys.stdout.write(output_string)
    sys.stdout.write("\n")
    sys.stdout.flush()

    for epoch in range(conf.max_epoch):
        counter = 0
        
        for counter in range(n_batches):
            input_train_batch, target_train_batch = input_train_batches[counter], target_train_batches[counter]

            if (conf.type=="stateless"):
                _, batch_loss = sess.run([lstm_trainer, lstm_model.loss], feed_dict={lstm_model.input:input_train_batch, lstm_model.target:target_train_batch,lstm_model.loss_weights:loss_weights})
            elif (conf.type=="statefull"):
                
                # Resetting the LSTM states every prop_horizon iterations
                if((counter+1)%(conf.prop_horizon+1)==0):
                    C_states[counter:counter+1,:], H_states[counter:counter+1,:] = np.zeros((1, conf.num_units)), np.zeros((1,  conf.num_units))
                        
                c_state = C_states[counter:counter+1, :]
                h_state = H_states[counter:counter+1, :]
                
                # Updating weights only every k iterations, with k varying, for statistically ind. updates
                if (counter-counter_trained_before==k):
                    counter_trained_before = counter
                    k = np.max([rand.randrange(1,1+np.ceil(input_train.shape[0]/(1.0*n_train_per_epoch_approx))),1])
                    n_training_steps_epoch += 1
                    sess.run(lstm_trainer, feed_dict={lstm_model.input:input_train_batch, lstm_model.target:target_train_batch,lstm_model.loss_weights:loss_weights, lstm_model.c_state:c_state, lstm_model.h_state:h_state})

                # Calculating the losses and the prediction
                batch_loss, new_states = sess.run([lstm_model.loss, lstm_model.states], feed_dict={lstm_model.input:input_train_batch, lstm_model.target:target_train_batch,lstm_model.loss_weights:loss_weights, lstm_model.c_state:c_state, lstm_model.h_state:h_state})

                # Saving the new (true) states (batch size has to be one)
                if conf.stacking_input_data == "S1": # batches in the form [1,2,3,4][5,6,7,8][9,10,11,12]
                    C_states[counter+1:counter+2,:] = new_states[0]
                    H_states[counter+1:counter+2,:] = new_states[1]
                elif conf.stacking_input_data == "S2": # batches in the form [1,2,3,4][2,3,4,5][3,4,5,6][4,5,6,7][5,6,7,8]
                    C_states[counter+conf.sequence_length:counter+conf.sequence_length+1,:] = new_states[0]
                    H_states[counter+conf.sequence_length:counter+conf.sequence_length+1,:] = new_states[1]


            if (counter % 10 ==0):
                sys.stdout.write("\r# E = {:d}, Iteration = {:d}, Loss={:.4f}".format(epoch, counter, batch_loss)+ " " * 10)
                sys.stdout.flush()


        if (conf.type=="stateless"):
            loss_train_epoch = sess.run(lstm_model.loss, feed_dict={lstm_model.input:input_train, lstm_model.target:target_train, lstm_model.loss_weights:loss_weights})
            loss_val_epoch = sess.run(lstm_model.loss, feed_dict={lstm_model.input:input_val, lstm_model.target:target_val, lstm_model.loss_weights:loss_weights})
        if (conf.type=="statefull"):
            loss_train_epoch = sess.run(lstm_model.loss, feed_dict={lstm_model.input:input_train, lstm_model.target:target_train, lstm_model.loss_weights:loss_weights, lstm_model.c_state:C_states, lstm_model.h_state:H_states})
            
            for counter in range(np.shape(input_val_batches)[0]):
                input_val_batch, target_val_batch = input_val_batches[counter], target_val_batches[counter]
                # Resetting the LSTM states every prop_horizon iterations
                if((counter+1)%(conf.prop_horizon+1)==0):
                    C_states_val[counter:counter+1,:], H_states_val[counter:counter+1,:] = np.zeros((1, conf.num_units)), np.zeros((1,  conf.num_units))
                
                c_state_val = C_states_val[counter:counter+1, :]
                h_state_val = H_states_val[counter:counter+1, :]

                batch_loss, new_states_val  = sess.run([lstm_model.loss, lstm_model.states], feed_dict={lstm_model.input:input_val_batch, lstm_model.target:target_val_batch, lstm_model.c_state: c_state_val, lstm_model.h_state: h_state_val, lstm_model.loss_weights:loss_weights})


                # Saving the new (true) states (batch size has to be one)
                if conf.stacking_input_data == "S1": # batches in the form [1,2,3,4][5,6,7,8][9,10,11,12]
                    C_states_val[counter+1:counter+1+1,:] = new_states_val[0]
                    H_states_val[counter+1:counter+1+1,:] = new_states_val[1]
                elif conf.stacking_input_data == "S2": # batches in the form [1,2,3,4][2,3,4,5][3,4,5,6][4,5,6,7][5,6,7,8]
                    C_states_val[counter+conf.sequence_length:counter+conf.sequence_length+1,:] = new_states_val[0]
                    H_states_val[counter+conf.sequence_length:counter+conf.sequence_length+1,:] = new_states_val[1]


            loss_val_epoch  = sess.run(lstm_model.loss, feed_dict={lstm_model.input:input_val, lstm_model.target:target_val, lstm_model.c_state: C_states_val, lstm_model.h_state: H_states_val, lstm_model.loss_weights:loss_weights})


        train_loss_vec.append(loss_train_epoch)
        val_loss_vec.append(loss_val_epoch)
        
        if (conf.type=="statefull"):
            # Resetting validation states for next validation performance
            C_states_val = np.zeros((input_val.shape[0], conf.num_units))
            H_states_val = np.zeros((input_val.shape[0], conf.num_units))
            C_states = np.zeros((input_train.shape[0], conf.num_units))
            H_states = np.zeros((input_train.shape[0], conf.num_units))


        if(min_loss_val>val_loss_vec[epoch]):
            epoch_smallest_val_error = epoch
            min_loss_val = val_loss_vec[epoch]
            saver.save(sess, conf.saving_model_path +".ckpt")

        output_string = "\r### EPOCH = {:d} ###".format(epoch)+ " " * 60
        output_string = output_string +"\nTRAIN LOSS = {:.4f}".format(loss_train_epoch)
        output_string = output_string + "\nVALIDATION LOSS = {:.4f}".format(loss_val_epoch)
        if (conf.type=="statefull"):
            output_string = output_string + "\nNUMBER OF TRAIN STEPS = {:d}".format(n_training_steps_epoch)
        sys.stdout.write(output_string)
        sys.stdout.write("\n")
        sys.stdout.flush()

sys.stdout.write("\n")
sys.stdout.flush()




# Renaming the checkpoint files to include the epoch
sys.stdout.write("############### RENAMING LSTM STATES ###############\n")
sys.stdout.flush()
print("EPOCH WITH SMALLEST VALIDATION ERROR: "+str(epoch_smallest_val_error))
cwd = os.getcwd()
os.rename(conf.saving_model_path +".ckpt.meta", conf.saving_model_path + "_E"+str(epoch_smallest_val_error)+".ckpt.meta")
os.rename(conf.saving_model_path +".ckpt.index", conf.saving_model_path + "_E"+str(epoch_smallest_val_error)+".ckpt.index")
os.rename(conf.saving_model_path +".ckpt.data-00000-of-00001", conf.saving_model_path + "_E"+str(epoch_smallest_val_error)+".ckpt.data-00000-of-00001")


print("CALCULATING TRAINING TIME")
time_end = time.time()
training_time = time_end - time_start
training_hours= int(training_time//3600)
training_minutes = int((training_time-training_hours*3600)//60)
training_seconds = int((training_time-training_hours*3600-training_minutes*60))
print("Training time : {:02.0f}:{:02.0f}:{:02.0f}".format(training_hours, training_minutes, training_seconds))


print("############### SAVING TRAINING DATA ###############")
# Saving training data
data = {
    "n_samples": n_samples,
    "total_parameters": total_parameters,
    "conf": conf,
    "training_time": training_time,
    "training_hours": training_hours,
    "training_minutes": training_minutes,
    "training_seconds": training_seconds,
    "train_loss_vec": train_loss_vec,
    "val_loss_vec": val_loss_vec,
    "epoch_smallest_val_error": epoch_smallest_val_error,
    "loss_weights": loss_weights,
    "scaler_input": scaler_input,
    "scaler_output": scaler_output,
}

train_result_path = conf.train_result_path

with open(train_result_path, "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)






