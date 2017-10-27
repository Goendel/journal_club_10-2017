import tensorflow as tf
from config import Config as conf
import numpy as np

tf_activations = {"relu": tf.nn.relu, "tanh": tf.tanh, "sigmoid": tf.sigmoid, "identity": tf.identity}
tf_initializers = {"xavier": tf.contrib.layers.xavier_initializer(), "normal": tf.truncated_normal_initializer()}

# Usefull functions


def computeNumberOfModelParameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters

def constructStatefullLSTMCell(num_units, activation, input, initial_state, init):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0, state_is_tuple = True, activation= tf_activations[activation]) # tf.nn.rnn_cell.BasicLSTMCell
    if(init=="xavier"):
        with tf.variable_scope("lstm", initializer=tf.contrib.layers.xavier_initializer()):
            outputs, states = tf.nn.dynamic_rnn(cell, input, dtype=tf.float64, initial_state=initial_state)
    else:
        with tf.variable_scope("lstm", initializer=tf.contrib.layers.xavier_initializer()):
            outputs, states = tf.nn.dynamic_rnn(cell, input, dtype=tf.float64, initial_state=initial_state)
    return cell, outputs, states

def computeStatelessLSTMCell(num_units, activation, input, init):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0, state_is_tuple = True, activation= tf_activations[activation]) # tf.nn.rnn_cell.BasicLSTMCell
    if(init=="xavier"):
        with tf.variable_scope("lstm", initializer=tf.contrib.layers.xavier_initializer()):
            outputs, states = tf.nn.dynamic_rnn(cell, input, dtype=tf.float64)
    else:
        with tf.variable_scope("lstm", initializer=tf.contrib.layers.xavier_initializer()):
            outputs, states = tf.nn.dynamic_rnn(cell, input, dtype=tf.float64)
    return cell, outputs, states

def constructOutputLayer(outputs, num_units, input_size, init, pred_time, output_activation):
    # Taking the last output of the lstm network
    last = outputs[:,-1,:]
    with tf.variable_scope("lstm"):
        weights = tf.get_variable("weights", shape=[num_units, pred_time, input_size], initializer=tf_initializers[init], dtype=tf.float64)
        biases = tf.get_variable("biases", shape=[pred_time, input_size], initializer=tf.constant_initializer(0), dtype=tf.float64)
    weights_reshaped = tf.reshape(weights, [num_units, pred_time*input_size])
    output_tile_unbiased = tf.matmul(last, weights_reshaped)
    output_unbiased = tf.reshape(output_tile_unbiased, [tf.shape(output_tile_unbiased)[0], pred_time, input_size])
    prediction = output_unbiased + biases
    prediction = tf_activations[output_activation](prediction)
    return prediction

class lstm(object):
    def __init__(self):
        # Set parameters from config file
        if conf.dtype == np.float64:
            self.dtype = tf.float64
        else:
            self.dtype = tf.float32
        
        self.sequence_length = conf.sequence_length
        self.input_size = conf.input_size
        self.num_units = conf.num_units
        self.activation = conf.activation
        self.init = conf.init
        self.output_activation = conf.output_activation
        self.pred_time = conf.pred_time
        self.type = conf.type
        self.lambda_loss_amount = conf.lambda_loss_amount
        # Construct lstm
        self.input = tf.placeholder(tf.float64, shape=[None, self.sequence_length, self.input_size]) # placeholder for training input
        if self.type=="statefull":
            self.c_state = tf.placeholder(tf.float64, [None, self.num_units])
            self.h_state = tf.placeholder(tf.float64, [None, self.num_units])
            self.initial_state = tf.contrib.rnn.LSTMStateTuple(self.c_state, self.h_state) # Tensorflow 1.0.1
            self.cell, self.outputs, self.states = constructStatefullLSTMCell(self.num_units, self.activation, self.input, self.initial_state, self.init)
        if self.type=="stateless":
            self.cell, self.outputs, self.states = computeStatelessLSTMCell(self.num_units, self.activation, self.input, self.init)
        self.prediction = constructOutputLayer(self.outputs, self.num_units, self.input_size, self.init, self.pred_time, self.output_activation)
        self.prediction = self.prediction
        
        # Training
        self.target = tf.placeholder(tf.float64, shape=[None, self.pred_time, self.input_size]) # placeholder for training target (output)
        self.unweighted_loss = tf.reduce_mean(tf.squared_difference(self.prediction, self.target), 0)
        self.loss_weights = tf.placeholder(tf.float64, shape=[self.pred_time, self.input_size]) # placeholder for weighting the loss
        
        # Different losses
        self.loss = tf.reduce_sum(tf.divide(tf.reduce_sum(tf.multiply(self.unweighted_loss, self.loss_weights), 1), tf.reduce_sum(self.loss_weights, 1)))/self.pred_time
        self.l2 = self.lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        self.l2_loss = self.loss + self.l2
        
        self.trainable_variables = tf.trainable_variables()
        self.trainable_variables_names = [var.name for var in tf.trainable_variables()]
        self.total_model_parameters = computeNumberOfModelParameters()








