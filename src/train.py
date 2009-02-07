#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

PyNeural - A simple feed-forward, artificial neural network library for python

Copyright (C) 2008 Justin Riley (justin.t.riley@gmail.com)

This library is free software; you can redistribute it and/or modify it under the terms of the 
GNU Lesser General Public License as published by the Free Software Foundation; either version 2.1 
of the License, or (at your option) any later version. This library is distributed in the hope that 
it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.  You 
should have received a copy of the GNU Lesser General Public License along with this library; if not, 
write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

"""

import os, sys, time
from numpy import loadtxt, savetxt, power
from numpy.random import rand
from numpy import matrix as Matrix
from pyneural.vectorneuron import VectorNeuron

class Train:
    """
    The Train class is responsible for loading inputs/targets,
    and creating/training the Neural Network.
    """

    __inputs = None
    __targets = None
    __single_input = None
    __single_target = None
    __next_input = None
    __error = None
    __vector_neurons = []
    __org_error = None
    __new_error = None
    __best_error = None
    __sum_of_x_squared = 0.0
    __sample_mean = None
    __sample_variance = None
    __change_helped = False
    __time_stamp = ""
    __neural_network = None

    def __init__(self, networkparameters):
        """Initializes train instance with NetworkParameters"""
        self.__neural_network = networkparameters

    def __set_train_timestamp(self, time_string):
        """Set the times_stamp """
        self.__time_stamp = time_string

    def __get_train_timestamp(self):
        """Returns the last time a neural network was trained """
        return self.__time_stamp

    def load_inputs(self, filename):
        """Load inputs to the neural network from filename"""
        self.set_inputs(self.__read_matrix_from_file(filename))
    
    def load_targets(self, filename):
        """Load corresponding targets for the inputs"""
        self.__targets = self.__read_matrix_from_file(filename)

    def __read_matrix_from_file(self, filename):
        """Returns a matrix read from filename"""
        if filename.endswith(".mat"):
            print 'TODO: implement readInputs for mat files'
            return None
        else:
            return Matrix(loadtxt(filename))

    def __network_error_to_file(self, filename):
        """Write the neural network's error to filename"""
        savetxt(filename, self.get_error())
        return True

    def __stats_to_file(self, filename, sample_mean, sample_variance):
        """Write the sample mean and variance to filename"""
        print '>>> Sample mean: %s' % sample_mean
        print '>>> Sample variance: %s' % sample_variance
        stats_file = open(filename,'w')
        stats_file.writelines("Mean of Network Error for Single Inputs: \n")
        stats_file.writelines(str(sample_mean))
        stats_file.writelines("\n")
        stats_file.writelines("Sample Variance of Network Error for Single Inputs: \n")
        stats_file.writelines(str(sample_variance))
        stats_file.writelines("\n")
        stats_file.close()
        return True

    def __create_vectorneurons(self):
        """Creates the vectorneurons for the neural network"""
        num_inputs = self.get_inputs().shape[0]
        neural_network = self.__neural_network
        num_neurons_per_layer = neural_network.get_num_neurons_per_layer()
        num_layers = neural_network.get_num_layers()
        transfer_functions = neural_network.get_transfer_functions()
        vector_neurons = [0 for i in range(0, num_layers)]

        for i in range(0, num_layers):
            if (i == 0):
                vector_neurons[i] = VectorNeuron(num_inputs, num_neurons_per_layer[i], transfer_functions[i])
            else:
                vector_neurons[i] = VectorNeuron(num_neurons_per_layer[i - 1], num_neurons_per_layer[i], transfer_functions[i])

            self.__time_stamp = time.strftime("%H.%M.%S-%M.%d.%Y")
            vector_neurons[i].write_weight_to_file("layer_weight_init_" + str(i) + "-" + self.__time_stamp)
            vector_neurons[i].write_bias_to_file("layer_bias_" + str(i) + "-" + self.__time_stamp)
        self.set_vector_neurons(vector_neurons)

    def __write_final_weights(self):
        """Write the final weight matrices of all vectorneurons to files"""
        vector_neurons = self.__vector_neurons 
        for i in range(0, len(vector_neurons)):
            vector_neurons[i].write_weight_to_file("layer_weight_final_" + str(i) + "-" + self.__time_stamp)

    def train(self):
        """Trains the neural network"""

        neural_network = self.__neural_network

        if neural_network is None:
            print "no neural network parameters defined."
            return

        self.__create_vectorneurons()
        vector_neurons = self.__vector_neurons

        self.__change_helped = False
        self.set_error(Matrix(rand(neural_network.get_num_epochs())))

        num_input_pairs = self.__inputs.shape[1]
        num_layers = neural_network.get_num_layers()

        for k in range(0, neural_network.get_num_epochs()):
            for j in range(0, neural_network.get_num_layers()):
                vector_neurons[j].weight_matrix_backup()
            self.__org_error = self.__get_training_error_for_epoch()
            if self.__change_helped:
                for j in range(0, num_layers):
                    vector_neurons[j].change_weights()
            else:
                for j in range(0, num_layers):
                    if neural_network.is_use_annealing():
                        vector_neurons[j].compute_delta_w_annealing(neural_network.get_n_value(), neural_network.get_m_value(), neural_network.get_learning_rate())
                    else:
                        vector_neurons[j].compute_delta_w(neural_network.get_m_value(), neural_network.get_learning_rate())
                    vector_neurons[j].change_weights()

            self.__new_error = self.__get_training_error_for_epoch()

            #print "old error: %s" % self.__org_error
            #print "new error: %s" % self.__new_error

            if self.__new_error >= self.__org_error:
                self.__best_error = self.__org_error
                for j in range(0, neural_network.get_num_layers()):
                    vector_neurons[j].rollback_weights()
                self.__change_helped = False
            else:
                self.__best_error = self.__new_error
                self.__change_helped = True

            self.__sample_mean = self.__best_error / num_input_pairs
            self.__sample_variance = (1.0/(num_input_pairs - 1.0)) * (self.__sum_of_x_squared - num_input_pairs * power(self.__sample_mean, 2.0))
            error = self.get_error()
            error[0, k] =  self.__best_error

            if (k == neural_network.get_num_epochs() - 1):
                self.__stats_to_file("network_stats_" + self.__get_train_timestamp(), self.__sample_mean, self.__sample_variance)
                self.__network_error_to_file("network_error_" + self.__get_train_timestamp())
                print ">>> Final error: %s" % self.__best_error
        self.__write_final_weights()

    def __get_training_error_for_epoch(self):
        """Returns the neural network's error for a single epoch"""
        error = 0.0
        difference = None
        neural_network = self.__neural_network
        vector_neurons = self.__vector_neurons
        num_input_pairs = self.__inputs.shape[1]
        num_layers = neural_network.get_num_layers()
        input_rows = self.__inputs.shape[0]
        target_rows = self.__targets.shape[0]

        for i in range(0, num_input_pairs):
            #self.single_input = self.inputs.getMatrix(0, self.inputs.getRowDimension() - 1, i, i)
            self.__single_input = self.__inputs[0:input_rows][:, i:i+1]

            #print "single_input:" 
            #print self.__single_input

            #self.single_target = self.targets.getMatrix(0, self.targets.getRowDimension() - 1, i, i)
            self.__single_target = self.__targets[0:target_rows][:, i:i+1]

            #print "single_target: " 
            #print self.__single_target

            #print "targets:"
            #print self.__targets


            for j in range(0, num_layers):
                if (j == 0):
                    vector_neurons[j].neuron_compute(self.__single_input)
                else:
                    vector_neurons[j].neuron_compute(self.__next_input)
                self.__next_input = vector_neurons[j].get_result()
            
            #print "network output:"
            #print self.__next_input

            #print "target:"
            #print self.__single_target

            difference = self.__next_input - self.__single_target
            error += 0.5 * (difference * difference.transpose())[0, 0]
            self.__sum_of_x_squared += power(0.5 * (difference * difference.transpose())[0, 0], 2)

        return error

    def get_inputs(self):
        """Returns the inputs to the neural network"""
        return self.__inputs

    def set_inputs(self, inputs):
        """Set the inputs to the neural network"""
        self.__inputs = inputs

    def set_error(self, error):
        """Set the current network error"""
        self.__error = error
    
    def get_error(self):
        """Returns the current network error"""
        return self.__error

    def get_targets(self):
        """Returns the inputs' corresponding targets"""
        return self.__targets

    def set_targets(self, targets):
        """Set the corresponding target data for the inputs"""
        self.__targets = targets

    def get_vector_neurons(self):
        """Returns the VectorNeurons for the neural network"""
        return self.__vector_neurons

    def set_vector_neurons(self, vector_neurons):
        """Set this neural network's VectorNeurons"""
        self.__vector_neurons = vector_neurons

    def set_network_parameters(self, np):
        """Set the network parameters for this neural network"""
        self.__neural_network = np
