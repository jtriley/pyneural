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

import os,sys,time
import math
from numpy import abs, exp, savetxt
from numpy.random import rand
from numpy import matrix as Matrix
from numpy.random.mtrand import RandomState as MersenneTwister

class VectorNeuron(object):
    """
    The VectorNeuron class represents a single weight matrix and a corresponding
    transfer function.  

    An input to a VectorNeuron is first multiplied by the weight matrix.  The 
    result is then fed through a transfer function to produce the VectorNeuron's 
    output. 

    The output is either the containing neural network's final output or the input
    to another VectorNeuron.
    """

    __weight_matrix = None
    __weight_matrix_backup = None
    __bias_vector = None
    __delta_w_matrix = None
    __result = None
    __transfer_function = ""
    __mersenne_twister = None

    def __init__(self, left_num_neurons, right_num_neurons, transfer_function):
        """
        Initializes a vectorneuron with a weight matrix of size (left_num_neurons, right_num_neurons), 
        a bias vector of size (right_num_neurons, 1), and a transfer function transfer_function.
        """

        print '>>> Creating VectorNeuron: (%s, %s) %s' % \
        (left_num_neurons, right_num_neurons, transfer_function)

        self.__weight_matrix = Matrix(rand(right_num_neurons, left_num_neurons))
        self.__weight_matrix_backup = self.__weight_matrix.copy()
        self.__bias_vector = Matrix(rand(right_num_neurons, 1))
        self.__delta_w_matrix = Matrix(rand(right_num_neurons, left_num_neurons))
        self.__mersenne_twister = MersenneTwister()
        self.__mersenne_twister.seed(int(1000*time.time()))
        self.__transfer_function = transfer_function

    def neuron_compute(self, input_matrix):
        """Computes the vectorneuron output for input_matrix"""
        self.__result = self.__weight_matrix * input_matrix
        current_value = None
        transfer_function = self.__transfer_function
        row_dim = self.__weight_matrix.shape[0]
        input_col_dim = input_matrix.shape[1]

        for i in range(0,row_dim):
            for j in range(0, input_col_dim):
                current_value = self.__result[i, j]
                self.__result[i, j]= self.__bias_vector[i, 0] + current_value
                cmd = "self.%s(current_value)" % transfer_function
                self.__result[i, j] = eval(cmd)

    def compute_delta_w(self, m, lr):
        """Computes new delta_w matrix"""
        k = 0
        delta_w = None
        delta_w_row_dim = self.__delta_w_matrix.shape[0]
        delta_w_col_dim = self.__delta_w_matrix.shape[1]

        for i in range(0, delta_w_row_dim):
            for j in range(0,delta_w_col_dim):
                k = abs(self.__mersenne_twister.randint(0,math.pow(2,32)) % m)
                if k == 0:
                    delta_w = lr
                elif k == 1:
                    delta_w = -1.0 * lr
                else:
                    delta_w = 0.0
                self.__delta_w_matrix[i, j] = delta_w

    def compute_delta_w_annealing(self, n, m, lr):
        """Computes new delta_w matrix (annealing style)"""
        k = 0
        delta_w = None
        delta_w_row_dim = self.__delta_w_matrix.shape[0]

        for i in range(0,delta_w_row_dim):
            delta_w_matrix_col = self.__delta_w_matrix.shape[1]
            for j in range(0, delta_w_matrix_col):
                k = abs(self.__mersenne_twister.randint(0,math.pow(2,32)) % m)
                if k < n:
                    if k % 2 == 0:
                        if (k == 0):
                            delta_w = lr
                        else:
                            delta_w = lr / k
                    elif k % 2 == 1:
                        delta_w = -1.0 * lr / k
                    else:
                        delta_w = 0.0
                else:
                    delta_w = 0.0
                self.__delta_w_matrix[i, j]  = delta_w

    def logsig(self, x):
        """Returns logsig of a single variable x"""
        return 1.0/(1.0 + exp(-1.0 * x))

    def purelin(self, x):
        """Returns purelin of a single variable x"""
        return x

    def tansig(self, x):
        """Returns tansig of a single variable x"""
        return 2.0/exp(1.0 + exp(-2.0 * x), -1.0)

    def linsig(self, x):
        """Returns linsig of a single variable x"""
        if x <= 1.0 and x >= -1.0:
            return x
        if x > 1:
            return 1.0
        else:
            return -1.0

    def change_weights(self):
        """Changes weight_matrix by adding delta_w_matrix"""
        #print 'weight_matrix orig'
        #print self.__weight_matrix
        self.__weight_matrix = self.__weight_matrix + self.__delta_w_matrix
        #print 'weight matrix new'
        #print self.__weight_matrix

    def rollback_weights(self):
        """Reset weight_matrix to weight_matrix_backup"""
        #print 'resetting weights'
        self.__weight_matrix = self.__weight_matrix_backup.copy()

    def weight_matrix_backup(self):
        """Copies the current weight_matrix to weight_matrix_backup"""
        self.__weight_matrix_backup = self.__weight_matrix.copy()

    def get_bias(self):
        """Returns the vectorneuron's bias vector"""
        return self.__bias_vector

    def get_delta_w(self):
        """Return the computed delta_w matrix used to alter the weights"""
        return self.__delta_w_matrix

    def get_result(self):
        """Returns the output of vectorneuron's neuron_compute function"""
        return self.__result

    def get_weight_matrix(self):
        """Returns the vectorneuron's current weight_matrix"""
        return self.__weight_matrix

    def get_weight_matrix_backup(self):
        """Returns a backup of the vectorneuron's previous weight_matrix"""
        return self.__weight_matrix_backup

    def get_transfer_function(self):
        """Returns the vectorneuron's transfer function"""
        return self.__transfer_function

    def write_weight_to_file(self, filename):
        """Write the vectorneuron's weight_matrix to filename """
        savetxt(filename, self.__weight_matrix)
        return True

    def write_bias_to_file(self, filename):
        """Write the vectorneuron's biias vector to filename"""
        savetxt(filename, self.__bias_vector)
        return True
