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

class NetworkParameters(object):

    """
    Configuration class for defining Neural Network parameters
    """

    __num_layers = 0
    __m_value = 0
    __n_value = 0
    __num_epochs = 0
    __num_neurons_per_layer = []
    __transfer_functions = []
    __learning_rate = 0.0
    __use_annealing = False

    def get_learning_rate(self):
        """doc string"""
        return self.__learning_rate

    def set_learning_rate(self, learning_rate):
        """doc string"""
        self.__learning_rate = learning_rate

    def get_m_value(self):
        """doc string"""
        return self.__m_value

    def set_m_value(self, m_value):
        """doc string"""
        self.__m_value = m_value

    def get_n_value(self):
        """doc string"""
        return self.__n_value

    def set_n_value(self, n_value):
        """doc string"""
        self.__n_value = n_value

    def get_num_epochs(self):
        """doc string"""
        return self.__num_epochs

    def set_num_epochs(self, num_epochs):
        """doc string"""
        self.__num_epochs = num_epochs

    def get_num_layers(self):
        """doc string"""
        return self.__num_layers

    def set_num_layers(self, num_layers):
        """doc string"""
        self.__num_layers = num_layers

    def get_num_neurons_per_layer(self):
        """doc string"""
        return self.__num_neurons_per_layer

    def set_num_neurons_per_layer(self, num_neurons_per_layer):
        """doc string"""
        self.__num_neurons_per_layer = num_neurons_per_layer

    def get_transfer_functions(self):
        """doc string"""
        return self.__transfer_functions

    def set_transfer_functions(self, transfer_functions):
        """doc string"""
        self.__transfer_functions = transfer_functions

    def is_use_annealing(self):
        """doc string"""
        return self.__use_annealing

    def set_use_annealing(self, use_annealing):
        """doc string"""
        self.__use_annealing = use_annealing

    def parametersummary(self):
        """doc string"""
        print "num_layers: " + self.__num_layers
        print "m_value: " + self.__m_value
        print "n_value: " + self.__n_value
        print "num_epochs: " + self.__num_epochs
        print "num_neurons_per_layer: " + self.__num_neurons_per_layer
        print "transfer_fuctions: " + self.__transfer_functions
        print "learning_rate : " + self.__learning_rate
        print "use_annealing: " + self.__use_annealing
