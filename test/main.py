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

import os

from pyneural.networkparameters import NetworkParameters
from pyneural.train import Train

def main():
    print ">>> Creating 3-layer neural network..."
    np = NetworkParameters()
    np.set_num_layers(3) 
    np.set_num_epochs(1000)
    np.set_num_neurons_per_layer([2,2,1])
    np.set_transfer_functions(["logsig", "logsig", "logsig"])
    np.set_use_annealing(False)
    np.set_m_value(3)
    np.set_learning_rate(0.5)

    print ">>> Loading inputs/targets..."
    train = Train(np)
    train.load_inputs('data/xor.inputs')
    train.load_targets('data/xor.targets')

    print ">>> Creating results directory for storing the results..."
    os.mkdir('results')
    os.chdir('./results')
    
    print ">>> Training the neural network..."
    train.train()
    print ">>> Training complete!"

if __name__ == "__main__":
    main()
