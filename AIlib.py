# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def sigmoid(x):
    return 2/(1+np.exp(-x))-1

def linearstep(x):
    if x<-1:
        return -1
    if x>1:
        return 1
    return x

def d_linearstep(x):
    if abs(x)>1:
        return 0
    return 1

class Neuron:
    def __init__(self,fathers=None, function=sigmoid):
        self.fathers = fathers #deben ser clse neuron.
        self.father_weights = np.random.uniform(low=-1.0, high=1.0, size=np.shape(fathers))
        self.value = 0.0
        self.fvalue = 0.0
        self.function = function
        
    def Get_value(self):
        return self.value

    def Get_fvalue(self):
        return self.fvalue
    
    def Get_weight(self, js):
        return self.father_weights[js]
    
    def Calculate_value(self):
        self.value = sum([father_neuron.Get_fvalue()*self.father_weights[i] for i,father_neuron in enumerate(self.fathers)])
        
    def Calculate_fvalue(self):
        self.fvalue = self.function(self.value)


class Net:
    def __init__(self, shape, function):
        self.layer = []
        self.layer.append([Neuron(None ,function) for j in range(shape[0])])#capa 1
        for i,layer_size in enumerate(shape[1:]):
            self.layer.append([Neuron(self.layer[i] ,function) for j in range(layer_size)])
            

    def Think(self, in_values):
        for in_value,neuron in zip(in_values,self.layer[0]):
            neuron.value = in_value
            neuron.fvalue = in_value
        for layer in self.layer[1:]:
            for neuron in layer:
                neuron.Calculate_value()
                neuron.Calculate_fvalue()
        return np.array([neuron.Get_fvalue() for neuron in self.layer[-1]])

    def Setting_Learning_Parameters(self, alpha_states):
        self.alphas = alpha_states#alpha changes depending of what type of state is learning: winn, loose, the games goes on
        
    def Learn(self, in_values, game_states):
        pass
    
Net_Test = Net([2,2,1],sigmoid)
print(Net_Test.Think([1,1]))


