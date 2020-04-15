import numpy as np
import Layers as ls
import ActivationFuctions as af
import Losses as Losses

class Model():
    def __init__(self):
        self.layers = []
    
    def Add(self, layer):
        self.layers.append(layer)

    def Call(self, value):
        output = value
        for layer in self.layers:
            #print(output)
            output = layer(output)

        return output

    def Train_Once(self, input, target, loss, learning_rate = 1e-3):
        weight_gradient, bias_gradient = np.zeros_like(self.layers[0].matrix), np.zeros_like(self.layers[0].bias)
        for i in range(0, input.shape[0]):
            extra_weight_gradient, extra_bias_gradient = self.layers[0].Calculate_Gradients(input[i], target[i], loss, learning_rate)
            
            
            weight_gradient += extra_weight_gradient
            bias_gradient += extra_bias_gradient

        weight_gradient /= input.shape[0]
        bias_gradient /= input.shape[0]
        self.layers[0].Apply_Gradients(weight_gradient, bias_gradient)
    



my_model = Model()
my_model.Add(ls.Dense1D(2, 2, activation=af.Sigmoid()))

for i in range(0,1000):
    my_model.Train_Once(np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([[1,0],[0,1],[0,1],[0,1]]), Losses.MSE(), 1)
    #print(my_model.layers[0].matrix)
    
print(my_model.Call([0,0]))
print(my_model.Call([0,1]))
print(my_model.Call([1,0]))
print(my_model.Call([1,1]))

