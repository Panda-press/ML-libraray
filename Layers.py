import numpy as np
import ActivationFuctions as af
import Losses as Losses

class Layer():
    def __call__(self, input_vector):
        return self.Call(input_vector)
    
    def Call(self, input_vector):
        print("write a call function!")


class Dense1D(Layer):
    def __init__(self, input_size, output_size, activation = af.Linear()):
        self.matrix = np.random.rand(output_size, input_size)# * 1e-100
        self.bias = np.random.rand(output_size)# * 1e-100
        self.activation = activation

    def Call(self, input_vector, use_activation = True):
        result = self.matrix @ np.array(input_vector).T
        result = result + self.bias
        if (use_activation):
            result = self.activation(result)
        return result

    def Calculate_Gradients(self, input_vector, target_output, loss, learning_rate = 1e-3):

        outNet_gradient = self.activation.OutInput_Gradient(self.Call(input_vector, use_activation=False))
        totalOut_gradient = loss.TotalOut_Gradient(target_output, self(input_vector))
        input_vector = np.array([input_vector])

        weights_gradient =  np.dot(input_vector.T, np.array([outNet_gradient * totalOut_gradient * learning_rate])).T
        bias_gradient = 1 * outNet_gradient * totalOut_gradient * learning_rate

        return weights_gradient, bias_gradient

    def Apply_Gradients(self, weights_gradient, bias_gradient):
        self.matrix += weights_gradient
        self.bias += bias_gradient

