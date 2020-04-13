import numpy as np
import ActivationFuctions as af

class Layer():
    def __call__(self, input_vector):
        return self.Call(input_vector)
    
    def Call(self, input_vector):
        print("write a call function!")


class Dense1D(Layer):
    def __init__(self, input_size, output_size, activation = af.Linear()):
        self.matrix = np.empty((input_size, output_size)).T
        self.activation = activation

    def Call(self, input_vector):
        result = np.dot(self.matrix, np.array([input_vector]).T)
        result = self.activation(result)
        return result


dense = Dense1D(3,2, activation=af.Relu())
print(dense.matrix)
print(dense([-2,3,2]))