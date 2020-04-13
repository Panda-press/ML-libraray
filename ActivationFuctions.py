import numpy as np

class ActivationFunction():
    def __call__():
        print("Write a call function for this activation function!")

class Linear(ActivationFunction):
    def __call__(input_vector):
        return input_vector


class Relu(ActivationFunction):
    def __call__(self, input_vector):
        func = np.vectorize(Relu.Rectify)
        return func(input_vector)

    def Rectify(value):
        if value > 0:
            return value
        else:
            return 0

