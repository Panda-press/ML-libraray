import numpy as np

class ActivationFunction():
    def __call__(self, input_vector):
        return self.Call(input_vector)

    def Call(self, input_vector):
        print("Write a call function for this activation function!")
        

class Linear(ActivationFunction):
    def Call(self, input_vector):
        return input_vector

    def OutInput_Gradient(self, x):
        return x

class Relu(ActivationFunction):
    def Call(self, input_vector):
        func = np.vectorize(Relu.Rectify)
        return func(input_vector)

    def Rectify(value):
        if value > 0:
            return value
        else:
            return 0

class Sigmoid(ActivationFunction):
    def Call(self, input_vector):
        func = np.vectorize(Sigmoid.function)
        return func(input_vector)
    
    def function(value):
        result = 1/(1 + np.exp(-value))
        return result

    def OutInput_Gradient(self, x):
        func = np.vectorize(Sigmoid.function_derivative)
        return func(x)

    def function_derivative(x):
        result = Sigmoid.function(x) * (np.ones_like(x) - Sigmoid.function(x))
        return result

