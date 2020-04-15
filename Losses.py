import numpy as np

class Loss():
    def __call__(prediction, target):
        return Call(prediction, target)

    def Call(prediction, target):
        print("Write a call function for this loss function!")

class MSE(Loss):
    def Call(prediction, target):
        target = np.array(target)
        prediction = np.array(prediction)
        output = (target - prediction)**2
        output = np.sum(output)
        output = output/target.shape[0]
        return output

    def TotalOut_Gradient(self, prediction, target):
        prediction = np.array(prediction)
        target = np.array(target)
        return prediction - target

