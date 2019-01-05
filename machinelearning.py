"""
Xavier McMaster - Hubner
January 4th 2019
Here is my simple machine learning applicatio nwith layers that can be decided by the user using numpy and MNIST.
"""
#Here I import numpy, the mnist images and scipy for my sigmoid activation function
import numpy as np
import mnistdb.io as mio
from scipy.special import expit

#Here I decide the function's learning rate and the layers to be used by the network
learningRate = 0.02
layers = [784,16,16,10]

#Here is the sigmoid activation function not currently being used by the network
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Here I create the network from the layers defined up above and give the layers randomised weights and biases
def creatingWeights(layers):
    weightsBiases = []

    #In this loop the weights and biases are created
    for i in range(len(layers)):
        if i < (len(layers) - 1):
            indexWeightsBiases = []
            indexWeightsBiases.append(np.random.uniform(low = -1, high = 1,size =(layers[i],layers[i+1])))
            indexWeightsBiases.append(np.random.uniform(low = -3, high = 3,size =(1,layers[i+1])))
            weightsBiases.append(indexWeightsBiases)
    return weightsBiases

#Here I train the network created above using the MNIST Database
def train(weightsBiases):

    #From the MNIST module I load the pixel greyscale values of the images and the correct number values of those images
    correctNumbers = mio.load()
    data = mio.load(scaled=True)
    correctNumbersIndex = mio.load(one_hot=True)
    
    #Here I train the network based on a predetermined number of "epochs"
    for k in range(30):
        totalTries = 0
        totalCorrect = 0

        #In this loop I run through every image in the database
        for i in range(60000):
            tries = 0
            correct = 0
            number = data.trainX[i]
            number = np.array(number,ndmin=2)
            currentValue = number
            layerOutput = np.empty(len(layers)-1, dtype=object)

            #In this loop I run the image through the neural network and create a list with the results of all of the layers
            for j in range(len(weightsBiases)):
                currentValue = currentValue.dot(weightsBiases[j][0]) + weightsBiases[j][1]
                currentValue = expit(currentValue)
                layerOutput[j] = currentValue

            #Here I take the correct value of the number vs the value spit out by the network and tally up the correct guesses for this ephoc and the entire training
            index = np.argmax(currentValue)
            if index == correctNumbers.trainY[i]:
                correct = correct + 1
                totalCorrect = totalCorrect + 1

            #Here I calculate the output cost (the error) of the network
            outputCost = (currentValue-correctNumbersIndex.trainY[i])**2

            #In this loop I take the derivative of the network and adjust the weights and biases based on that derivative
            for j in range(len(weightsBiases)):
                layer = len(weightsBiases) - j - 1
                derivative = (learningRate * outputCost * (layerOutput[layer] * (1 - layerOutput[layer]))).transpose()
                weightDerivative = derivative.dot(layerOutput[layer-1]).transpose()
                weightsBiases[layer-1][0] = weightsBiases[layer-1][0] + weightDerivative
                weightsBiases[layer-1][1] = weightsBiases[layer-1][1] + (derivative)
                outputCost = weightsBiases[layer-1][0].transpose().dot(outputCost)
            
            #Here I increment the tries of the network and print out the results of the current epoch and of all of the epochs later
            tries = tries + 1
            totalTries = totalTries + 1
            print(correct/tries)
        print(totalCorrect/totalTries)

#Here is the main function, where the function calls are located
def main():
    weightsBiases = creatingWeights(layers)
    train(weightsBiases)

#Here I call main
main()
