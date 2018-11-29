import numpy as np
from matplotlib import pyplot as plt
import mnistdb.io as mio
from scipy.special import expit
learningRate = 0.02
biasLearningRate = 0.01
#layers = [3,5,5,1]
layers = [784,16,16,10]
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def creatingWeights(layers):
    weightsBiases = []
    for i in range(len(layers)):
        if i < (len(layers) - 1):
            indexWeightsBiases = []
            indexWeightsBiases.append(np.random.uniform(low = -1, high = 1,size =(layers[i],layers[i+1])))
            indexWeightsBiases.append(np.random.uniform(low = -3, high = 3,size =(1,layers[i+1])))
            weightsBiases.append(indexWeightsBiases)
    return weightsBiases
weightsBiases = creatingWeights(layers)

def train(weightsBiases):
    correctNumbers = mio.load()
    data = mio.load(scaled=True)
    correctNumbersIndex = mio.load(one_hot=True)
    
    
    for k in range(30):
        tries = 0
        correct = 0
        for i in range(60000):
            
            number = data.trainX[i]
            number = np.array(number,ndmin=2)
            currentValue = number
            layerOutput = np.array([])
            for j in range(len(weightsBiases)):
                currentValue = currentValue.dot(weightsBiases[j][0]) + weightsBiases[j][1]
                #print(currentValue)
                currentValue = expit(currentValue)
                #print(currentValue)
                layerOutput.np.append(currentValue)
            #print(currentValue)
            index = np.argmax(currentValue)
            #print(index)
            if index == correctNumbers.trainY[i]:
                correct = correct + 1
            outputCost = (currentValue-correctNumbersIndex.trainY[i])**2
            #print(correctNumbersIndex.trainY[i])
            #print(outputCost)
            for j in range(len(weightsBiases)):
                layer = len(weightsBiases) - j - 1
                derivative = outputCost * layerOutput[layer]*(1- layerOutput[layer])
                weightDerivative = derivative * learningRate
                biasDerivative = derivative * biasLearningRate
                #print(weightsBiases[layer][0])
                weightsBiases[layer][0] = weightsBiases[layer][0] + (layerOutput[layer-1][0]).dot(weightDerivative)
                #weightDerivative = weightDerivative.dot(layerOutput[layer][0])
                #print(weightsBiases[layer][0])
                weightsBiases[layer][1] = weightsBiases[layer][1] + (biasDerivative)
                #biasDerivative = (weightDerivative/learningRate)*biasLearningRate
                outputCost = weightsBiases[layer][0].dot(outputCost.T)
                outputCost = outputCost.T
            for j in range(len(weightsBiases)):
                layer = len(weightsBiases) - j - 1
                weightsBiases[layer][0] = weightsBiases[layer][0]
            tries = tries + 1
        print(correct/tries)
train(weightsBiases)