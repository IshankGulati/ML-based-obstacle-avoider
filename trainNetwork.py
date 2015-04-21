import loader
import NeuralNetwork as nn 


def trainNetwork():
    trainingData = loader.load()
    nn.nNetwork(trainingData,'network.mat')

if __name__ == "__main__":
    trainNetwork()
