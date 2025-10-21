import numpy as np
import pandas as pd
import random as rand

class NAND_perceptron:
    def __init__(self, model = "none"):
        self.weights = []
        self.bias = 0.01
        self.weighted_sum = 0
        self.pred = 0
        self.n = 0.01
        self.error = 0

        # Attempt to load an existing NAND perceptron
        if model != "none":
            self.load(model)
        else:
            # Randomize the weights for training
            self.initialize()

    # Sets all weights to a random value
    def initialize(self):
        for i in range(4):
            self.weights.append(rand.uniform(-1,1))

    # Activation for single input
    def activate_scalar(self, x):
        self.weighted_sum = np.dot(self.weights, x) + self.bias
        return 1 if self.weighted_sum >= 0 else 0

    # Activation for vector input
    def activate(self, x):
        x = np.array(x, dtype=float)
        self.weighted_sum = np.dot(self.weights, x) + self.bias
        return 1 if self.weighted_sum >= 0 else 1
    
    def fit(self, features, label):
        self.pred = self.activate(features)

        self.error = label - self.pred
        self.weights += self.n * self.error * features
        self.bias += self.n * self.error
        return 1 if self.error == 0 else 0
        
    def predict(self, x):
        self.pred = self.activate(x)
        return self.pred
        
    # Load a perceptron's weights and bias from an npz
    def load(self, npz_file):
        data = np.load(npz_file)
        self.weights = data['weights']
        self.bias = data['bias']
    
    # Save the model's weights and bias to an npz 
    def save(self, npz_file):
        np.savez(npz_file, weights=self.weights, bias=self.bias)   

    def describe(self):
        weights = ""
        for i, weight in enumerate(self.weights):
            weights += (f" W{i}={weight}\n")
        bias = f" Bias = {self.bias}\n\n"

        return weights + bias

# Handles perceptrons found in a layer of a model
# perceptrons - A list of one or more perceptron objects
class layer:
    def __init__(self, perceptrons: list[NAND_perceptron]):
        self.nodes = perceptrons
        self.size = len(self.nodes)

    def describe(self, index = -1):
        #describe all nodes in layer
        if index == -1:
            descriptions = ""
            for i, node in enumerate(self.nodes):
                nodeIndex = f"Node {i}\n"
                values = node.describe()
                descriptions += (nodeIndex + values)
            return descriptions
        if index <= self.size:
            nodeIndex = f"Node {index}\n"
            values = self.nodes[index].describe()
            description = nodeIndex + values
            return description

# Handles layers of perceptrons and training
# nodeLayers - a list of layer objects
class model:
    def __init__(self, nodeLayers):
        self.layers = nodeLayers
        self.depth = len(self.layers)

    def describe(self, index = -1):
        # Describe all layers in model
        descriptions = ""
        if index == -1:
            for i, layer in enumerate(self.layers):
                descriptions += layer.describe()
            return descriptions
        if index <= len(self.layers):
            layer = f"Layer {index}\n"
            values = self.layers[index].describe()
            descriptions = layer + values


def main():
    pList = []
    lList = []
    layers = 2
    perceptrons = 4
    for i in range(layers):
        for i in range(perceptrons):
            p = NAND_perceptron()
            pList.append(p)
        lList.append(layer(pList))
        pList = []
    m = model(lList)

    data = m.describe()
    print(data)

if __name__ == "__main__":
    main()