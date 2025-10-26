import numpy as np
import pandas as pd
import random as rand
import sys

class NAND_perceptron:
    def __init__(self, weights, bias, model = "none"):
        self.weights = weights
        self.bias = bias
        self.weighted_sum = 0
        self.pred = 0
        self.n = 0.1
        self.error = 0

        # Attempt to load an existing NAND perceptron
        if model != "none":
            self.load(model)

    # Sets all weights to a random value
    def initialize(self):
        self.weights = np.random.uniform(-1,1,size=4)

    # Activation for vector input
    def activate(self, x):
        x = np.array(x, dtype=float)
        self.weighted_sum = np.dot(self.weights, x) + self.bias
        return 1 if self.weighted_sum >= 0 else 0
    
    def fit(self, features, label):
        features = np.array(features, dtype=float)
        self.pred = self.activate(features)

        self.error = label - self.pred
        self.weights += self.n * self.error * features
        self.bias += self.n * self.error
        return self.pred # Initially returned pred based on error for some reason. Fixed training
  
    def predict(self, features):
        self.pred = self.activate(features)
        return self.pred

    def load(self, model_npz):
        model_data = np.load(model_npz)
        self.weights = model_data["weights"]
        self.bias = model_data["bias"]

    def describe(self):
        weights = ""
        for i, weight in enumerate(self.weights):
            weights += (f"    W{i}={weight}\n")
        bias = f"    Bias = {self.bias}\n\n"
        return weights + bias

# Handles perceptrons found in a layer of a model
# perceptrons - A list of one or more perceptron objects
class layer:
    def __init__(self, perceptrons: list[NAND_perceptron] = []):
        self.nodes = perceptrons
        self.size = len(self.nodes)

    def describe(self, index = -1):
        #describe all nodes in layer
        if index == -1:
            descriptions = ""
            for i, node in enumerate(self.nodes):
                nodeIndex = f"  Node {i}\n"
                values = node.describe()
                descriptions += (nodeIndex + values)
            return descriptions
        if index <= self.size:
            nodeIndex = f"  Node {index}\n"
            values = self.nodes[index].describe()
            description = nodeIndex + values
            return description

# Handles layers of perceptrons and training
# nodeLayers - a list of layer objects
class model:
    def __init__(self, layers: list[layer] = [], npz_file = "none"):
        #in terms of layers, a "second layer" would mean 1 input layer(no activation), 1 hidden layer and 1 output layer
        self.layers = layers
        self.depth = len(self.layers)
        self.fit_range = 100
        
        if npz_file != "none":
            self.load(npz_file)

    def describe(self, index = -1):
        # Describe all layers in model
        descriptions = ""
        if index == -1:
            for i, layer in enumerate(self.layers):
                descriptions += f"Layer {i}\n"
                descriptions += layer.describe()
            return descriptions
        if index <= len(self.layers):
            layer = f"Layer {index}\n"
            values = self.layers[index].describe()
            descriptions = layer + values

    # Inset a layer into layers
    def debugAddLayer(self, n):
        for i in range(n):
            pList = []
            for j in range(4): # 4 randomly generated weights
                weights = []
                for k in range(4):
                    weights.append(rand.uniform(-1,1))
                p = NAND_perceptron(weights, rand.uniform(-1,1))
                pList.append(p)
            newLayer = layer(pList)
            self.layers.insert(-1, newLayer)
        self.depth = len(self.layers)

    # Train the model on a dataset
    # dataset - A pandas dataframe containing all features and labels
    def fit(self, dataset):
        for k in range(1):
            # Randomize dataset order
            pm = np.random.permutation(dataset.index)
            sh_features = dataset.iloc[pm, :-1].to_numpy(dtype=float)
            sh_labels = dataset.iloc[pm, -1].to_numpy(dtype=int)

            correct = 0
            guesses = []
            finalOutput = 0
            for i, feature in enumerate(sh_features):
                guesses = []
                finalOutput = 0
                print(f"Trying: {feature}")
                for l in range(self.depth - 1):
                    thisLayer = self.layers[l]
                    for node in thisLayer.nodes:
                        # have layer make predictions
                        guess = node.predict(feature)
                        guesses.append(guess)
                        # pass predictions to next layer as the inputs
                    for node in self.layers[l+1].nodes:
                        next = node.predict(guesses)
                        if next == sh_labels[i]:
                            correct+=1
                        print(f"Guesses: {guesses}, Final Output: {next}, Expected: {sh_labels[i]}")
            accuracy = correct / len(sh_features)
            print(accuracy * 100)
                        
    # Layer X, Node Y = weights and bias
    def save(self, npz_file):
        model_data = {}
        for l, layer in enumerate(self.layers):
            for n, node in enumerate(layer.nodes):
                model_data[f"L{l}_N{n}_weights"] = node.weights
                model_data[f"L{l}_N{n}_bias"] = node.bias
        np.savez(npz_file, **model_data, model_depth=int(self.depth))

    # Load in the same fashion as save. Update: nevermind
    # https://www.reddit.com/r/gifs/comments/26aikq/there_has_to_be_a_better_way/
    def load(self, npz_file):
        print("LOADING MODEL...\n")
        model_data = np.load(npz_file)
        self.depth = model_data["model_depth"]
        print(f"Loaded depth = {self.depth}")

        for l in range(self.depth):
            #print(f"Setting layer {l}")
            newLayer = layer()
            newLayer.nodes = []
            newNode_weights = []
            newNode_bias = 0
            for key in model_data.keys():
                #print("Checking keys...")
                if key.startswith(f"L{l}") and key.endswith("_weights"):
                    #print(f"Weight hit for layer {l} with:\n  {key}")
                    newNode_weights = model_data[key]
                if key.startswith(f"L{l}") and key.endswith("_bias"):
                    #print(f"Bias hit for layer {l} with:\n  {key}")
                    newNode_bias = model_data[key]
                if newNode_weights is not None and newNode_bias != 0:
                    #print(f"Finalizing node for layer {l}\n")
                    node = NAND_perceptron(newNode_weights, newNode_bias)
                    newLayer.nodes.append(node)
                    newNode_weights = []
                    newNode_bias = 0
            #print(f"---Finalizing layer {l}---\n")
            newLayer.size = len(newLayer.nodes)
            self.layers.append(newLayer)

# Create a model l layers deep with n perceptrons per hidden layer and 1 perceptron on the last layer
def debugGenerateModel(l, n, model_npz = "none"):
    pList = []
    lList = []

    for i in range(l - 1):
        for k in range(n):
            weights = []
            for j in range(4):
                weights.append(rand.uniform(-1,1))
            p = NAND_perceptron(weights, rand.uniform(-1,1), model_npz)
            pList.append(p)
        lList.append(layer(pList))
        pList = []

    # Output layer single perceptron
    weights = []
    for i in range(4):
        weights.append(rand.uniform(-1,1))
    p = [NAND_perceptron(weights, rand.uniform(-1,1), model_npz)] # even one must be passed as a list because foresight
    lList.append(layer(p))
    m = model(lList)
    return m

def debugLoadModel(file):
    m = model([], file)
    return m

def main():
    m3 = debugGenerateModel(2, 4, "NAND_single_params.npz")

    df = pd.read_csv("AND.csv")
    m3.fit(df)

if __name__ == "__main__":
    main()