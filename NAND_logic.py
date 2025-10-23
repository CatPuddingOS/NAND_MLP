import numpy as np
import pandas as pd
import random as rand

class NAND_perceptron:
    def __init__(self, weights, bias, model = "none"):
        self.weights = weights
        self.bias = bias
        self.weighted_sum = 0
        self.pred = 0
        self.n = 0.01
        self.error = 0

        # Attempt to load an existing NAND perceptron
        if model != "none":
            self.load(model)
        # else:
        # depreciated
        #     # Randomize the weights for training
        #     self.initialize()

    # Sets all weights to a random value
    # Depreciated
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
    
    # depreciated
    def fit(self, features, label):
        self.pred = self.activate(features)

        self.error = label - self.pred
        self.weights += self.n * self.error * features
        self.bias += self.n * self.error
        return 1 if self.error == 0 else 0

    # depreciated   
    def predict(self, x):
        self.pred = self.activate(x)
        return self.pred

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
    def __init__(self, layers = [], npz_file = "none"):
        self.layers = layers
        self.depth = len(self.layers)
        self.iter_range = 1
        
        if npz_file != "none":
            self.load(npz_file)
            return

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

    # Train the model on a dataset
    # dataset - A pandas dataframe containing all features and labels
    def fit(self, dataset):
        for k in range(self.iter_range):
            # Randomize dataset order
            pm = np.random.permutation(dataset.index)
            sh_features, sh_labels = dataset.iloc[pm, :-1], dataset.iloc[pm, -1]

            # Check hits
            hit = 0
            # Training n' stuff
            for l, layer in enumerate(self.layers):
                print(f"{layer.size} nodes in layer {l}\n")
                for p, perceptron in enumerate(layer.nodes):
                    print(f"evaluating perceptron {p} of layer {l}...\n")
                    for i, feature in enumerate(np.array(sh_features, dtype=float)):
                         print(f"dataset = {feature}, label = {sh_labels[i]}")

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
                if newNode_weights is not None and newNode_bias is not 0:
                    #print(f"Finalizing node for layer {l}\n")
                    node = NAND_perceptron(newNode_weights, newNode_bias)
                    newLayer.nodes.append(node)
                    newNode_weights = []
                    newNode_bias = 0
            #print(f"---Finalizing layer {l}---\n")
            self.layers.append(newLayer)

# Create a model l layers deep with n perceptrons per layer
def debugGenerateModel(l, n):
    pList = []
    lList = []
    layers = l
    perceptrons = n
    for i in range(layers):
        for i in range(perceptrons):
            weights = []
            for j in range(4):
                weights.append(rand.uniform(-1,1))
            p = NAND_perceptron(weights, rand.uniform(-1,1))
            pList.append(p)
        lList.append(layer(pList))
        pList = []
    m = model(lList)
    return m

def debugLoadModel(file):
    m = model([], file)
    return m

def main():
    # Setting up a model w/ 2 layers of 4 perceptrons each
    m = debugGenerateModel(2, 4)
    print(m.describe())
    m.save("NAND_model_test.npz")
    m.load("NAND_model_test.npz")
    print(m.describe())

    # Loading to instantiate
    m2 = debugLoadModel("NAND_model_test.npz")
    print(m2.describe())


    # Dataset
    df = pd.read_csv("NAND_model_dataset.csv")

if __name__ == "__main__":
    main()