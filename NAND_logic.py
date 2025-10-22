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

    def describe(self):
        weights = ""
        for i, weight in enumerate(self.weights):
            weights += (f"    W{i}={weight}\n")
        bias = f"    Bias = {self.bias}\n\n"
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
    def __init__(self, nodeLayers):
        self.layers = nodeLayers
        self.depth = len(self.layers)
        self.iter_range = 1

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
                    # # ensure feature is np array
                    #     x = np.array(feature, dtype=float)
                    # # activate perceptron
                    #     perceptron.fit(x, sh_labels[i])

                    #     if perceptron.pred == sh_labels[i]:
                    #         hit += 1
                    # print(f"Layer [{layer}], {hit} out of {dataset.shape[0]}")
                    # hit = 0

    def save(self, npz_file):
        model_data = {}
        for l, layer in enumerate(self.layers):
            for n, node in enumerate(layer.nodes):
                model_data[f"L{l}_N{n}_weights"] = node.weights
                model_data[f"L{l}_N{n}_bias"] = node.bias
        np.savez(npz_file, **model_data)

    def load(self, npz_file):
        model_data = np.load(npz_file)
        for l, layer in enumerate(self.layers):
            for n, node in enumerate(layer.nodes):
                node.weights = model_data[f"L{l}_N{n}_weights"]
                node.bias = model_data[f"L{l}_N{n}_bias"]

def main():
    # Setting up a model w/ 2 layers of 4 perceptrons each
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

    # Dataset
    df = pd.read_csv("NAND_model_dataset.csv")

    m.save("NAND_model_test.npz")
    m.describe()
    m.load("NAND_model_test.npz")
    m.describe()

    #results = m.fit(df)

    #print(f"Model accuracy: {results * 100:.2f}%")

if __name__ == "__main__":
    main()