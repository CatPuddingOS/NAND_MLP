import numpy as np
import pandas as pd
import random as rand

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
        # else:
        # depreciated
        #     # Randomize the weights for training
        #     self.initialize()

    # Sets all weights to a random value
    def initialize(self):
        for i in range(4):
            self.weights.append(rand.uniform(-1,1))

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
        for k in range(self.fit_range):
            # Randomize dataset order
            pm = np.random.permutation(dataset.index)
            sh_features = dataset.iloc[pm, :-1].to_numpy(dtype=float)
            sh_labels = dataset.iloc[pm, -1].to_numpy(dtype=int)

            total_hits = 0
            for i, feature in enumerate(sh_features):
                # Should be a better way of cycling all but the last layer
                inputs = feature
                for l in range(self.depth - 1): # self.depth = len(layers), -1 = all but last
                    current_layer = self.layers[l]
                    outputs = []
                    for perceptron in current_layer.nodes: # Grab each node in layer
                        print(f"Check in layer {l}")
                        output = perceptron.predict(inputs) # produce a prediction with the current node
                        outputs.append(output)  # Collect output in a list
                    inputs = np.array(outputs, dtype=float) # Prep inputs for next layer

                output_layer = self.layers[-1] # last layer of the list will be the output layer
                for perceptron in output_layer.nodes:   # Should only be one node, but scalability could be nice
                    print(f"Check in output...")
                    hit = perceptron.fit(inputs, sh_labels[i]) # this perceptron should FIT
                    if hit != sh_labels[i]:
                        print(f"Missed on input {feature} | Expected {sh_labels[i]}, got {perceptron.pred}")
                    total_hits += hit # if 

            # Calculate cycle accuracy
            accuracy = total_hits / len(dataset)
            print(f"Cycle {k + 1}/{self.fit_range} | Accuracy: {accuracy:.3f}")
            # Failed
            # if k == self.fit_range - 1:
            #     print("contingency!!!")
            #     self.debugAddLayer(1)
            #     self.fit(dataset)
            # elif accuracy == 1.0:
            #     print("Training complete with 100% accuracy!")
            #     break
                        
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
def debugGenerateModel(l, n):
    pList = []
    lList = []

    for i in range(l - 1):
        for k in range(n):
            weights = []
            for j in range(4):
                weights.append(rand.uniform(-1,1))
            p = NAND_perceptron(weights, rand.uniform(-1,1))
            pList.append(p)
        lList.append(layer(pList))
        pList = []

    # Output layer single perceptron
    weights = []
    for i in range(4):
        weights.append(rand.uniform(-1,1))
    p = [NAND_perceptron(weights, rand.uniform(-1,1))] # even one must be passed as a list because foresight
    lList.append(layer(p))
    m = model(lList)
    return m

    # layers = l
    # perceptrons = n
    # for i in range(layers):
    #     for i in range(perceptrons):
    #         weights = []
    #         for j in range(4):
    #             weights.append(rand.uniform(-1,1))
    #         p = NAND_perceptron(weights, rand.uniform(-1,1))
    #         pList.append(p)
    #     lList.append(layer(pList))
    #     pList = []
    # m = model(lList)
    # return m

def debugLoadModel(file):
    m = model([], file)
    return m

def main():
    # checking for errors in save() and load()
    m3 = debugGenerateModel(2, 4)
    #m3.save("NAND_MODEL.npz")
    ##print(m3.describe())
    #m3 = debugLoadModel("NAND_MODEL.npz")
    #print(m3.describe())
    
    #Checking for errors in fit()
    df = pd.read_csv("Xor_Dataset.csv")
    print(m3.fit(df))
    m3.save("Xor_MODEL.npz") # XOR could also not be learned

if __name__ == "__main__":
    main()