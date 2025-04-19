'''
1.1 What are Neural Networks?
Neural networks are a type of machine learning model inspired by the structure and function of the human brain. They consist of interconnected nodes (neurons) organized in layers. The defining characteristic of a deep neural network is having two or more hidden layers.
Example:
Imagine a neural network that predicts whether an email is spam or not. The inputs could be features like the email's content, sender, and subject line. The output would be a classification: spam or not spam.

1.2 A Brief History
Neural networks have a long history, dating back to the 1940s. They gained popularity in the 1960s with the introduction of backpropagation, but it wasn't until the 2010s that they started winning competitions and gaining widespread attention.
Example:
In the 2010s, neural networks started winning competitions like the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), which significantly boosted their popularity.

1.3 What is a Neural Network?
A neural network consists of layers of interconnected neurons. Each neuron performs a weighted sum of its inputs, adds a bias, and applies an activation function.

'''

inputs = [1, 2, 3, 2.5]  # Input values
weights = [0.2, 0.8, -0.5, 1]  # Weights for each input
bias = 2  # Bias value

# Calculate the neuron's output
output = (inputs[0] * weights[0] +
          inputs[1] * weights[1] +
          inputs[2] * weights[2] +
          inputs[3] * weights[3] +
          bias)

print(output)  # Output: 4.8


'''
Dense Layers
Dense layers are the most common type of layer in neural networks. 
Each neuron in a dense layer is connected to every neuron in the next layer.

Example:
Let's create a layer with 3 neurons:
'''

inputs = [1, 2, 3, 2.5]  # Input values
weights1 = [0.2, 0.8, -0.5, 1]  # Weights for neuron 1
weights2 = [0.5, -0.91, 0.26, -0.5]  # Weights for neuron 2
weights3 = [-0.26, -0.27, 0.17, 0.87]  # Weights for neuron 3
bias1 = 2  # Bias for neuron 1
bias2 = 3  # Bias for neuron 2
bias3 = 0.5  # Bias for neuron 3

# Calculate outputs for each neuron
output1 = (inputs[0] * weights1[0] +
           inputs[1] * weights1[1] +
           inputs[2] * weights1[2] +
           inputs[3] * weights1[3] +
           bias1)

output2 = (inputs[0] * weights2[0] +
           inputs[1] * weights2[1] +
           inputs[2] * weights2[2] +
           inputs[3] * weights2[3] +
           bias2)

output3 = (inputs[0] * weights3[0] +
           inputs[1] * weights3[1] +
           inputs[2] * weights3[2] +
           inputs[3] * weights3[3] +
           bias3)

layer_outputs = [output1, output2, output3]
print(layer_outputs)  # Output: [4.8, 1.21, 2.385]


'''
Activation Functions
Activation functions introduce non-linearity into the network, allowing it to learn more complex patterns.
Example:
Let's apply the ReLU activation function:

'''

def relu(x):
    return max(0, x)

# Apply ReLU to each output
activated_outputs = [relu(output) for output in layer_outputs]
print(activated_outputs)  # Output: [4.8, 1.21, 2.385]


'''
Forward Pass
The forward pass involves passing the input data through the network to get the output. 
This involves calculating the weighted sum of inputs, adding biases, and applying activation functions.

Example:
Let's perform a forward pass through a simple neural network with one hidden layer:
'''

# Input data
inputs = [1, 2, 3, 2.5]

# Weights and biases for the first layer
weights1 = [[0.2, 0.8, -0.5, 1],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
biases1 = [2, 3, 0.5]

# Calculate outputs for the first layer
layer1_outputs = [relu(sum(inputs[i] * weights1[j][i] for i in range(len(inputs))) + biases1[j]) for j in range(len(weights1))]

# Weights and biases for the second layer
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

# Calculate outputs for the second layer
layer2_outputs = [relu(sum(layer1_outputs[i] * weights2[j][i] for i in range(len(layer1_outputs))) + biases2[j]) for j in range(len(weights2))]

print(layer2_outputs)  # Output: [0.5031, -1.04185, -2.03875]

'''
Training Data
Training data is used to train the neural network. It consists of input data and corresponding target values (labels).
Example:
Let's generate some training data using the spiral_data function from the nnfs package:

'''

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Generate spiral data
X, y = spiral_data(samples=100, classes=3)

# Plot the data
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()