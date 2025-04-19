'''
What Are Neural Networks?
Neural networks are computational models that mimic the brain’s ability to process information and learn from data. They consist of interconnected neurons organized in layers, which transform input data into meaningful outputs through a process of weighted connections, activation functions, and learning algorithms.

Key Elements:
Neurons: Basic units that process inputs and produce outputs.
Connections: Links between neurons with adjustable weights that determine the influence of one neuron on another.
Layers: Groups of neurons organized into:
Input Layer: Receives raw data.
Hidden Layers: Perform intermediate computations.
Output Layer: Produces the final result.
Activation Functions: Mathematical functions that introduce non-linearity, deciding whether a neuron “fires.”
Learning: Adjusting weights based on data to minimize errors, typically through backpropagation and gradient descent.
Example:
Imagine a neural network designed to classify images of cats vs. dogs. The input layer takes pixel values, hidden layers extract features (e.g., edges, textures), and the output layer predicts “cat” or “dog.”


Understanding Neurons
A neuron is the fundamental building block of a neural network. It receives multiple inputs, processes them, and produces an output.

Components of a Neuron:
Inputs: Data from other neurons or external sources (e.g., pixel values in an image).
Weights: Parameters that scale the importance of each input.
Bias: A constant added to the weighted sum to shift the activation function.
Activation Function: Determines the output based on the weighted sum. Common functions include:
Sigmoid: Maps values to (0,1), useful for binary classification.
ReLU (Rectified Linear Unit): Outputs the input if positive; otherwise, zero. Fast and avoids vanishing gradients.
Tanh: Maps values to (-1,1), centered around zero.
Output: The result passed to the next layer or as the final prediction.

a neuron computes its output as follows:
1. Multiply each input by its corresponding weight.
2. Sum the weighted inputs and add the bias.
3. Apply the activation function to the result.
4. Pass the output to the next layer or as the final prediction.

# Example:
# Consider a simple neuron with three inputs:
# x1 = 0.5, x2 = 0.2, x3 = 0.8
# Weights: w1 = 0.4, w2 = 0.6, w3 = 0.9
# Bias: b = 0.1
# Activation Function: Sigmoid
#
# The neuron computes the output as follows:
# 1. Weighted Sum: z = (0.5 * 0.4) + (0.2 * 0.6) + (0.8 * 0.9) + 0.1 = 1.18
# 2. Sigmoid Activation: output = 1 / (1 + exp(-1.18)) ≈ 0.765
# 3. Output: The neuron outputs approximately 0.765, which can be interpreted as the probability of a certain class (e.g., cat vs. dog).
#
# This simple example illustrates how a neuron processes inputs, applies weights and biases, and uses an activation function to produce an output. In a neural network, multiple neurons work together in layers to learn complex patterns from data.



'''


import numpy as np

def relu(x):
    return np.maximum(0, x)

# Inputs, weights, and bias
inputs = np.array([2, 3])
weights = np.array([0.5, -0.2])
bias = 1

# Compute weighted sum
z = np.dot(weights, inputs) + bias

# Apply ReLU activation
output = relu(z)
print(f"Neuron output: {output}")  # Output: 1.4


'''
Neural Network Architecture
Neural networks are organized into layers:

Input Layer: Takes raw data (e.g., pixel values for images, word embeddings for text).
Hidden Layers: Perform feature extraction and transformations. More layers allow learning complex patterns.
Output Layer: Produces the final prediction (e.g., class probabilities, regression values).
Connections:
Neurons in one layer are connected to neurons in the next. Each connection has a weight, and the output of one neuron becomes the input to the next.

Example:
For a neural network with:

3 input neurons (e.g., RGB values of a pixel)
4 neurons in a hidden layer
2 output neurons (e.g., probabilities for “cat” or “dog”)
Each input neuron connects to all 4 hidden neurons (12 connections), and each hidden neuron connects to both output neurons (8 connections).


Activation Functions
Activation functions introduce non-linearity, enabling neural networks to solve complex problems. Without them, a network would behave like a linear model.

'''

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

# Test
x = np.array([1, -2, 3])
print(f"Sigmoid: {sigmoid(x)}")
print(f"ReLU: {relu(x)}")
print(f"Tanh: {tanh(x)}")
print(f"Softmax: {softmax(x)}")

'''
Training a Neural Network
Training involves adjusting weights to minimize the difference between predicted and actual outputs, using:

Loss Function: Measures error (e.g., Mean Squared Error for regression, Cross-Entropy for classification).
Gradient Descent: Optimizes weights by moving in the direction that reduces the loss.
Backpropagation: Computes gradients of the loss with respect to weights, propagating errors backward.
Learning Rate: Controls the step size of weight updates.
Epochs: Number of passes through the training data.
Batch Size: Number of samples processed before updating weights.


Challenges in Training
Overfitting: The model memorizes training data, failing to generalize. Solutions:
Regularization: Add penalties (e.g., L1, L2) to weights.
Dropout: Randomly disable neurons during training.
Data Augmentation: Generate new training examples (e.g., rotate images).
Underfitting: The model is too simple or undertrained. Solutions:
Increase model complexity (more layers/neurons).
Train for more epochs.
Generalization: Ensure the model performs well on unseen data using techniques like cross-validation.

Specialized Neural Networks
Convolutional Neural Networks (CNNs):
Designed for grid-like data (e.g., images).
Use convolutional layers to extract spatial features and pooling layers to reduce dimensions.
Example: Classifying images in ImageNet.
Recurrent Neural Networks (RNNs):
Designed for sequential data (e.g., time series, text).
Use recurrent layers to maintain memory of previous inputs.
Variants: LSTM (Long Short-Term Memory), GRU (Gated Recurrent Unit).
Transformers:
Use self-attention and multi-head attention for tasks like natural language processing.
Examples: BERT (bidirectional), GPT (generative).

Advanced Techniques
Batch Normalization: Normalizes layer inputs to stabilize training.
Transfer Learning: Use pre-trained models (e.g., BERT, ResNet) and fine-tune for specific tasks.
Learning Rate Decay: Reduce the learning rate over time for finer weight updates.
Early Stopping: Stop training when validation performance degrades.

Model Evaluation
Evaluate models using metrics like:

Accuracy: Proportion of correct predictions.
Precision: True positives / (True positives + False positives).
Recall: True positives / (True positives + False negatives).
F1 Score: Harmonic mean of precision and recall.
ROC Curve and AUC: Visualize and quantify classification performance.

Practical Considerations
Data Preprocessing: Clean and normalize data (e.g., scale features to [0,1]).
Feature Engineering: Create or select relevant features to improve performance.
Hyperparameter Tuning: Experiment with learning rate, batch size, number of layers, etc.
Cross-Validation: Split data into training/validation/test sets to assess generalization.

Neural networks are powerful tools for pattern recognition, built from interconnected neurons organized in layers. They learn through training, adjusting weights to minimize errors, and can be enhanced with techniques like regularization, dropout, and transfer learning. Specialized architectures like CNNs, RNNs, and Transformers tackle specific data types, while evaluation metrics ensure robust performance.

This step-by-step explanation, with examples and Python code, provides a comprehensive understanding of neural networks, from basic concepts to advanced applications. Let me know if you’d like to dive deeper into any specific aspect!


'''