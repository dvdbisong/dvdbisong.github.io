---
layout: page-seminar
title: 'Deep Learning Explained: Artificial Neural Networks'
permalink: gcp-ml-seminar/deep-learning/
---

Table of contents:

- [The representation challenge](#the-representation-challenge)
- [An Inspiration from the Brain](#an-inspiration-from-the-brain)
- [The Neural Network Architecture](#the-neural-network-architecture)
- [Training the Network](#training-the-network)
- [Cost Function or Loss Function](#cost-function-or-loss-function)
- [The Backpropagation Algorithm](#the-backpropagation-algorithm)
- [Activation Functions](#activation-functions)

Deep learning is a branch of learning systems that is built on the Machine Learning algorithm called Neural Network. Deep learning extends the neural network in very interesting ways that enables the network algorithm to perform very well in building prediction models around complex problems such as computer vision and language modeling. A number of technologies that have resulted from this advancement include self-driving cars and automatic speech translation to mention just a few.

<a name="representation_challenge"></a>

### The representation challenge
Learning is a non-trivial task. How we learn deep representations as humans are high up there as one of the great enigmas of the world. What we consider trivial and to some others natural is a complex web of fine-grained and intricate processes that indeed have set us apart as unique creations in the universe both seen and unseen.

One of the greatest challenges of AI research is to get the computer to understand or to innately decompose structural representations of problems just like a human being would.

Deep learning approaches this conudrum by learning the underlying representations, also called the deep representations or the hierarchical representations of the dataset based. That is why deep learning is also called representation learning.

<a name="inspiration_from_the_brain"></a>

### An Inspiration from the Brain
Scientists more often than not look to nature for inspiration when performing incredible feats. Notably, the birds inspired the airplane. In that vein, there is no better type to look to as an antitype of intelligence as the human brain.

We can view the brain as a society of intelligent agents that are networked together and communicate by passing information via electrical signals from one agent to another. These agents are known as neurons. Our principal interest here is to have a glimpse of what neurons are, what are their components, and how they pass information around to create intelligence.

A neuron is an autonomous agent in the brain and is a central part of the nervous system. Neurons are responsible for receiving and transmitting information to other cells within the body based on external or internal stimuli. Neurons react by firing electrical impulses generated at the stimuli source to the brain and other cells for the appropriate response. The intricate and coordinated workings of neurons are central to human intelligence.

The following are the three most important components of neurons that are of primary interest to us:
 * The Axon,
 * The Dendrite, and
 * The Synapse.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/brain.png" width="90%" height="90%">
    <div class="figcaption" style="text-align: center;">
        Figure 1: A Neuron
    </div>
</div>

Building on the inspiration of the biological neuron, the artificial neural network (ANN) is a society of connectionist agents that learn and transfer information from one artificial neuron to the other. As data transfers between neurons, a hierarchy of representations or a hierarchy of features is learned. Hence the name deep representation learning or deep learning.

<a name="neural_network_architecture"></a>

### The Neural Network Architecture
An artificial neural network is composed of:
* An input layer,
* Hidden layer(s), and
* An output layer.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/basic-NN.png" width="70%" height="70%">
    <div class="figcaption" style="text-align: center;">
        Figure 2: Neural Network Architecture
    </div>
</div>

The input layer receives information from the features of the dataset, some computation takes place, and data propagates to the hidden layer(s).
The hidden layer(s) is where the workhorse of deep learning occurs. The hidden layer(s) can consist of multiple neuron modules as shown in the image above. Each hidden network layer learns a more sophisticated set of feature representations. The decision on the number of neurons in a layer (network width) and the number of hidden layers (network depth) which forms the network topology are all design choices.

<a name="training_the_network"></a>

### Training the Network
A weight is assigned to every neuron. This controls the activations in the neuron as the information of what the neural network is trying to learn moves from one layer of the network neurons to another. The weights (also called parameters) are initially initialized as a random value but are later adjusted as the network begins to learn.

So the activations of the neurons in the next layer are determined by the sum of the neuron’s weight times the activations in the previous layer acted upon by a non-linear activation function. Every neuron layer also has a bias neuron of  that controls the weighted sum. This is similar to the bias term in the logistic regression model.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/information-flow.png" width="80%" height="80%">
    <div class="figcaption" style="text-align: center;">
        Figure 3: Information flowing from a previous neural layer to a neuron in the next layer
    </div>
</div>

<a name="cost_function"></a>

### Cost Function or Loss Function
The quadratic cost which is also known as the mean squared error or the maximum likelihood estimate finds the sum of the difference between the estimated probability and the actual class label - used for regression problems. The cross-entropy cost function, also called the negative log-likelihood or binary cross-entropy, increases as the predicted probability estimates differ from the actual class label in a classification problem.

<a name="backpropagation"></a>

### The Backpropagation Algorithm
Backpropagation is the process by which we train the neural network to get better at improving its prediction accuracy. To train the neural network we need to find a mechanism for adjusting the weights of the network, this in turns affects the value of the activations within each neuron and consequently updates the value of the predicted output layer. The first time we run the feedforward algorithm, the activations at the output layer are most likely incorrect with a high error estimate or cost function.

The goal of backpropagation is to repeatedly go back and adjust the weights of each preceding neural layer and perform the feedforward algorithm again until we minimize the error made by the network at the output layer.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/backpropagation.png">
    <div class="figcaption" style="text-align: center;">
        Figure 4: Backpropagation
    </div>
</div>

The Backpropagation algorithm works by computing the cost function at the output layer by comparing the predicted output of the neural network with the actual outputs from the dataset. It then employs gradient descent (earlier discussed) to calculate the gradient of the cost function using the weights of the neurons at each successive layer and update the weights propagating back through the network.

<a name="activation_function"></a>

### Activation Functions
Activation functions act on the weighted sum in the neuron (which is nothing more than the weighted sum of weights and their added bias) by passing it through a non-linear function to decide if that neuron should fire or propagate its information or not to the succeeding neural layers.

In other words, the activation function determines if a particular neuron has the information to result in a correct prediction at the output layer for an observation in the training dataset. Activation functions are analogous to how neurons communicate and transfer information in the brain, by firing when the activation goes above a particular threshold value.

These activation functions are also called non-linearities because they inject non-linear capabilities to our network and can learn a mapping from inputs to output for a dataset whose fundamental structure is non-linear.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/activation-function.png" width="80%" height="80%">
    <div class="figcaption" style="text-align: center;">
        Figure 5: Activation Function
    </div>
</div>

They are various activations function we can use in a neural network, but the more popular functions include:
* Sigmoid,
* Hyperbolic Tangent (tanh),
* Rectified Linear Unit (ReLU),
* Leaky ReLU, and
* Maxout