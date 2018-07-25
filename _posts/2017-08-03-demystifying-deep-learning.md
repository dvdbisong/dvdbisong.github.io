---
layout: post
title: Demystifying Deep Learning
date: 2017-08-03 09:07:02 +00:00
permalink: /demystifying-deep-learning/
comments: true
excerpt: "Learning is a non-trivial task. How we learn deep representations as humans are high up there as one of the great enigmas of the world. What we consider trivial and to some others natural is a complex web of fine-grained and intricate processes that indeed have set us apart as unique creations in the universe both seen and unseen. In this post, I explain in simple terms the origins and promise of deep learning."
category: Writings
---
Learning is a non-trivial task. How we learn deep representations as humans are high up there as one of the great enigmas of the world. What we consider trivial and to some others natural is a complex web of fine-grained and intricate processes that indeed have set us apart as unique creations in the universe both seen and unseen.

A few shreds of evidence of the profound mysteries of the human mind include the innate ability to recognize faces at a millionth of a fraction of a second (probably must faster), the uncanny aptitude to learn and understand deep linguistic representations and form symbols for intelligent communications. Furthermore, the adept skill to compose and perform brilliant musical pieces to the envy of the angels and to write inspiring poems and poetry, and the otherworldly brilliance of classical arts and visual representations all stand as a testament to humanity&#8217;s beautiful natural intelligence.

### What is Deep Learning?

Deep learning is a learning scheme that approaches the learning problem by learning the underlying representations, or the deep representations, or the hierarchical representations of the dataset based on the problem domain or from the environment of interaction. That is why deep learning is also called representation learning.

Although the above definition is a mouthful, but fear not, for this post is poised to deconstruct this issue to its bare bones.

### The Problem of Representation

One of the greatest challenges of AI research is to get the computer to understand or to innately decompose structural representations of problems just like a human being would.

They are some problems that can more easily be represented by way of recipes. A recipe in computer science terms is a set of instructions provided to the computer to follow. That is, a computer can be explicitly programmed to perform a set of pre-defined logical steps to solve a particular problem or sets of problems.

For example, we can teach a computer to take a satellite to space, but it becomes inconceivably difficult to teach a computer to show empathy, to love, or even more simply, to write a poem or to understand verbal communication. Please note, to translate is not the same as to interpret because interpretation requires &#8220;deep&#8221; understanding of linguistic structures. While translation is hard enough, understanding is a different conundrum altogether.

In theoretical computer science, there are two sets of problems known as tractable (P) and intractable (NP) problems. Without delving into any technicality on this matter, NP problems or intractable problems are problems that a computer cannot entirely solve by following an instruction set, no matter the amount of time made available.

So when we talk about representation of problems, we refer to those suites of problems that are especially difficult to produce a recipe for, that is intractable problems. In other words, we are referring to problems that are inconceivable to describe, or cannot be defined as a set of logical instructions to be executed. These are the problems that deep learning sets out to solve.

### An Inspiration from the Brain

Scientists more often than not look to nature for inspiration when performing incredible feats. Notably, the birds inspired the airplane. In that vein, there is no better type to look to as an antitype of intelligence as the human brain.

Let&#8217;s talk briefly about the human brain. We will not even dare to venture too far; for one, that surpasses the ability of the writer, and two, centuries of scholars have devoted their lives to this cause and it seems like only the surface is scratched.

However, we would talk about what we know about the brain and how it forms the foundation of the algorithm that deep learning rests upon, which is called &#8220;Neural Networks.&#8221;

We can view the brain as a society of intelligent agents that are networked together and communicate by passing information via electrical signals from one agent to another. These agents are known as neurons. Our principal interest here is to have a glimpse of what neurons are, what are their components, and how they pass information around to create intelligence.

A neuron is an autonomous agent in the brain and is a central part of the nervous system. Neurons are responsible for receiving and transmitting information to other cells within the body based on external or internal stimuli. Neurons react by firing electrical impulses generated at the stimuli source to the brain and other cells for the appropriate response. The intricate and coordinated workings of neurons are central to human intelligence.

The following are the three most important components of neurons that are of primary interest to us:
* The Axon,
* The Dendrite, and
* The Synapse

<div class="imgcap">
<img src="../assets/demyst_dl/Depositphotos_4087535_original.jpg" style="width: 30%; height: 30%">
<div class="thecap">A Neuron</div>
</div>

The axon is a long tail connected to the nucleus of the neuron as seen in the figure above. The axon is responsible for transmitting electrical signals from the nucleus to other neuron cells via the axon terminals. The dendrite, on the other hand, receives information as electrical impulses from other neuron cells via synapses to the nucleus of a neuron cell.

We focused briefly on these three components as they are the biological inspiration that forms the kernel of the design and structure of an artificial neural network. There is much hope that if we can mimic this natural wonder of the brain from a science and engineering perspective, we can make giant strides in building a machine that can learn the complex hierarchical features of a problem and create a learning model using these features.

### Deep Learning Foundations: Artificial Neural Networks (ANN)

Building on the inspiration of the biological neuron, the artificial neural network (ANN) is a society of connectionist agents that learn and transfer information from one artificial neuron to the other. As data transfers between neurons, a hierarchy of representations or a hierarchy of features is learned. Hence the name deep representation learning or deep learning.

An artificial neural network is composed of:
* An input layer,
* Hidden layer(s), and
* An output layer

<div class="imgcap">
<img src="../assets/demyst_dl/neural-network.jpg" style="width: 50%; height: 50%">
<div class="thecap">Neural network architecture</div>
</div>

The input layer receives information as features from the dataset, some computation takes place, and data propagates to the hidden layer(s).

The hidden layer(s) is where the magic of deep learning occurs. The hidden layer(s) can consist of multiple neurons as shown in the image below. Each hidden network layer learns a more sophisticated set of features. The decision on the number of neurons in a layer (network width) and the number of hidden layers (network depth) which forms the network topology is a technical detail which is beyond the scope of this post. From a simplistic perspective, deep learning differs from a more traditional neural network primarily due to the presence of multiple hidden layers.

<div class="imgcap">
<img src="../assets/demyst_dl/Depositphotos_123753224_original.jpg" style="width: 50%; height: 50%">
<div class="thecap">A neural network network with 4 hidden layers</div>
</div>

The three top algorithmic extensions to neural networks that are revolutionizing deep learning today are:
* Dense Feed-Forward Networks,
* Convolutional Neural Networks (CNN), and
* Recurrent Neural Networks (RNN)

Finally, the output layer is where the learned representations come together to produce the desired output depending on the learning problem under consideration.

### The Tipping Point: Computational Power and Big Data
The successes of deep learning are mainly due to the surge in computational power and big data. The advent of Graphical Processing Units (GPUs) and Tensor Processing Units (TPUs) are upping the speed of processing that a few years back was simply out of reach. Moreover, wearable technologies, the internet of things, social media, and the world wide web are creating tons of data every day.

It has been well published that &#8220;the data we have generated as a civilization over the past ten years is more than all the information generated over the course of human existence bunched together&#8221;. Moreover, as we saw in a previous post on <a href="/understanding-machine-learning-executive-overview/" target="_blank" rel="noopener">"understanding machine learning</a>, there is no learning without data. The advent of really massive data, big data, has made deep learning very powerful and efficient. In truth, without big data, deep learning cannot produce good results.

<span class='bctt-click-to-tweet'><span class='bctt-ctt-text'><a href='https://twitter.com/intent/tweet?url=https://ekababisong.org/demystifying-deep-learning/&#038;text=In%20truth%2C%20without%20big%20data%2C%20deep%20learning%20cannot%20produce%20good%20results.&#038;related' target='_blank'>In truth, without big data, deep learning cannot produce good results. </a></span><a href='https://twitter.com/intent/tweet?url=https://ekababisong.org/demystifying-deep-learning/&#038;text=In%20truth%2C%20without%20big%20data%2C%20deep%20learning%20cannot%20produce%20good%20results.&#038;related' target='_blank' class='bctt-ctt-btn'>Click To Tweet</a></span> 

### Notable Application Areas

The following are some of the application areas of deep learning:
* Object detection,
* Image classification,
* Speech recognition,
* Document segmentation,
* Language translation,
* Audio to text conversion,
* Game playing (Atari, Go)
* Self-Driving cars,
* Social network analysis,
* Recommendation systems, and much more.

### Software Packages
The following software packages are great to get started using Deep Learning without being exposed to the underlying mathematical details.
* <a href="https://www.tensorflow.org/" target="_blank" rel="noopener">TensorFlow (Python)</a>
* <a href="https://deeplearning4j.org/" target="_blank" rel="noopener">DeepLearning4J (Java)</a>
* <a href="http://deeplearning.net/software/theano/" target="_blank" rel="noopener">Theano (Python)</a>
* <a href="http://cs.stanford.edu/people/karpathy/convnetjs/" target="_blank" rel="noopener">ConvNetJS (JavaScript)</a>
* <a href="https://www.rdocumentation.org/packages/deepnet/versions/0.2" target="_blank" rel="noopener">deepnet (R)</a>

Notable frameworks that make working with deep learning technologies even easier are:
* <a href="http://torch.ch/" target="_blank" rel="noopener">Torch (Lua)</a>
* <a href="http://caffe.berkeleyvision.org/" target="_blank" rel="noopener">Caffe</a>, and
* <a href="https://keras.io/" target="_blank" rel="noopener">Keras</a> (my personal favorite and package of choice)

Congratulations! Now we have an understanding of the philosophy and foundations of deep learning and can hold an informed discussion about it over coffee with a colleague or a friend.

### An Interesting Joke

I leave you with a joke that has been floating over social media. It paints a stark reality of the future impact of deep learning technology in our lives. The original author is unknown. So I quote with all due attribution respected.

> * Hello! Gordon&#8217;s pizza?
* No sir it is Google&#8217;s pizza
* So is it a wrong number?
* No sir, Google bought it
* OK. Take my order, please.
* Well, sir, you want the usual?
* The usual? You know me?
* According to our caller ID, in the last 12 times, you ordered pizza with cheeses, sausage, thick crust
* OK! This is it
* May I suggest to you this time ricotta, arugula with dry tomato?
* What? I hate vegetables
* Your cholesterol is not good
* How do you know? -through the subscribers guide We have the result of your blood tests for the last seven years
* Okay, but I do not want this pizza, I already take medicine -you have not taken medicine regularly, four months ago, you only purchased a box with 30 tablets at Drug sale Network
* I bought more from another drugstore
* It is not showing on your credit card
* I paid in cash
* But you did not withdraw that much cash according to your bank statement
* I have other sources of cash
* This is not showing as per you last Tax form unless you bought them from undeclared income source -WHAT THE HELL?
* Enough! I am sick of google, Facebook, twitter, Whats App. I am going to an Island without the internet, where there is no cell phone line and no one to spy on me &#8220;I understand sir, but you need to renew your passport as it expired five weeks ago&#8230;

&nbsp;

Please share your thoughts/ comments below. What do you think of deep learning as the future of artificial intelligence? Is is feasible? Is it possible? Are you a skeptic, or a believer? I look to hear from you.

Thanks for reading.