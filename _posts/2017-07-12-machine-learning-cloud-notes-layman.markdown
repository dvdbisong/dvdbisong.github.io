---
layout: post
title: "Machine Learning on the Cloud: Notes for the Layman"
date: 2017-07-12 00:28:30.000000000 -04:00
comments: true
excerpt: "Computational expenses have always been the bane of large-scale machine learning. In this post, I explain the fundamentals of Machine Learning on the cloud and the opportunities of unbridled computational horsepower made available by leveraging cloud infrastructures."
category: Technical Writings
permalink: /machine-learning-cloud-notes-layman/
---
### What is Machine Learning?
Machine learning is an assortment of tools and techniques for predicting outcomes and classifying events, based on a set of interactions between variables (also referred to as features or attributes) belonging to a particular data set. Today, we see the increased democratization of machine learning technology due to the proliferation of software packages that encapsulate the mathematical techniques that form the kernel of machine learning into application programming interfaces (APIs).

A new wave of amazing software/ analytic products making use of machine learning and in extension deep learning technology is the direct consequence of moving machine learning from the mathematical space into the software engineering domain by packaging and open-sourcing of software libraries.

### A Word on Deep Learning
Deep learning is a particular set or suite of machine learning algorithms called "Neural Networks" with major applications in computer vision (e.g. object recognition, image captioning), natural language processing, audio translation and other classification tasks such as document segmentation, to mention just a few. The following three extensions of neural networks form the core of deep learning:

1. Multi-layer Perceptrons (MLP)  
2. Convolutional Neural Networks (CNN), and  
3. Recurrent Neural Networks (RNN).

It is important to note that Deep Learning is the primary technology involved in self-driving cars.

### Enter Big Data
The concept of big data is continuously evolving. Previously Gigabytes of data was considered significant, but nowadays we are in the era where the order of magnitude of data sizes is exponentially increasing. The availability of big data spurned the gains and success of machine learning and deep learning. To put it simply there is no learning without data.

<span class='bctt-click-to-tweet'><span class='bctt-ctt-text'><a href='https://twitter.com/intent/tweet?url=https://ekababisong.org/machine-learning-cloud-notes-layman/&#038;text=there%20is%20no%20learning%20without%20data.&#038;related' target='_blank'>there is no learning without data. </a></span><a href='https://twitter.com/intent/tweet?url=https://ekababisong.org/machine-learning-cloud-notes-layman/&#038;text=there%20is%20no%20learning%20without%20data.&#038;related' target='_blank' class='bctt-ctt-btn'>Click To Tweet</a></span> 

### Computational Challenges of Learning
A significant challenge involved in training computers to learn using machine learning/ deep learning (ML/ DL) is computational power. Running a suite of an experiment on a decent CPU (e.g. a QuadCore i7, with 8GB RAM) can take upwards of 3 hours to days and even weeks for the algorithms to converge and produce a result set.

This computational lag is especially dire because getting a decent result requires several iterations of experiments either to tune the different parameters of the algorithm or to carry out various forms of feature engineering to achieve the desired classifier/ model that generalizes "optimally" to new examples.

The cost of on-premise high-end machines may be untenable for an aspiring ML/ DL practitioner, researcher, or amateur enthusiast. Moreover, the technical operations skill set required to build a cluster of commodity machines running Hadoop might be overwhelming and even sometimes a distraction for the casual user who just wants to delve into the nitty gritty of ML/ DL and explore.

### The Cloud to the Rescue
**The Cloud**. No, it is not the bright blue curtain over our heads with packs of thick white sheets arrayed in brilliant formations.

The cloud is a terminology that describes large sets of computers that are networked together in groups called data centers. These data centers are often distributed across multiple geographical locations. The size of a group, for example, is over 100, 000 sq ft (and those are the smaller sizes!).

Big companies like Google, Microsoft, Amazon & IBM, have large data centers that they are provisioning to the public (i.e. both enterprise and personal users) for use at a very reasonable cost.

Cloud technology/ infrastructure are allowing individuals to leverage the computing resources of big business for ML/ DL experimentation, design and development. For example, one can make use of Google's resources via <a href="https://cloud.google.com/" target="_blank" rel="noopener">Google Cloud Platform (GCP)</a> or Amazon's resources via <a href="https://aws.amazon.com/" target="_blank" rel="noopener">Amazon Web Services (AWS)</a>, or <a href="https://azure.microsoft.com/" target="_blank" rel="noopener">Microsoft's Azure</a> platform to run a suite of algorithms with multiple test grids for approx. 10 minutes, whereas such a series can take over 10 hours or more on a local device. Such is the power of the cloud.

Instead of running on a quad core machine for several hours, if not days, we can leverage thousands of cores to perform the same task for a short period and relinquish these resources after completing the job.

Another key advantage of using the cloud for ML/ DL is the cost effectiveness. Imagine the cost of purchasing a high-end computational machine, which may or may not be performing the actual job of high-end supercomputing all the time. Alternatively, even consider the "cost" (both time and otherwise) of setting up an on-premise Hadoop infrastructure, which can need constant technical operations attention, and there is the danger of spending more time doing operations than actual analytics processing.

In all the scenarios presented, the cloud comes to the rescue, where thousands of CPUs are available on-demand for turbocharged computing at a very affordable price. The principle is, use the resources needed to get the job done, and relinquish them after use.
