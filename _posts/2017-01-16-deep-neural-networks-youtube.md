---
layout: post
title: 'Deep Neural Networks for YouTube Recommendations: A Paper Review'
date: 2017-01-16 01:15:19 +00:00
category: Paper Reviews
comments: true
excerpt: "YouTube as of yet remains the world&#8217;s biggest video content delivery network, and each day is responsible for recommending videos to billions of users worldwide. This paper by a team of engineers at Google provides an overview of the data representation methods applied in developing a video recommendation system for YouTube and the performance gains observed using deep learning."
permalink: /deep-neural-networks-youtube/
---
### [Deep Neural Networks for YouTube Recommendations](http://dl.acm.org/citation.cfm?id=2959190)
Paul Covington, Jay Adams, Emre Sargin  
(RecSys &#8217;16 Proceedings of the 10th ACM Conference on Recommender Systems, Boston, Massachusetts, USA — September 15 &#8211; 19, 2016)  
Pages 191-198. 2016.  
Type: Proceedings  
**Date Reviewed:** January 15 2017

This paper provides an overview of the data representation methods applied in developing a video recommendation system for YouTube and describes the significant performance gains observed when using deep learning. YouTube as of yet remains the world&#8217;s biggest video content delivery network, and each day is responsible for recommending videos to billions of users worldwide. A YouTube recommender system presents the difficult challenge of scale (can our algorithms operate with astronomic data), freshness (how do we evaluate new videos as an observation unit in the model), and noise (how do we prevent the model from overfitting by memorizing noisy units).

YouTube recommender system is built on Google Brain, which was recently open sourced as TensorFlow. The dataset contains approximately one billion parameters with hundreds of billions of examples. The system is composed of two neural networks for managing the different tasks of “_candidate generation_” and “_ranking_”. Typically, both are components of the learning pipeline.

The _candidate generation_ stage takes events from the users YouTube activity history as inputs and collects a small subset (hundreds) of videos from a large corpus (i.e. a large set of video files) to produce broad personalization using the collaborative filtering technique. The _ranking_ stage prepares the filtered videos from the “collaborative generation” stage by allocating a score to each video based on a set of features that characterize the videos and the user to build a “best” recommendation list. The highest rated videos are shown to the user as recommendations.

The _candidate generation_ stage is set up as a multi-class classification problem that classifies a particular “video watch (wt)” at “time (t)” amidst millions of videos (i) classes from a corpus based on a user and context. An estimated nearest neighbor lookup is performed to create hundreds of candidate video recommendations from the corpus. The deep neural network utilizes a “softmax classifier” to distinguish the videos by learning user embeddings as a function of the user’s history and context. These embeddings are simply a mapping of sparse entities (e.g. individual videos and users) into a dense vector.

A procedure is used to sample out negative classes from the distribution, which is then adjusted using “importance weighing” to speed up and augment the effectiveness of training a model with millions of classes. Each sample then has its cross entropy loss function minimized for the true label and the sampled negative class.

The network architecture has a variable-length sequence of sparse video IDs which depicts users watch history mapped to a dense vector representation through the embeddings. The embeddings are trained using gradient descent back propagation updates rules with hidden-layers of Rectified Linear Units (ReLU) as activation functions.

A deep neural network allows different features from other sources to be added to the model. Binary and continuous features such as the user&#8217;s gender, logged-in state and age were normalized from real values to [0, 1] and added to the model. Adding age as a feature during model training ensures newly uploaded videos are included in the training.

Other proxy problems described to improve the model involved getting training examples from all YouTube watches, even from those found on other sites, this encouraged the model to take note of new uploads. Also, a fixed number of training examples are assigned per user to prevent a small number of active users from having an undue advantage on the loss function. Also, to avoid overfitting, the classifier ignores previous search labels and rather denotes search queries as an unordered bag of tokens so that the classifier is no longer aware of the original search label. The model also has an improved accuracy by adding features and depth to the softmax classifier.

The _Ranking_ stage uses “video impression data” to gauge predictions for a particular user interface. A deep neural network with a similar architecture described above is used to assign scores to each video impression using logistic regression. This list is then ordered and showed to the user. The _ranking_ process is frequently adjusted by performing live A/B testing of expected watch time per impression.

The user and video data used for training the _ranking_ model is manually transformed into useful features. The most significant signals were those that expressed a user’s previous interactions with the item (video) and another similar item, as they generalized well across different items. Also, information created from the _candidate generation_ state was added to the _ranking_ model as features as well as features describing the frequency of past video impressions.

Similar to _candidate generation_, embedding was also used to map sparse categorical features to dense forms suitable for neural networks. Also, continuous features were normalized by centering and scaling to improve convergence of the network model.