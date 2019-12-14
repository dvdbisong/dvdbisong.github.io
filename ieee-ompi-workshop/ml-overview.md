---
layout: page-ieee-ompi-workshop
title: 'Machine Learning: An Overview'
permalink: ieee-ompi-workshop/ml-overview/
---

Table of contents:

- [Why Machine Learning?](#why-machine-learning)
- [Types of Learning](#types-of-learning)
- [Foundations of Machine Learning](#foundations-of-machine-learning)
- [A Formal Model for Machine Learning Theory](#a-formal-model-for-machine-learning-theory)
  - [Empirical Risk Minimization](#empirical-risk-minimization)
- [How do we assess learning?](#how-do-we-assess-learning)
  - [Norms of Learning](#norms-of-learning)
- [Training and Validation data](#training-and-validation-data)
- [The Bias/ Variance tradeoff](#the-bias-variance-tradeoff)
- [Evaluation metrics](#evaluation-metrics)
  - [Confusion Matrix](#confusion-matrix)
  - [Root Mean Squared Error (RMSE)](#root-mean-squared-error-rmse)
  - [Resampling](#resampling)

<span style="color:blue; font-weight:bold">Goal:</span> Teach the machine to learn based on the input they receive from the Environment.

> Tom Mitchell (1997) defines machine learning as the ability for a machine to gain expertise on a particular task based on experience.

<a id="why-machine-learning"></a>

## Why Machine Learning?
- <span style="color:blue; font-weight:bold">Solving complexity:</span> Excels in building computational engines to solve task that cannot possibly be explicitly programmed.
- Examples include natural language understanding, speech recognition and facial recognition.

<a id="types-of-learning"></a>

## Types of Learning
Machine Learning can be categorized into (at least) three components based on its approach.

The three predominant schemes of learning are:
- Supervised,
- Unsupervised, and
- Reinforcement Learning

**Supervised Learning**
- Each data point is associated with a label
- <span style="color:blue; font-weight:bold">Goal:</span> Teach the computer using this labeled data.
- <span style="color:blue; font-weight:bold">Learning:</span> The computer learns the patterns from data.
- <span style="color:blue; font-weight:bold">Inference:</span> Makes decisions about "unknown" samples.

<div class="fig">
<img src="/assets/ieee_ompi/supervised_learning.png" alt="Supervised Learning." height="70%" width="70%" />
</div>

**Unsupervised Learning**
- No corresponding labels - no guidance.
- <span style="color:blue; font-weight:bold">Goal:</span> Computer attempts to determine data's unknown structure.
- <span style="color:blue; font-weight:bold">Scheme:</span> By "grouping" similar samples together adaptively.

<div class="fig">
<img src="/assets/ieee_ompi/unsupervised_learning.png" alt="Unsupervised Learning." height="70%" width="70%" />
</div>

**Reinforcement Learning**
- Reinforcement Learning: Agent interacts with an Environment.
- <span style="color:blue; font-weight:bold">Scheme:</span> A "feedback configuration".
- <span style="color:blue; font-weight:bold">Method:</span> Chooses an action from the set of actions.
- <span style="color:blue; font-weight:bold">Learning:</span> Based on the responses from the Environment.

<div class="fig">
<img src="/assets/ieee_ompi/reinforcement_learning.png" alt="Reinforcement Learning." height="60%" width="60%" />
</div>

<a id="types-of-learning"></a>

## Foundations of Machine Learning
The foundational disciplines that contribute to the field of machine learning are:

- Statistics,
- Mathematics,
- The Theory of Computation, and to a considerable extent
- Behavioral Psychology

The diagram below visually shows the interaction between these fields.

<div class="fig">
<img src="/assets/ieee_ompi/foundations_ml.png" alt="Foundations of Machine Learning." height="50%" width="50%" />
</div>


<a id="a-formal-model-for-machine-learning-theory"></a>

## A Formal Model for Machine Learning Theory

<span style="color:blue; font-weight:bold;">The learner’s input:</span>
- <span style="font-weight:bold;">Domain set:</span> An arbitrary set $$\mathcal{X}$$ that characterizes a set of objects that we want to label. The domain points are represented as a vector of features and are also called instances, with $$\mathcal{X}$$ the instance space.
- <span style="font-weight:bold;">Label set:</span> Let $$\mathcal{Y}$$ denote our set of possible labels. For example, let the label set be restricted to a two-element set, {0, 1} or {−1, +1}.
- <span style="font-weight:bold;">Training data:</span> $$S = ((x_1, y_1)...(x_m, y_m))$$ is a finite sequence of pairs in $$\mathcal{X} \times \mathcal{Y}$$ (i.e. a sequence of labeled domain points). The training data is the input that the learner has access to. $$S$$ is referred to as a training set.

<span style="color:blue; font-weight:bold;">The learner’s output:</span>
The learner returns a prediction rule, $$h : \mathcal{X} \rightarrow \mathcal{Y}$$. This function is also called a predictor, a hypothesis, or a classifier. The predictor is used to predict the label of new domain points.
The notation $$A(S)$$ denotes the hypothesis that an algorithm, $$A$$, outputs, operating on the training set $$S$$.

<span style="color:blue; font-weight:bold;">The data generating model:</span>
The training set are generated by some probability distribution that is "unknown" to the learner. The probability distribution over $$\mathcal{X}$$ is denoted by $$\mathcal{D}$$.
In instances where the correct labelling is known, then each pair in the training data $$\mathcal{S}$$ is generated by first sampling a point $$x_i$$ according to $$\mathcal{D}$$ and then labeling it by $$f$$, where,  $$\;f : \mathcal{X} \rightarrow \mathcal{Y}$$, and that $$y_i = f(x_i)$$ for all $$i$$.

<span style="color:blue; font-weight:bold;">Measuring success:</span>
The error of a classifier is the probability that it does not predict the correct label on a random data point generated by the underlying distribution.
Therefore, the error of $$h$$ is the probability to draw a random instance $$x$$, according to the distribution $$\mathcal{D}$$, such that $$h(x)$$ does not equal $$f(x)$$.

The error of the prediction rule such $$\;h : \mathcal{X} \rightarrow \mathcal{Y}\;$$ is the probability of randomly choosing an example $$x$$ for which $$h(x) \neq f(x)$$:

\begin{equation}
L_{\mathcal{D},\;f} (h) \quad \equiv \quad \mathbb{P}_{x\sim D} [h(x) \neq f(x)] \quad \equiv \quad \mathcal{D}(\{x : h(x) \neq f(x)\}).
\end{equation}

where,
- The subscript $$(\mathcal{D}, f)$$: indicates that the error is measured with respect to the probability distribution \mathcal{D} and the correct labeling function $$f$$.
- $$L_{\mathcal{D},\;f} (h)$$: is the generalization error, the risk, or the true error of $$h$$.

<span style="color:blue; font-weight:bold;">Note:</span>
The learner is blind to the underlying distribution $$\mathcal{D}$$ over the world and to the labeling function $$f$$.

<a id="empirical-risk-minimization"></a>

### Empirical Risk Minimization

<div class="fig">
<img src="/assets/ieee_ompi/learning_model.png" alt="Learning Model."height="80%" width="80%"/>
</div>

The goal of the learning algorithm is to find the $$h_S$$ that minimizes the error with respect to the unknown $$\mathcal{D}$$ and $$f$$. Since the learner is unaware of $$\mathcal{D}$$ and $$f$$, the true error is not directly
available to the learner.

The training error (also called empirical error or empirical risk) is the error the classifier incurs over the training sample:

\begin{equation}
L_S(h) = \frac{|\{i \in [m] : h(x_i) \neq y_i\}|} {m}
\end{equation}

where,
- [m] is the total number of training examples.

The hypothesis $$h$$ that minimizes the error $$L_S(h)$$ is called <span style="color:blue; font-weight:bold;">Empirical Risk Minimization</span> or ERM.

<a id="how-do-we-assess-learning"></a>

## How do we assess learning?
Assume a teacher teaches a physics class for three months, and at the end of the lecture sessions, the teacher administers a test to ascertain if the student has learned.

Let us consider two different sub-plots:
1. The teacher tests the student with an exact word for word replica of questions that he used as examples while teaching.
2. The teacher evaluates the student with an entirely different but similar problem set based on the principles taught in class.
In which of the subplots can the teacher be confident that the student has learned?

### Norms of Learning
1. <span style="color:blue; font-weight:bold">Memorization:</span> Memorization is the act of mastering and storing a pattern for future recollection. Therefore it is inaccurate to use training samples to carry out learning evaluation. In machine learning, this is also known as data snooping.
2. <span style="color:blue; font-weight:bold">Generalization:</span> The ability for the student to extrapolate using the principles taught in class to solve new examples is known as Generalization.
Hence, we can conclude that learning is the ability to generalize to new cases.

<a id="training-and-validation-data"></a>

## Training and Validation data
Again, the goal of machine learning is to predict or classify outcomes on unseen observations. Hence, the dataset is partitioned into:
- <span style="color:blue; font-weight:bold;">Training set:</span> for training the model.
- <span style="color:blue; font-weight:bold;">Validation set:</span> for fine-tuning the model parameters.
- <span style="color:blue; font-weight:bold;">Test set:</span> for assessing the true model performance on unseen examples.

A common strategy is to split 60\% of the dataset for training, 20% for validation, and the remaining 20% for testing. This is commonly known as the 60/20/20 rule.

<div class="fig">
<img src="/assets/ieee_ompi/train-test-validation-set.png" alt="Dataset partitions."height="60%" width="60%"/>
</div>

<a id="the-bias-variance-tradeoff"></a>

## The Bias/ Variance tradeoff
- The bias/ variance tradeoff is critical for assessing the performance of the machine learning model.
- <span style="color:blue; font-weight:bold;">Bias</span> is when the learning algorithm fails oversimplifies the learning problem and fails to generalize to new examples.
- <span style="color:blue; font-weight:bold;">Variance</span> is when the model "closely" learns the irreducible error of the dataset. This leads to high variability in the presence of an unseen observation.

<div class="fig">
<img src="/assets/ieee_ompi/bias-and-variance.png" alt="Bias and variance." height="80%" width="80%"/>
</div>

The goal is to have a model that generalizes to new examples. Finding this middle ground is what is known as the bias/ variance tradeoff.

<div class="fig">
<img src="/assets/ieee_ompi/graph-overfit-underfit.png" alt="Left: Good fit. Center: Underfit (bias). Right: Overfit (variance)." />
</div>

<a id="evaluation-metrics"></a>

## Evaluation metrics
Evaluation metrics give us a way to measure the performance of our model. Let's see some common evaluation metrics.

<span style="color:blue; font-weight:bold;">Classification.</span>
- Confusion matrix.
- AUC-ROC (Area under ROC curve).

<span style="color:blue; font-weight:bold;">Regression</span>
- Root Mean Squared Error (RMSE).
- R-squared ( $$R^2$$ ).

<a id="confusion-matrix"></a>

### Confusion Matrix
Confusion matrix is one of the more popular metrics for assessing the performance of a classification supervised machine learning model.

<div class="fig">
<img src="/assets/ieee_ompi/confusion_matrix.png" alt="Confusion matrix." height="60%" width="60%"/>
</div>

<a id="root-mean-squared-error"></a>

### Root Mean Squared Error (RMSE)
- Root Mean Squared Error shortened for RMSE is an important evaluation metric in supervised machine learning for regression problems.
- The goal of RMSE is to calculate the error difference between the original targets and the predicted targets made by the learning algorithm.

\begin{equation}
RMSE=\sqrt{\displaystyle \frac{ \sum_{i=1}^{n} {(y_{i}-{\hat{y}}_{i})^{2}} } {n} }.
\end{equation}

<a id="resampling"></a>

### Resampling
Resampling is another important concept for evaluating the performance of a supervised learning algorithm and it involves selecting a subset of the available dataset, training on that data subset and using the reminder of the data to evaluate the trained model. Methods for resampling include:
- The validation set technique,
- The Leave-one-out cross-validation technique (LOOCV), and
- The k-fold cross-validation technique.

**Validation set**
<div class="fig">
<img src="/assets/ieee_ompi/validation_set.png" alt="Validation set" height="60%" width="60%"/>
</div>

**k-Fold validation**
<div class="fig">
<img src="/assets/ieee_ompi/k-fold.png" alt="k-fold validation" height="80%" width="80%"/>
</div>
