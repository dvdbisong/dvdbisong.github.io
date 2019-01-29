---
layout: page-ieee-ompi-workshop
title: 'Machine Learning: An Overview'
permalink: ieee-ompi-workshop/ml-overview/
---

Table of contents:

- [Why Machine Learning?](#why-machine-learning)
- [How do we assess learning?](#how-do-we-assess-learning)
- [Types of Learning](#types-of-learning)
- [Foundations of Machine Learning](#foundations-of-machine-learning)
- [A Formal Model for Machine Learning Theory](#a-formal-model-for-machine-learning-theory)

<span style="color:blue; font-weight:bold">Goal:</span> Teach the machine to learn based on the input they receive from the Environment.

> Tom Mitchell (1997) defines machine learning as the ability for a machine to gain expertise on a particular task based on experience.

<a id="why-machine-learning"></a>

## Why Machine Learning?
- <span style="color:blue; font-weight:bold">Solving complexity:</span> Excels in building computational engines to solve task that cannot possibly be explicitly programmed.
- Examples include natural language understanding, speech recognition and facial recognition.

<a id="how-do-we-assess-learning"></a>

## How do we assess learning?
Assume a teacher teaches a physics class for three months, and at the end of the lecture sessions, the teacher administers a test to ascertain if the student has learned.

Let us consider two different sub-plots:
1. The teacher tests the student with an exact word for word replica of questions that he used as examples while teaching.
2. The teacher evaluates the student with an entirely different but similar problem set based on the principles taught in class.
In which of the subplots can the teacher be confident that the student has learned?

**Norms of Learning**
1. <span style="color:blue; font-weight:bold">Memorization:</span> Memorization is the act of mastering and storing a pattern for future recollection. Therefore it is inaccurate to use training samples to carry out learning evaluation. In machine learning, this is also known as data snooping.
2. <span style="color:blue; font-weight:bold">Generalization:</span> The ability for the student to extrapolate using the principles taught in class to solve new examples is known as Generalization.
Hence, we can conclude that learning is the ability to generalize to new cases.

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

<div class="fig figcenter">
    <img src="/assets/ieee_ompi/supervised_learning.png" width="55%" height="55%">
    <div class="figcaption" style="text-align: center;">
        Figure 1: Supervised Learning.
    </div>
</div>

**Unsupervised Learning**
- No corresponding labels - no guidance.
- <span style="color:blue; font-weight:bold">Goal:</span> Computer attempts to determine data's unknown structure.
- <span style="color:blue; font-weight:bold">Scheme:</span> By "grouping" similar samples together adaptively.

<div class="fig figcenter">
    <img src="/assets/ieee_ompi/unsupervised_learning.png" width="55%" height="55%">
    <div class="figcaption" style="text-align: center;">
        Figure 2: Unsupervised Learning.
    </div>
</div>

**Reinforcement Learning**
- Reinforcement Learning: Agent interacts with an Environment.
- <span style="color:blue; font-weight:bold">Scheme:</span> A "feedback configuration".
- <span style="color:blue; font-weight:bold">Method:</span> Chooses an action from the set of actions.
- <span style="color:blue; font-weight:bold">Learning:</span> Based on the responses from the Environment.

<div class="fig figcenter">
    <img src="/assets/ieee_ompi/reinforcement_learning.png" width="55%" height="55%">
    <div class="figcaption" style="text-align: center;">
        Figure 3: Reinforcement Learning.
    </div>
</div>

<a id="types-of-learning"></a>

## Foundations of Machine Learning
The foundational disciplines that contribute to the field of machine learning are:

- Statistics,
- Mathematics,
- The Theory of Computation, and to a considerable extent
- Behavioral Psychology

The diagram below visually shows the interaction between these fields.

<div class="fig figcenter">
    <img src="/assets/ieee_ompi/foundations_ml.png" width="40%" height="40%">
    <div class="figcaption" style="text-align: center;">
        Figure 4: Foundations of Machine Learning.
    </div>
</div>


## A Formal Model for Machine Learning Theory

<span style="color:blue; font-weight:bold;">The learner’s input:</span>
- <span style="font-weight:bold;">Domain set:</span> An arbitrary set $$\mathcal{X}$$ that characterizes a set of objects that we want to label. The domain points are represented as a vector of features and are also called instances, with $$\mathcal{X}$$ the instance space.
- <span style="font-weight:bold;">Label set:</span> Let $$\mathcal{Y}$$ denote our set of possible labels. For example, let the label set be restricted to a two-element set, {0, 1} or {−1, +1}.
- <span style="font-weight:bold;">Training data:</span> $$S = ((x_1, y_1)...(x_m, y_m))$$ is a finite sequence of pairs in $$\mathcal{X} \times \mathcal{Y}$$ (i.e. a sequence of labeled domain points). The training data is the input that the learner has access to. $$S$$ is referred to as a training set.

<span style="color:blue; font-weight:bold;">The learner’s output:</span>
