---
layout: page-ieee-ompi-workshop
title: 'Deep Learning'
permalink: ieee-ompi-workshop/deep_learning/
---

Table of contents:

- [The representation challenge](#the-representation-challenge)
- [An Inspiration from the Brain](#an-inspiration-from-the-brain)
- [The Neural Network Architecture](#the-neural-network-architecture)
- [Training the Network](#training-the-network)
- [Cost Function or Loss Function](#cost-function-or-loss-function)
- [The Backpropagation Algorithm](#the-backpropagation-algorithm)
- [Activation Functions](#activation-functions)
- [MultiLayer Perceptron (MLP) with Tensorflow Estimator API](#multilayer-perceptron-mlp-with-tensorflow-estimator-api)
  - [Importing the dataset](#importing-the-dataset)
  - [Prepare the dataset for modeling](#prepare-the-dataset-for-modeling)
  - [The MultiLayer Perceptron Model](#the-multilayer-perceptron-model)

Deep learning is a branch of learning models that extend the Neural Network algorithm. It enables the network algorithm to learn classifiers for complex problems such as computer vision and language modeling.

<a name="the-representation-challenge"></a>

## The representation challenge
Learning is a non-trivial task. How we learn deep representations as humans are high up there as one of the great enigmas of the world. What we consider trivial and to some others natural is a complex web of fine-grained and intricate processes that indeed have set us apart as unique creations in the universe both seen and unseen.

One of the greatest challenges of AI research is to get the computer to understand or to innately decompose structural representations of problems just like a human being would. Deep learning approaches this conudrum by learning the underlying representations, also called the deep representations or the hierarchical representations of the dataset based. That is why deep learning is also called representation learning.

<a name="an-inspiration-from-the-brain"></a>

## An Inspiration from the Brain
A neuron is an autonomous agent in the brain and is a central part of the nervous system. Neurons are responsible for receiving and transmitting information to other cells within the body based on external or internal stimuli. Neurons react by firing electrical impulses generated at the stimuli source to the brain and other cells for the appropriate response. The intricate and coordinated workings of neurons are central to human intelligence.

The following are the three most important components of neurons that are of primary interest to us:
- The Axon,
- The Dendrite, and
- The Synapse.

<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/brain.png" alt="A Neuron." height="90%" width="90%" />
</div>

Building on the inspiration of the biological neuron, the artificial neural network (ANN) is a society of connectionist agents that learn and transfer information from one artificial neuron to the other. As data transfers between neurons, a hierarchy of representations or a hierarchy of features is learned. Hence the name deep representation learning or deep learning.

<a name="neural-network-architecture"></a>

## The Neural Network Architecture
An artificial neural network is composed of:
- An input layer,
- Hidden layer(s), and
- An output layer.

<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/basic-NN.png" alt="Neural Network Architecture." height="50%" width="50%" />
</div>

The input layer receives information from the features of the dataset, some computation takes place, and data propagates to the hidden layer(s).
The hidden layer(s) are the workhorse of deep neural networks. They consist of multiple neuron modules where each hidden network layer learns a more sophisticated set of feature representations. The decision on the number of neurons in a layer (network width) and the number of hidden layers (network depth) which forms the network topology are all design choices.

<a name="training-the-network"></a>

## Training the Network
A weight is assigned to every neuron in the network. They control the activations as information moves from one neural layer to the next. The weights (also called parameters) are initially initialized as a random value but are later adjusted as the network begins to learn via the backpropagation algorithm.

Hence, the activations of the neurons in the next layer are determined by the sum of the neuron’s weight times the activations in the previous layer acted upon by a non-linear activation function. Every neuron layer also has a bias neuron that controls the weighted sum. This is similar to the bias term in the logistic regression model.

<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/information-flow.png" alt="Information flowing from a previous neural layer to a neuron in the next layer." height="80%" width="80%" />
</div>

<a name="cost-function-or-loss-function"></a>

## Cost Function or Loss Function
The quadratic cost which is also known as the mean squared error or the maximum likelihood estimate finds the sum of the difference between the estimated probability and the actual class label - used for regression problems. The cross-entropy cost function, also called the negative log-likelihood or binary cross-entropy, increases as the predicted probability estimates differ from the actual class label in a classification problem.

<a name="the-backpropagation-algorithm"></a>

## The Backpropagation Algorithm
Backpropagation is an algorithm for training the neural network to get better at improving its predicted outcomes by adjusting the weights of the network. The first time we run the feedforward algorithm, the activations at the output layer are most likely incorrect with a high error estimate or cost function. The goal of backpropagation is to repeatedly go back and adjust the weights of each preceding neural layer and perform the feedforward algorithm again until we minimize the error made by the network at the output layer.

<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/backpropagation.png" alt="Backpropagation." height="80%" width="80%" />
</div>

The cost function at the output layer is obtained by comparing the predicted output of the neural network with the actual outputs from the dataset. Gradient descent (earlier discussed) then computes the gradient of the costs using the weights of the neurons at each successive layer and updating the weights as it propagates back through the network.

<a name="activation-functions"></a>

## Activation Functions
Activation functions operate on the neurons affine transformations (which is nothing more than the sum of weights and their added bias) by passing it through a non-linear function to decide if that neuron should fire or propagate its information to the succeeding neural layers.

In other words, an activation function determines if a particular neuron has the information to result in a correct prediction at the output layer. Activation functions are analogous to how neurons communicate and transfer information in the brain, by firing when the activation goes beyond a particular threshold.

These activation functions are also called non-linearities because they inject non-linear capabilities to our network and can learn a mapping from inputs to output for a dataset whose fundamental structure is non-linear.

<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/activation-function.png" alt="Activation Function." height="80%" width="80%" />
</div>

They are various activations function we can use in neural networks, popular options include:
- Sigmoid
<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/sigmoid.png" alt="Sigmoid Function." height="50%" width="50%" />
</div>
- Hyperbolic Tangent (tanh)
<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/tanh.png" alt="Tanh Function." height="50%" width="50%" />
</div>
- Rectified Linear Unit (ReLU)
<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/ReLU.png" alt="ReLU Function." height="50%" width="50%" />
</div>
- Leaky ReLU
<div style="display: inline-block;width: 100%;">
<img src="/assets/ieee_ompi/Leaky-ReLU.png" alt="Lealy ReLU Function." height="50%" width="50%" />
</div>
</div>

<a id="mlp-with-tensorflow-estimator-api"></a>

## MultiLayer Perceptron (MLP) with Tensorflow Estimator API


```python
# import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```

<a id="importing-the-dataset"></a>

### Importing the dataset
The dataset used in this example is from the [Climate Model Simulation Crashes Data Set ](https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes). The dataset contains 540 observations and 21 variables ordered as follows:
- Column 1: Latin hypercube study ID (study 1 to study 3)  
- Column 2: simulation ID (run 1 to run 180)
- Columns 3-20: values of 18 climate model parameters scaled in the interval [0, 1]
- Column 21: simulation outcome (0 = failure, 1 = success)

The goal is to predict climate model simulation outcomes (column 21, fail or succeed) given scaled values of climate model input parameters (columns 3-20).


```python
# load data
data = np.loadtxt("data/climate-simulation/pop_failures.dat", skiprows=1)
```


```python
# preview data
data[:2,:]
```




    array([[1.        , 1.        , 0.85903621, 0.92782454, 0.25286562,
            0.29883831, 0.1705213 , 0.73593604, 0.42832543, 0.56794694,
            0.4743696 , 0.24567485, 0.10422587, 0.8690907 , 0.9975185 ,
            0.44862008, 0.30752179, 0.85831037, 0.79699724, 0.86989304,
            1.        ],
           [1.        , 2.        , 0.60604103, 0.45772836, 0.35944842,
            0.30695738, 0.84333077, 0.93485066, 0.44457249, 0.82801493,
            0.29661775, 0.6168699 , 0.97578558, 0.91434367, 0.84524714,
            0.86415187, 0.34671269, 0.35657342, 0.43844719, 0.51225614,
            1.        ]])




```python
# number of rows and columns
data.shape
```




    (540, 21)



<a id="prepare-the-dataset-for-modeling"></a>

### Prepare the dataset for modeling


```python
# separate features and target
X = data[:,:-1]
y = data[:,-1]
```


```python
# sample of features
X[:3,:]
```




    array([[1.        , 1.        , 0.85903621, 0.92782454, 0.25286562,
            0.29883831, 0.1705213 , 0.73593604, 0.42832543, 0.56794694,
            0.4743696 , 0.24567485, 0.10422587, 0.8690907 , 0.9975185 ,
            0.44862008, 0.30752179, 0.85831037, 0.79699724, 0.86989304],
           [1.        , 2.        , 0.60604103, 0.45772836, 0.35944842,
            0.30695738, 0.84333077, 0.93485066, 0.44457249, 0.82801493,
            0.29661775, 0.6168699 , 0.97578558, 0.91434367, 0.84524714,
            0.86415187, 0.34671269, 0.35657342, 0.43844719, 0.51225614],
           [1.        , 3.        , 0.99759978, 0.37323849, 0.51739936,
            0.50499255, 0.61890334, 0.60557082, 0.74622533, 0.19592829,
            0.81566694, 0.67935503, 0.80341308, 0.64399516, 0.71844113,
            0.92477507, 0.31537141, 0.25064237, 0.28563553, 0.36585796]])




```python
# targets
y[:10]
```




    array([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.])




```python
# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.15)
```

<a id="the-mlp-model"></a>

### The MultiLayer Perceptron Model


```python
import re
```


```python
# get column names
col_names = pd.read_csv("data/climate-simulation/pop_failures.dat", nrows=0)
```


```python
names_lst=list(col_names.columns)
```


```python
names_str=re.sub("\s+", ",", names_lst[0].strip())
```


```python
names=[x.strip() for x in names_str.split(',')]
```


```python
# column names
print(names)
print("\nlength:", len(names))
```

    ['Study', 'Run', 'vconst_corr', 'vconst_2', 'vconst_3', 'vconst_4', 'vconst_5', 'vconst_7', 'ah_corr', 'ah_bolus', 'slm_corr', 'efficiency_factor', 'tidal_mix_max', 'vertical_decay_scale', 'convect_corr', 'bckgrnd_vdc1', 'bckgrnd_vdc_ban', 'bckgrnd_vdc_eq', 'bckgrnd_vdc_psim', 'Prandtl', 'outcome']
    
    length: 21



```python
# create an input_fn
def input_fn(X, y, batch_size=30, training=True):
    # convert to dictionary
    X_dict={}
    for ind, name in enumerate(names[:-1]):
        X_dict[name]=X[:,ind]
    dataset = tf.data.Dataset.from_tensor_slices((X_dict, y))
    if training:
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()    
    return features, labels
```


```python
# use feature columns to define the attributes to the model
Study = tf.feature_column.numeric_column('Study')
Run = tf.feature_column.numeric_column('Run')
vconst_corr = tf.feature_column.numeric_column('vconst_corr')
vconst_2 = tf.feature_column.numeric_column('vconst_2')
vconst_3 = tf.feature_column.numeric_column('vconst_3')
vconst_4 = tf.feature_column.numeric_column('vconst_4')
vconst_5 = tf.feature_column.numeric_column('vconst_5')
vconst_7 = tf.feature_column.numeric_column('vconst_7')
ah_corr = tf.feature_column.numeric_column('ah_corr')
ah_bolus = tf.feature_column.numeric_column('ah_bolus')
slm_corr = tf.feature_column.numeric_column('slm_corr')
efficiency_factor = tf.feature_column.numeric_column('efficiency_factor')
tidal_mix_max = tf.feature_column.numeric_column('tidal_mix_max')
vertical_decay_scale = tf.feature_column.numeric_column('vertical_decay_scale')
convect_corr = tf.feature_column.numeric_column('convect_corr')
bckgrnd_vdc1 = tf.feature_column.numeric_column('bckgrnd_vdc1')
bckgrnd_vdc_ban = tf.feature_column.numeric_column('bckgrnd_vdc_ban')
bckgrnd_vdc_eq = tf.feature_column.numeric_column('bckgrnd_vdc_eq')
bckgrnd_vdc_psim = tf.feature_column.numeric_column('bckgrnd_vdc_psim')
Prandtl = tf.feature_column.numeric_column('Prandtl')
feature_columns = [Study, Run, vconst_corr, vconst_2, vconst_3, vconst_4, vconst_5, vconst_7, convect_corr,
                  ah_corr, ah_bolus, slm_corr, efficiency_factor, tidal_mix_max, vertical_decay_scale,
                  bckgrnd_vdc1, bckgrnd_vdc_ban, bckgrnd_vdc_eq, bckgrnd_vdc_psim, Prandtl]
```


```python
# instantiate a DNNClassifier Estimator
estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    optimizer='Adam',
    hidden_units=[1024, 512, 256],
    activation_fn=tf.nn.relu,
    n_classes=2
)
```

    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: /var/folders/gh/mqsbbqy55bb4z763xxgddw780000gn/T/tmp4mah8k5o
    INFO:tensorflow:Using config: {'_num_worker_replicas': 1, '_num_ps_replicas': 0, '_eval_distribute': None, '_tf_random_seed': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x1331449b0>, '_evaluation_master': '', '_global_id_in_cluster': 0, '_experimental_distribute': None, '_log_step_count_steps': 100, '_save_summary_steps': 100, '_service': None, '_train_distribute': None, '_is_chief': True, '_task_type': 'worker', '_keep_checkpoint_max': 5, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_protocol': None, '_task_id': 0, '_save_checkpoints_steps': None, '_device_fn': None, '_model_dir': '/var/folders/gh/mqsbbqy55bb4z763xxgddw780000gn/T/tmp4mah8k5o', '_master': '', '_save_checkpoints_secs': 600, '_keep_checkpoint_every_n_hours': 10000}



```python
# train model
estimator.train(input_fn=lambda:input_fn(X_train, y_train), steps=2000)
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 0 into /var/folders/gh/mqsbbqy55bb4z763xxgddw780000gn/T/tmp4mah8k5o/model.ckpt.
    INFO:tensorflow:loss = 12.81837, step = 1
    INFO:tensorflow:global_step/sec: 153.705
    INFO:tensorflow:loss = 7.897155, step = 101 (0.651 sec)
    INFO:tensorflow:global_step/sec: 222.691
    INFO:tensorflow:loss = 7.369482, step = 201 (0.449 sec)
    INFO:tensorflow:global_step/sec: 236.702
    INFO:tensorflow:loss = 7.373659, step = 301 (0.422 sec)
    INFO:tensorflow:global_step/sec: 226.485
    INFO:tensorflow:loss = 12.280007, step = 401 (0.442 sec)
    INFO:tensorflow:global_step/sec: 200.164
    INFO:tensorflow:loss = 9.850716, step = 501 (0.499 sec)
    INFO:tensorflow:global_step/sec: 215.473
    INFO:tensorflow:loss = 7.3498316, step = 601 (0.464 sec)
    INFO:tensorflow:global_step/sec: 227.987
    INFO:tensorflow:loss = 2.8503788, step = 701 (0.439 sec)
    INFO:tensorflow:global_step/sec: 213.14
    INFO:tensorflow:loss = 16.974874, step = 801 (0.469 sec)
    INFO:tensorflow:global_step/sec: 216.459
    INFO:tensorflow:loss = 7.389231, step = 901 (0.462 sec)
    INFO:tensorflow:global_step/sec: 210.224
    INFO:tensorflow:loss = 16.682652, step = 1001 (0.476 sec)
    INFO:tensorflow:global_step/sec: 202.052
    INFO:tensorflow:loss = 5.142111, step = 1101 (0.496 sec)
    INFO:tensorflow:global_step/sec: 233.348
    INFO:tensorflow:loss = 7.4850435, step = 1201 (0.427 sec)
    INFO:tensorflow:global_step/sec: 232.342
    INFO:tensorflow:loss = 2.9581068, step = 1301 (0.431 sec)
    INFO:tensorflow:global_step/sec: 229.923
    INFO:tensorflow:loss = 7.57767, step = 1401 (0.435 sec)
    INFO:tensorflow:global_step/sec: 229.363
    INFO:tensorflow:loss = 5.0159426, step = 1501 (0.436 sec)
    INFO:tensorflow:global_step/sec: 241.49
    INFO:tensorflow:loss = 4.823575, step = 1601 (0.414 sec)
    INFO:tensorflow:global_step/sec: 234.74
    INFO:tensorflow:loss = 9.993468, step = 1701 (0.426 sec)
    INFO:tensorflow:global_step/sec: 236.354
    INFO:tensorflow:loss = 7.405856, step = 1801 (0.424 sec)
    INFO:tensorflow:global_step/sec: 231.335
    INFO:tensorflow:loss = 7.3990912, step = 1901 (0.432 sec)
    INFO:tensorflow:Saving checkpoints for 2000 into /var/folders/gh/mqsbbqy55bb4z763xxgddw780000gn/T/tmp4mah8k5o/model.ckpt.
    INFO:tensorflow:Loss for final step: 5.0065947.

    <tensorflow.python.estimator.canned.dnn.DNNClassifier at 0x1331449e8>


```python
# evaluate model
metrics = estimator.evaluate(input_fn=lambda:input_fn(X_test, y_test, training=False))
```

    INFO:tensorflow:Calling model_fn.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Starting evaluation at 2019-02-01-20:28:43
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from /var/folders/gh/mqsbbqy55bb4z763xxgddw780000gn/T/tmp4mah8k5o/model.ckpt-2000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Finished evaluation at 2019-02-01-20:28:44
    INFO:tensorflow:Saving dict for global step 2000: accuracy = 0.90123457, accuracy_baseline = 0.90123457, auc = 0.49999994, auc_precision_recall = 0.9506173, average_loss = 0.3239939, global_step = 2000, label/mean = 0.90123457, loss = 8.747835, precision = 0.90123457, prediction/mean = 0.9173998, recall = 1.0
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 2000: /var/folders/gh/mqsbbqy55bb4z763xxgddw780000gn/T/tmp4mah8k5o/model.ckpt-2000



```python
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**metrics))
```

    
    Test set accuracy: 0.901
    