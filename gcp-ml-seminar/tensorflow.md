---
layout: page-seminar
title: 'TensorFlow'
permalink: gcp-ml-seminar/tensorflow/
---

Table of contents:

- [The case for Computational Graphs](#the-case-for-computational-graphs)
- [Navigating through the TensorFlow API](#navigating-through-the-tensorflow-api)
    - [The Low-Level TensorFlow API](#the-low-level-tensorflow-api)
    - [The Mid-Level TensorFlow API](#the-mid-level-tensorflow-api)
    - [Layers](#layers)
    - [Datasets](#datasets)
    - [The High-Level TensorFlow API](#the-high-level-tensorflow-api)
    - [FeatureColumns](#featurecolumns)
    - [Estimator API](#estimator-api)
- [TensorBoard](#tensorboard)
- [The Low/Mid-Level API: Building Computational Graphs and Sessions](#the-lowmid-level-api-building-computational-graphs-and-sessions)
- [A Simple TensorFlow Programme](#a-simple-tensorflow-programme)
- [Working with Placeholders](#working-with-placeholders)
- [Session.run() vs. Tensor.eval()](#sessionrun-vs-tensoreval)
- [Working with Variables](#working-with-variables)
- [Variable scope](#variable-scope)
- [Linear Regression with TensorFlow](#linear-regression-with-tensorflow)
- [Classification with TensorFlow](#classification-with-tensorflow)
- [Multilayer Perceptron (MLP)](#multilayer-perceptron-mlp)
- [Visualizing with TensorBoard](#visualizing-with-tensorboard)
- [Running TensorFlow with GPUs](#running-tensorflow-with-gpus)
- [Convolutional Neural Networks](#convolutional-neural-networks)
- [Save and Restore TensorFlow Graph Variables](#save-and-restore-tensorflow-graph-variables)
- [Recurrent Neural Networks](#recurrent-neural-networks)
    - [Univariate Timeseries with RNN](#univariate-timeseries-with-rnn)
    - [Deep RNN](#deep-rnn)
    - [Multivariate Timeseries with RNN](#multivariate-timeseries-with-rnn)
- [Autoencoders](#autoencoders)
- [Building Efficient Input Pipelines with the Dataset API](#building-efficient-input-pipelines-with-the-dataset-api)
- [TensorFlow High-Level APIs: Using Estimators](#tensorflow-high-level-apis-using-estimators)
    - [Using the Pre-Made or Canned Estimator](#using-the-pre-made-or-canned-estimator)
    - [Building a Custom Estimator](#building-a-custom-estimator)
- [Eager Execution](#eager-execution)

TensorFlow is a specialized numerical computation library for Deep Learning. It is as of writing the preferred tool by numerous deep learning researchers and industry practitioners for developing deep learning models and architectures as well as for serving learned models into production servers and software products.

They are two fundamental principles behind working with TensorFlow, one is the idea of a Tensor, and the other is the idea of building and executing a Computational Graph. 

A Tensor is a mathematical name for an array with $$n$$ dimensions, so loosely speaking, a scalar is a Tensor without a dimension, an array or vector is a tensor with 1 dimension, and a matrix is a Tensor with 2 dimensions. An array with dimensions higher than 2 are simply tensors of $$n$$ dimensions. The dimension of a Tensor is also called the **rank** of a Tensor.

A Computational Graph is a tree-like representation of information as it flows from one computation unit, which is a Tensor or a Node in graph terminology to another via applying different forms of mathematical operations. The operations are the edges of the graph. It is from this representation that the package derives its name Tensor - Flow.

<a name="the-case-for-computational-graphs"></a>

### The case for Computational Graphs
The idea of modeling operations on data as a graph where information flows from one stage or process to another is a powerful abstract representation for visualizing what exactly needs to be achieved, and it also brings with it significant computational benefits.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/computational_graph.png">
    <div class="figcaption" style="text-align: center;">
        Figure 1: A computational graph for computing the roots of a Quadratic expression using the Quadratic formula.
    </div>
</div>

When data and operations are designed as a graph of events, it makes it possible to carve up the graph and execute different portions of it in parallel across multiple processors; this can be a CPU, GPU or TPU (more on this later). Also, the dataset or Tensors can be sharded and trained in a distributed fashion on millions of machines. This capability makes TensorFlow apt for building Large Scale machine learning products.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/parallel_graph.png">
    <div class="figcaption" style="text-align: center;">
        Figure 2: Parallel processing of a carved up graph.
    </div>
</div>

### Navigating through the TensorFlow API
Understanding the different levels of the TensorFlow API hierarchy is critical to working effectively with TensorFlow. The task of building a TensorFlow deep learning model can be addressed using different API levels, and each API level goes about the process in various forms. So a clear understanding of the API hierarchy will make it easier to work with TensorFlow, as well as make it easier to learn more from reading other TensorFlow implementations with clarity. The TensorFlow API hierarchy is primarily composed of three API levels. They are the high-level API, the mid-level API, and the low-level API. A diagrammatic representation of this is shown in Figure 3. 

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/tensorflow_api.png">
    <div class="figcaption" style="text-align: center;">
        Figure 3: TensorFlow API hierarchy.
    </div>
</div>

#### The Low-Level TensorFlow API
The low-level API gives the tools for building network graphs from the ground-up using mathematical operations. This level exposes you to the bare-bones of designing a Computational Graph of class `tf.Graph`, and executing them using the TensorFlow runtime, `tf.Session`. This API level affords the greatest level of flexibility to tweak and tune the model as desired. Moreover, the higher-level APIs implement low-level operations under the hood.

#### The Mid-Level TensorFlow API
TensorFlow provides a set of reusable packages for simplifying the process involved in creating Computational Graphs. Some example of these functions include the Layers `(tf.layers)`, Datasets `(tf.data)`, Metrics `(tf.metrics)` and Loss `(tf.losses)` packages.

#### Layers
The Layers package `(tf.layers)` provides a handy set of functions to simplify the construction of layers in a neural network architecture. For example, consider the convolutional network architecture in Figure 4, and how the layers API simplifies the creation of the network layers.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/layers_api.png">
    <div class="figcaption" style="text-align: center;">
        Figure 4: Using the Layers API to simplify creating the layers of a Neural Network.
    </div>
</div>

#### Datasets
The Datasets package `(tf.data)` provides a convenient set of high-level functions for creating complex dataset input pipelines. The goal of the Dataset package is to have a fast, flexible and easy to use interface for fetching data from various data sources, performing data transform operations on them before passing them as inputs to the learning model. The Dataset API provides a more efficient means of fetching records from a dataset. The major classes of the Dataset API are illustrated in Figure 5 below:

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/datasets_api.png">
    <div class="figcaption" style="text-align: center;">
        Figure 5: Datasets API class hierarchy.
    </div>
</div>

From the illustration in Figure 5, the subclasses perform the following functions:
- TextLineDataset: This class is used for reading lines from text files.
- TFRecordDataset: This class is responsible for reading records from TFRecord files. A TFRecord file is a TensorFlow binary storage format. It is faster and easier to work with data stored as TFRecord files as opposed to raw data files. Working with TFRecord also makes the data input pipeline more easily aligned for applying vital transformations such as shuffling and returning data in batches.
- FixedLengthRecordDataset: This class is responsible for reading records of fixed sizes from binary files.
- Iterator: The Iterator class is an object of the Dataset class, and it provides an interface for accessing records one at a time.

#### The High-Level TensorFlow API
The High-level API provides a simplified API calls that encapsulate lots of the details that are typically involved in creating a deep learning TensorFlow model. These high-level abstractions make it easier to develop powerful deep learning models quickly, with fewer lines of code. However, it is not so flexible when it boils down to tweaking some nitty-gritty parameter of the model. But it is the preferred go to to get an end-to-end modeling pipeline up and running.

#### FeatureColumns
TensorFlow offers a high-level API called FeatureColumns `tf.feature_column` for describing the features of the dataset that will be fed into an Estimator for training and validation. This makes easy the preparation of data for modeling, such as the conversion of categorical features of the dataset into a one-hot encoded vector. The canned estimators which we'll mention in the next sub-section have an attribute called `feature_columns`, which will receive the features of the dataset, wrapped with the `tf.feature_column` API functions.

The `feature_column` API is broadly divided into two categories, they are the categorical and dense columns, and together they consist of nine function calls. The categories and subsequent functions  are illustrated in Figure 6.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/feature_column_api.png">
    <div class="figcaption" style="text-align: center;">
        Figure 6: Functions calls of the Feature Column API.
    </div>
</div>

Let's go through each API function briefly.

<table id="my_table">
<thead>
<tr class="header">
<th>Function name</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td markdown="span">Numeric column `tf.feature_column.numeric_column()`</td>
<td markdown="span">This is a high-level wrapper for numeric features in the dataset.</td>
</tr>
<tr>
<td markdown="span">Indicator column `tf.feature_column.indicator_column()`</td>
<td markdown="span">The indicator column takes each as input a categorical column and transforms it into a one-hot encoded vector.</td>
</tr>
<tr>
<td markdown="span">Embedding column `tf.feature_column.embedding_column()`</td>
<td markdown="span">The embedding column function transforms a categorical column with multiple levels or classes into a lower-dimensional numeric representation that captures the relationships between the categories. Using embeddings mitigates the problem of a large sparse vector (an array with mostly zeros) created via one-hot encoding for a dataset feature with lots of different classes.</td>
</tr>
<tr>
<td markdown="span">Categorical column with identity `tf.feature_column.categorical_ column_with_identity()`</td>
<td markdown="span">This function creates a one-hot encoded output of a categorical column containing identities e.g. ['0', '1', '2', '3'].</td>
</tr>
<tr>
<td markdown="span">Categorical column with vocabulary list `tf.feature_column.categorical_ column_with_vocabulary_list()`</td>
<td markdown="span">This function creates a one-hot encoded output of a categorical column with strings. It maps each string to an integer based on a vocabulary list. However, if the vocabulary list is long, it is best to create a file containing the vocabulary and use the function `tf.feature_column.categorical_ column_with_vocabulary_file()`</td>
</tr>
<tr>
<td markdown="span">Categorical column with hash bucket `tf.feature_column.categorical_ column_with_hash_buckets()`</td>
<td markdown="span">This function specifies the number of categories by using the hash of the inputs. It is used when it is not possible to create a vocabulary for the number of categories due to memory considerations.</td>
</tr>
<tr>
<td markdown="span">Crossed column `tf.feature_columns.crossed_column()`</td>
<td markdown="span">The function gives the ability to combine multiple inputs features into a single input feature.</td>
</tr>
<tr>
<td markdown="span">Bucketized column `tf.feature_column.bucketized_column()`</td>
<td markdown="span">The function splits a column of numerical inputs into buckets to form new classes based on a specified set of numerical ranges.</td>
</tr>
</tbody>
</table>

#### Estimator API
The Estimator API is a high-level TensorFlow functionality that is aimed at reducing the complexity involved in building machine learning models by exposing methods that abstract common models and processes. There are two ways of working with Estimators and they include:
- **Using the Pre-made Estimators:** The pre-made Estimators, are black-box models made available by the TensorFlow team for building common machine learning/ deep learning architectures such as Linear Regression/ Classification, Random Forests Regression/ Classification and Deep Neural Networks for regression and classification. An illustration of the pre-made Estimators as subclasses of the Estimator class is shown in Figure 7.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/estimator_api.png">
    <div class="figcaption" style="text-align: center;">
        Figure 7: Estimator class API hierarchy.
    </div>
</div>

- **Creating a Custom Estimator:** It is also possible to use the low-level TensorFlow methods to create a custom black-box model for easy reusability. To do this, you must put your code in a method called the `model_fn`. The model function will include code that defines operations such as the labels or predictions, loss function, the training operations and the operations for evaluation.

The Estimator class exposes four major methods, namely, the `fit()`, `evaluate()`, `predict()` and `export_savedmodel()` methods. The `fit()` method is called to train the data by running a loop of training operations. The `evaluate()` method is called to evaluate the model performance by looping through a set of evaluation operations. The `predict()` method uses the trained model to make predictions, while the method `export_savedmodel()` is used for exporting the trained model to a specified directory. For both the pre-made and the custom Estimators, we must write a method to build the data input pipeline into the model. This pipeline is built for both the training and evaluation data inputs. This is further illustrated in Figure 8.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/input_pipelines_estimators.png">
    <div class="figcaption" style="text-align: center;">
        Figure 8: Estimator data input pipeline.
    </div>
</div>

### TensorBoard
TensorBoard is an interactive visualization tool that comes bundled with TensorFlow. The goal of TensorBoard is to gain a visual insight into how the computational graph is constructed and executed. This information provides greater visibility for understanding, optimizing and debugging deep learning models.

To use TensorBoard, summaries of operations (also called summary ops) within the Graph are exported to a location on disk using the helper class `tf.summaries`. TensorBoard visualizations can be accessed with by pointing the `--logdir` attribute of the `tensorboard` command to the log path or by running the command `ml.TensorBoard.start` from the Datalab cell using the `google.datalab.ml` package.

```bash
# running tensorboard via the terminal
tensorboard --logdir=/path/to/logs
```

```python
# running tensorboard from Datalab cell
tensorboard_pid = ml.TensorBoard.start('/path/to/logs')

# After use, close the TensorBoard instance by running:
ml.TensorBoard.stop(tensorboard_pid)
```

TensorBoard has a variety of visualization dashboard, namely the:
- Scalar Dashboard: This dashboard captures metrics that change with time, such as the loss of a model or other model evaluation metrics such as accuracy, precision, recall, f1, etc. 
- Histogram Dashboard: This dashboard shows the histogram distribution for a Tensor as it has changed over time.
- Distribution Dashboard: This dashboard is similar to the Histogram dashboard. However, it displays the histogram as a distribution.
- Graph Explorer: This dashboard gives a graphical overview of the TensorFlow Computational Graph and how information flows from one node to the other. This dashboard provides invaluable insights into the network architecture.
- Image Dashboard: This dashboard displays images saved using the method `tf.summary.image`.
- Audio Dashboard: This dashboard provides audio clips saved using the method `tf.summary.audio`.
- Embedding Projector: The dashboard makes it easy to visualize high-dimensional datasets after they have been transformed using **Embeddings**. The visualization uses Principal Component Analysis (PCA) and another technique called t-distributed Stochastic Neighbor Embedding (t-SNE). Embedding is a technique for capturing the latent variables in a high-dimensional dataset by converting the data units into real numbers that capture their relationship. This technique is broadly similar to how PCA reduces data dimensionality. Embeddings are also useful for converting sparse matrices (matrices made up of mostly zeros) into a dense representation.
- Text Dashboard: This dashboard is for displaying textual information.

### The Low/Mid-Level API: Building Computational Graphs and Sessions
The first thing to do before working with TensorFlow is to import the package by running:
```python
import tensorflow as tf
```

When working with TensorFlow at the low-level, the fundamental principle to keep in mind is that a computational graph is first created of class `tf.Graph` which contains a set of data of class `tf.Tensor` and operations of class `tf.Operations`. As earlier mentioned, an operation is a unit of mathematical computation or data transformation that is carried out on data of $$n$$ dimension, which is also called a Tensor. So in a computational graph, the Tensors are the **Nodes** and the Operations are the **Edges**.

When building a low-level TensorFlow programme, a default Graph is automatically created. As we will later see, calling methods in the TensorFlow API adds operations and tensors to the default Graph. At this point, it is important to note that no computation is performed when the Graph is being constructed using the methods from the TensorFlow API. A Graph is only executed when a Session of class `tf.Session` has been created to initialize the Graph variables and execute the Graph.

High-Level APIs like the Estimators and Keras API handle the construction and execution of the Computational Graph under the hood.

### A Simple TensorFlow Programme
Let's start by building a simple computational graph. In this programme, we will build a graph to find the roots of the quadratic expression $$x^2 + 3x – 4 = 0$$.
```python
a = tf.constant(1, name='a')
b = tf.constant(3, name='b')
c = tf.constant(-4, name='c')

print(a)
print(b)
print(c)

'Output':
Tensor("a_3:0", shape=(), dtype=float32)
Tensor("b_3:0", shape=(), dtype=float32)
Tensor("c_3:0", shape=(), dtype=float32)
```

`tf.constant()` is a Tensor for storing a constant type. Other Tensor types include `tf.placeholder()` for passing data into the computational graph and `tf.Variable()` for storing values that changes or updates during the course of running the programme.

Now let's calculate the roots of the expression.
```python
x1 = tf.divide(-b + tf.sqrt(tf.pow(b,tf.constant(2, dtype=tf.float32)) - (4*a*c)),
               tf.pow(tf.constant(2, dtype=tf.float32), a))
x2 = tf.divide(-b - tf.sqrt(tf.pow(b, tf.constant(2, dtype=tf.float32)) - (4*a*c)), 
               tf.pow(tf.constant(2, dtype=tf.float32), a))

roots = (x1, x2)

print(roots)
'Output':
(<tf.Tensor 'truediv_1:0' shape=() dtype=float32>, 
                <tf.Tensor 'truediv_2:0' shape=() dtype=float32>)
```

Note that the variables `x1` and `x2` do not carry out any computation, it only describes an operation that is added to the graph. Hence the print-out of the variable `roots`, does not contain the roots of the equation but rather an indication that it is a Tensor of some shape.

A `Session` object must be created to evaluate the computational graph. A Session object will initialize the variables and execute the operations. There are a couple of ways to work with sessions, let's go through them.

**Option 1:** Directly creating a `tf.Session` object.
```python
sess = tf.Session()         # create Session object
sess.run(roots)
'Output': (1.0, -4.0)
sess.close()                # remember to close the Session object
```

**Option 2:** Using the `with` method in Python to automatically close the `Session` object.
```python
# create Session object
with tf.Session() as sess:
     result = sess.run(roots)
     print(result)
'Output': (1.0, -4.0)
```

Alternatively,
```python
with tf.Session() as sess:
     root1 = x1.eval()
     root2 = x2.eval()
     result = [root1, root2]
     print(result)
'Output': (1.0, -4.0)
```

**Option 3:** Using an Interactive Session.
```python
# create interactive session object
sess = tf.InteractiveSession()
result = (x1.eval(), x2.eval())
print(result)
'Output': (1.0, -4.0)
sess.close()    # remember to close the Session object
```

Alternatively,
```python
sess = tf.InteractiveSession()
result = sess.run(roots)
print(result)
'Output': (1.0, -4.0)
sess.close()
```

### Working with Placeholders
Let's run another simple example, but this time, we'll be working with placeholders `tf.placeholder`. In this example, we will build a computational graph to find the solution to the Pythagorean Theorem, $$a^2 + b^2 = c^2$$ given that $$a = 7$$ and $$b = 24.$$

```python
# create placeholder
a = tf.placeholder(dtype=tf.float32, shape=None)
b = tf.placeholder(dtype=tf.float32, shape=None)

c = tf.sqrt(tf.pow(a,2) + tf.pow(b,2))
```

The `placeholder` Tensor type has a `dtype` and `shape` attribute for defining the datatype and the shape of the data it will later receive when the computational graph is being executed. When `shape` is set to `None` it implies that the dimensions of the data are unknown.

Now let's execute the graph by creating a Session.
```python
with tf.Session() as sess:
     result = sess.run(c, feed_dict={a: 7, b: 24})
     print(result)
'Output': 25.0
```

Observe that the `run()` method Session object referenced as `sess.run()` has an attribute called `feed_dict`. This attribute is responsible for passing data into the placeholder before executing the graph.

### Session.run() vs. Tensor.eval()
The method `Tensor.eval()` is similar to running `Session.run(Tensor)` with the distinct difference being that a Session run can compute the result of more than one Tensor at the same time. Consider the following example:
```python
import tensorflow as tf

# clear graph (if any) before running
tf.reset_default_graph()

a = tf.constant(2)
b = tf.constant(4)
# multiplication
c = a * b

x = tf.constant(3)
y = tf.constant(5)
# powers
z = x**y

# execute graph for the Tensor c and z
with tf.Session() as sess:
     result = sess.run([c, z])
     print(result)
'Output': [8, 243]

# using Tensor.eval()
with tf.Session() as sess:
    print(c.eval())
    print(z.eval())
'Output':
8
243
```

### Working with Variables
The `Variable` Tensor is used to store persistent data that can be shared and updated by operations in a computational graph when a Session is executed. A major difference between the `tf.Variable`  Tensor and other Tensor objects is that the data stored in `tf.Variable` is accessible across multiple sessions. So different graphs within the same application can share and modify a value in a `tf.Variable` tensor.

Before working with `tf.Variables`, here are a few key points to have at hand:
- A Variable is created either by using `tf.Variable()` or `tf.get_variable()`. 
- Use the method `tf.asssign()` to assign a new value to a Variable, and the methods `tf.assign_add()` and `tf.assign_sub()` to add and subtract a value from the Variable.
- A `tf.Variable` Tensor must be initialized before it can be manipulated with a Session. There are two ways of initializing a `tf.Variable` object. One way is all at once, before running any operation in a Session by calling the method `tf.global_variables_initializer()`. The other way is to independently initialize each Variable by calling `.initialized()` on a `tf.Variable` object.

All the Variables within a TensorFlow programme can be accessed through a named list of Tensors called Collections. When a `tf.Variable` object is created, it is placed automatically into two default collections.
- The `tf.GraphKeys.GLOBAL_VARIABLES` collection stores all variables so that they can be shared across multiple devices (i.e., CPU, GPU, etc.).
- The `tf.GraphKeys.TRAINABLE_VARIABLES` collection stores variables that will be used to calculate gradients trained by an optimization algorithm.

If a Variable is not to be used for training, it can be manually added to the `LOCAL_VARIABLES` member of the `tf.GraphKeys` class (i.e., `tf.GraphKeys.LOCAL_VARIABLES`) using the `collections` attribute of either `tf.Variable()` or `tf.get_variable()` methods for creating a Variable. Another option is to set the `trainable` attribute of either of the Variable creation methods to `False`.

Let's see examples of creating and initializing and working with a `tf.Variable` Tensor. In this example, we will write a TensorFlow programme to calculate the running average of a sequence of $$n$$ numbers.

```python
import tensorflow as tf

# clear graph (if any) before running
tf.reset_default_graph()

data = [2, 4, 6, 8, 10, 12, 14, \
         16, 18, 20, 22, 24, 26, \
         28, 30, 32, 34, 36, 38, \
         40, 42, 44, 46, 48, 50]
running_sum = tf.Variable(0, dtype=tf.float32)
running_average = tf.Variable(0, dtype=tf.float32)

with tf.Session() as sess:
    # initialize all variables
    sess.run(tf.global_variables_initializer())
    
    for index, value in enumerate(data):
        sum_op = tf.assign_add(running_sum, value)
        average_op = tf.assign(running_average, sum_op/(index+1))
        print(sess.run(average_op))
    print('Running sum: {}'.format(running_sum.eval()))
    print('Running average: {}'.format(running_average.eval()))

'Output':
Running sum: 650.0
Running average: 26.0
```

### Variable scope
Variable scope is a technique for regulating how variables are reused in a TensorFlow program especially when working with functions that implicitly create and manipulate variables. Another advantage of variable scopes is that it assigns name scopes to Variables, thereby making it easier for code readability and debugging. Variable scope is implemented using the class `tf.variable_scope`. In the following example, the variables for computing a Pythagorean triple is wrapped into a function for which we make repeated calls. Using variable scope makes it clear that a new set of variables are to be created else the code will result in an error after the first call to the function because the variables already exist. Also very importantly, calling the method `tf.reset_default_graph()` clears the graph, and hence the variable scopes. This is important if you want to run the same script over again.

```python
# clear graph (if any) before running
tf.reset_default_graph()

# function to calculate a pythagorean triple
def pythagorean_triple(a, b):
     a = tf.get_variable(name = "a", initializer = tf.constant(a, dtype=tf.float32))
     b = tf.get_variable(name = "b", initializer = tf.constant(b, dtype=tf.float32))
     return tf.sqrt(tf.pow(a,2) + tf.pow(b,2))

# variables here will be named "example1/a", "example1/b"
with tf.variable_scope("example1"):
     c_1 = pythagorean_triple(3, 4)
# variables here will be named "example2/a", "example2/b"
with tf.variable_scope("example2"):
     c_2 = pythagorean_triple(5, 12)

# execute in Session
with tf.Session() as sess:
     # initialize all variables
     tf.global_variables_initializer().run()

     print('Triple of 3, 4 ->{}'.format(sess.run(c_1)))
     print('Triple of 5, 12 ->{}'.format(sess.run(c_2)))

'Output':
Triple of 3, 4 ->5.0
Triple of 5, 12 ->13.0
```

Without using variable scopes, the above program will result in an error. An exmaple is illustrated below.
```python
# function to calculate a pythagorean triple
def pythagorean_triple(a, b):
     a = tf.get_variable(name = "a", initializer = tf.constant(a, dtype=tf.float32))
     b = tf.get_variable(name = "b", initializer = tf.constant(b, dtype=tf.float32))
     return tf.sqrt(tf.pow(a,2) + tf.pow(b,2))

c_1 = pythagorean_triple(3, 4)
c_2 = pythagorean_triple(5, 12)     # running this line results in the error below

'Output':
ValueError: Variable a already exists, disallowed. Did you mean to set reuse=True 
or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:

  File "<ipython-input-151-ffc629ee0763>", line 2, in pythagorean_triple
    a = tf.get_variable(name = "a", initializer = tf.constant(a, dtype=tf.float32))
  File "<ipython-input-152-95f3a5cca1ea>", line 1, in <module>
    c_1 = pythagorean_triple(3, 4)
  File ".../interactiveshell.py", line 2910, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
```

### Linear Regression with TensorFlow
In this section, we use TensorFlow to implement a Linear Regression machine learning model. This example builds on previous examples in familiarizing with TensorFlow for building learning models. The goal is to strengthen experience with common TensorFlow low-level modeling functions as well as the two-part step to modeling with includes building computational graphs and executing them in Sessions.

We define a set of flags using the Module `tf.app.flags`. These flags are a convenient way to wrap and define the static variables that are used in the program. However, if the value of a flag is changed, it must be cleared before it can be re-initialized. The code in the method below is called to reset the flags in a TensorFlow programme. It is used a lot in this book.

```python
# method to clear flags
def delete_flags(FLAGS):
     flags_dict = FLAGS._flags()    
     keys_list = [keys for keys in flags_dict]    
     for keys in keys_list:    
         FLAGS.__delattr__(keys)
```

In the example below, we use the sample Boston house-prices dataset from the **Scikit-learn dataset package** to build a Linear Regression model with TensorFlow.

```python
# import packages
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# clear graph (if any) before running
tf.reset_default_graph()

# load dataset
data = datasets.load_boston()

# separate features and target
X = data.data
y = data.target

# split in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

# standardize the dataset
scaler_X_train = StandardScaler().fit(X_train)
scaler_X_test = StandardScaler().fit(X_test)
X_train = scaler_X_train.transform(X_train)
X_test = scaler_X_test.transform(X_test)

# call method to clear existing flags
delete_flags(tf.flags.FLAGS)

# wrap parameters as flags
flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'number of steps to run model trainer.')
flags.DEFINE_integer('display', 100, 'display training information per step.')
flags.DEFINE_integer('ncols', X_train.shape[1], 'number of features in dataset.')
flags.DEFINE_integer('batch_size', 100, 'number of batches.')
flags.DEFINE_integer('length', X_train.shape[0], 'number of observations.')
# initialize flags
FLAGS = flags.FLAGS

# reshape y-data to become column vector
y_train = np.reshape(y_train, [-1, 1])
y_test = np.reshape(y_test, [-1, 1])

# construct the data place-holders
input_X = tf.placeholder(name='input_X', dtype = tf.float32,
                          shape=[None, FLAGS.ncols])
input_y = tf.placeholder(name='input_y', dtype = tf.float32,
                          shape=[None, 1])

# initialize weight and bias variables
weight = tf.Variable(name='weight', initial_value = tf.random_normal([FLAGS.ncols, 1]))
bias = tf.Variable(name='bias', initial_value = tf.constant(1.0, shape=[]))

# build the linear model
prediction = tf.add(tf.matmul(input_X, weight), bias)

# squared error loss or cost function for linear regression
loss = tf.losses.mean_squared_error(labels = input_y, predictions = prediction)

# optimizer to minimize cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate = FLAGS.learning_rate)
training_op = optimizer.minimize(loss)

# define root-mean-square-error (rmse) metric
rmse = tf.metrics.root_mean_squared_error(labels = input_y,
                                          predictions = prediction,
                                          name = "rmse")

# execute in Session
with tf.Session() as sess:
    # initialize all variables
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
   
    # Train the model
    for steps in range(FLAGS.epochs):
        mini_batch = zip(range(0, FLAGS.length, FLAGS.batch_size),
                   range(FLAGS.batch_size, FLAGS.length+1, FLAGS.batch_size))
        
        # train data in mini-batches
        for (start, end) in mini_batch:
            sess.run(training_op, feed_dict = {input_X: X_train[start:end],
                                               input_y: y_train[start:end]})
    
        # evaluate loss function
        if (steps+1) % FLAGS.display == 0:            
            train_loss = sess.run(loss, feed_dict = {input_X: X_train,
                                                     input_y: y_train})
            test_loss = sess.run(loss, feed_dict = {input_X: X_test,
                                                   input_y: y_test})
    
            print('Step {}: \tTrain loss: {:.2f} \tTest loss: {:.2f}'.format(
                    (steps+1), train_loss, test_loss))

    # report rmse for training and test data
    print('\nTraining set (rmse): {}'.format(sess.run(rmse,
          feed_dict = {input_X: X_train, input_y: y_train})[1]))
    print('Test set (rmse): {}'.format(sess.run(rmse,
          feed_dict = {input_X: X_test, input_y: y_test})[1]))
          
'Output':
Step 100:       Train loss: 23.52       Test loss: 25.54
Step 200:       Train loss: 22.66       Test loss: 24.68
Step 300:       Train loss: 22.46       Test loss: 24.49
Step 400:       Train loss: 22.39       Test loss: 24.40
Step 500:       Train loss: 22.36       Test loss: 24.35
Step 600:       Train loss: 22.34       Test loss: 24.32
Step 700:       Train loss: 22.33       Test loss: 24.30
Step 800:       Train loss: 22.33       Test loss: 24.28
Step 900:       Train loss: 22.32       Test loss: 24.27
Step 1000:      Train loss: 22.32       Test loss: 24.27

Training set (rmse): 4.724502086639404
Test set (rmse): 4.775963306427002
```

Here are a few points and methods to take note of in the preceding code listing for Linear Regression with TensorFlow:
 - Note that transformation to standardize the feature dataset is performed after splitting the data into train and test sets. This action is performed in this manner to prevent information from the training data to pollute the test data which must remain unseen by the model.
 - The squared error loss function for Linear Regression is implemeted in TensorFlow using the method `tf.losses.mean_squared_error`.
 - The Gradient Descent optimization technique is used to train the model. It is implemented by calling `tf.train.GradientDescentOptimizer`, which uses the `.minimize()` method to update the loss function.
 - The `tf.metrics.root_mean_squared_error` method is used to implemented the operation to compute the Root mean squared error evaluation metric for accessing the performance of the Linear Regression model.
 - Flags implemented with `tf.app.flags` is used to cleanly and elegantly manage the static parameters for tuning the model.
 - The `tf.local_variables_initializer()` works just like the `tf.global_variables_initializer()`, however, the former initializes variables that are local to the machine whereas the later initializes variables that are shared across a distributed environment.

### Classification with TensorFlow
In this example, we use the popular Iris flowers dataset to build a multivariable Logistic Regression machine learning classifier with TensorFlow. The dataset is gotten from the Scikit-learn dataset package. 

```python
# import packages
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# clear graph (if any) before running
tf.reset_default_graph()

# load dataset
data = datasets.load_iris()

# separate features and target
X = data.data
y = data.target

# apply one-hot encoding to targets
one_hot_encoder = OneHotEncoder()
encode_categorical = y.reshape(len(y), 1)
y = one_hot_encoder.fit_transform(encode_categorical).toarray()

# split in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

# call method to clear existing flags
delete_flags(tf.flags.FLAGS)

# wrap parameters as flags
flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.3, 'initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'number of steps to run model trainer.')
flags.DEFINE_integer('display', 100, 'display training information per step.')
flags.DEFINE_integer('ncols', X_train.shape[1], 'number of features in dataset.')
flags.DEFINE_integer('batch_size', 30, 'number of batches.')
flags.DEFINE_integer('length', X_train.shape[0], 'number of observations.')
# initialize flags
FLAGS = flags.FLAGS

# construct the data place-holders
input_X = tf.placeholder(name='input_X', dtype = tf.float32,
                         shape=[None, FLAGS.ncols])
input_y = tf.placeholder(name='input_y', dtype = tf.float32,
                         shape=[None, 3])

# initialize weight and bias variables
weight = tf.Variable(name='weight', initial_value = tf.random_normal([FLAGS.ncols, 3]))
bias = tf.Variable(name='bias', initial_value = tf.random_normal([3]))

# build the linear model
prediction = tf.add(tf.matmul(input_X, weight), bias)

# softmax cross entropy loss or cost function for logistic regression
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels = input_y, logits = prediction))

# optimizer to minimize cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate = FLAGS.learning_rate)
training_op = optimizer.minimize(loss)

# define accuracy metric
accuracy = tf.metrics.accuracy(labels =  tf.argmax(input_y, 1),
                               predictions = tf.argmax(prediction, 1),
                               name = "accuracy")

# execute in Session
with tf.Session() as sess:
    # initialize all variables
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    
    # Train the model
    for steps in range(FLAGS.epochs):
        mini_batch = zip(range(0, FLAGS.length, FLAGS.batch_size),
                   range(FLAGS.batch_size, FLAGS.length+1, FLAGS.batch_size))
        
        # train data in mini-batches
        for (start, end) in mini_batch:
            sess.run(training_op, feed_dict = {input_X: X_train[start:end],
                                               input_y: y_train[start:end]})
    
        # evaluate loss function
        if (steps+1) % FLAGS.display == 0:            
            train_loss = sess.run(loss, feed_dict = {input_X: X_train,
                                                     input_y: y_train})
    
            test_loss = sess.run(loss, feed_dict = {input_X: X_test,
                                                   input_y: y_test})
    
            print('Step {}: \tTrain loss: {:.2f} \tTest loss: {:.2f}'.format(
                    (steps+1), train_loss, test_loss))
            
    # report accuracy for training and test data
    print('\nTraining set (accuracy): {}'.format(sess.run(accuracy,
           feed_dict = {input_X: X_train, input_y: y_train})[1]))
    print('Test set (accuracy): {}'.format(sess.run(accuracy,
           feed_dict = {input_X: X_test, input_y: y_test})[1]))

'Output':
Step 100:       Train loss: 0.10        Test loss: 0.21
Step 200:       Train loss: 0.08        Test loss: 0.20
Step 300:       Train loss: 0.07        Test loss: 0.18
Step 400:       Train loss: 0.06        Test loss: 0.17
Step 500:       Train loss: 0.05        Test loss: 0.17
Step 600:       Train loss: 0.05        Test loss: 0.16
Step 700:       Train loss: 0.05        Test loss: 0.16
Step 800:       Train loss: 0.04        Test loss: 0.15
Step 900:       Train loss: 0.04        Test loss: 0.15
Step 1000:      Train loss: 0.04        Test loss: 0.15

Training set (accuracy): 0.9910714030265808
Test set (accuracy): 0.9733333587646484
```

From the preceding code listing, take note of the following steps and functions:
- Note how the target variable `y` is converted to a one-hot encoded matrix by using the `OneHotEncoder` function from Scikit-learn. They exist a TensorFlow method named `tf.one_hot` for performing the same function, even easier! The reader is encouraged to experiment with this.
- Observe how the `tf.reduce_mean` and the `tf.nn.softmax_cross_entropy_with_logits_v2` methods are used to implement the logistic function for learning the probabilities that a data record or observation belongs to a particular class.
- The Gradient Descent optimization algorithm ` tf.train.GradientDescentOptimizer` is used to train the logistic model.
- Observe how the `weight` and `bias` variables are updated by the gradient descent optimizer within the `Session` when the `training_op` variable is executed. The variable `training_op` minimizes the logistic model which is the `loss` variable in the code. The `loss` variable, in turn, calls the `weight` and `bias` variables in implementing the linear model.
- The `tf.metrics.accuracy` method is used to implement the operation to compute the Accuracy of the model after training.
- Again observe how `tf.app.flags` is used to cleanly and elegantly manage the static parameters for tuning the model.


### Multilayer Perceptron (MLP)
In this section, we'll use the popular MNIST handwriting dataset to classify a set of handwriting images into their respective classes. This dataset is the _defacto_ dataset to work with when starting off with building a classifier for **image classification** problems. Nicely, because of its popularity, the MNIST dataset comes prepacked with TensorFlow in the module `tensorflow.examples.tutorials.mnist`. However, the original dataset can easily be downloaded from Yann LecCun's MNIST page (<a href="http://yann.lecun.com/exdb/mnist/">http://yann.lecun.com/exdb/mnist/</a>).

The following code example will build a simple MLP Neural Network for the computer to classify a handwriting image into its appropriate class. This tutorial will introduce working with several TensorFlow methods for building neural networks. The code snippet follows with comments (as usual). A more detailed explanation of the code listing is provided thereafter. The network architecture has the following layers:
- A dense hidden layer with 250 neurons,
- A dropout layer, and
- An output layer with 10 output classes.

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# clear graph (if any) before running
tf.reset_default_graph()

# download data into current directory
data = input_data.read_data_sets('./tmp/mnist_data', one_hot = True)
# split data int training and evaluation sets
trainX = data.train.images
trainY = data.train.labels
testX = data.test.images
testY = data.test.labels

# parameters
BATCH_SIZE = 100
EPOCHS = 10
NUM_OBS = trainX.shape[0] # ==>55000
PRINT_FLAG = 2

# build the computational graph
X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int64, [None, 10])

rate_hidden = tf.placeholder(tf.float32)

# hidden layer, input dimension = 784,  output dimension = 256
hidden1 = tf.layers.dense(inputs=X, units=256, activation=tf.nn.relu, name='hidden1')
# dropout layer, input dimension = 256,  output dimension = 256
dropout1 = tf.layers.dropout(inputs=hidden1, rate=rate_hidden)
# output layer, input dimension = 256,  output dimension = 10
output = tf.layers.dense(inputs=dropout1, units=10, activation=tf.nn.relu, name='output')

# optimization operations
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output))
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)

# define accuracy metric
accuracy = tf.metrics.accuracy(labels=tf.argmax(y, 1),
                               predictions=tf.argmax(output, 1),
                               name="accuracy")

# operation to initialize all variables
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
# execute Session
with tf.Session() as sess:
    # initialize all variables
    sess.run(init_op)
    
    # Train the model
    for steps in range(EPOCHS):
        mini_batch = zip(range(0, NUM_OBS, BATCH_SIZE),
                   range(BATCH_SIZE, NUM_OBS+1, BATCH_SIZE))
        
        # train data in mini-batches
        for (start, end) in mini_batch:
            sess.run(optimizer, feed_dict = {X: trainX[start:end],
                                             y: trainY[start:end],
                                             rate_hidden: 0.5})
        # print accuracy after some steps 
        if (steps+1) % PRINT_FLAG == 0:
            accuracy_train = sess.run(accuracy, {X: trainX, y: trainY,
                                                rate_hidden: 1.0})
            accuracy_test = sess.run(accuracy, {X: testX, y: testY,
                                                rate_hidden: 1.0})
            print('Step {}: \tTraining Accuracy: {:.2f} \tTest Accuracy: {:.2f}'.format(
                    (steps+1), accuracy_train[1], accuracy_test[1]))

'Output':
Extracting ./tmp/mnist_data/train-images-idx3-ubyte.gz
Extracting ./tmp/mnist_data/train-labels-idx1-ubyte.gz
Extracting ./tmp/mnist_data/t10k-images-idx3-ubyte.gz
Extracting ./tmp/mnist_data/t10k-labels-idx1-ubyte.gz
Step 2:         Training Accuracy: 0.96         Test Accuracy: 0.96
Step 4:         Training Accuracy: 0.96         Test Accuracy: 0.96
Step 6:         Training Accuracy: 0.97         Test Accuracy: 0.97
Step 8:         Training Accuracy: 0.97         Test Accuracy: 0.97
Step 10:        Training Accuracy: 0.97         Test Accuracy: 0.97
```

From the preceding code listing, take note of the following steps and functions:
- A neural network layer is constructed with the method `tf.layers.dense`. This method implements the operation to compute an affine transformation (or weighted sum) and pass it through an activation function i.e., `activation((weight * input) + bias)`.
- The method `tf.layers.dropout` is a utility method for creating a dropout layer. The attribute `rate` determines the percentage of neurons to drop from the layer.
- The method `tf.train.RMSPropOptimizer()` implements the RMSProp gradient optimization technique. Other adaptive learning rate optimization methods include the:
  -  Adam optimizer `tf.train.AdamOptimizer`,
  -  Adadelta optimizer `tf.train.AdadeltaOptimizer`, and
  -  Adagrad optimizer `tf.train.AdagradOptimizer`.

### Visualizing with TensorBoard
In this section, before proceeding with building advanced deep neural network models with TensorFlow, we will first examine visualizing our TensorFlow graphs and statistics using TensorBoard, as this becomes an indispensable tool moving forward. Here we improve on the previous code to build a model for handwriting image classification by adding methods to visualize the graph and other variable statistics in TensorBoard. Details of the code are provided after the listing.

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import google.datalab.ml as ml

# clear graph (if any) before running
tf.reset_default_graph()

# download data into current directory
data = input_data.read_data_sets('./tmp/mnist_data', one_hot = True)

# split data int training and evaluation sets
trainX = data.train.images
trainY = data.train.labels
testX = data.test.images
testY = data.test.labels

# parameters
BATCH_SIZE = 100
EPOCHS = 10
NUM_OBS = trainX.shape[0] # ==> 55000
PRINT_FLAG = 2

# build the computational graph
X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int64, [None, 10])

rate_hidden = tf.placeholder(tf.float32)

# hidden layer, input dimension = 784,  output dimension = 256
hidden1 = tf.layers.dense(inputs=X, units=256, activation=tf.nn.relu, name='hidden1')
# dropout layer, input dimension = 256,  output dimension = 256
dropout1 = tf.layers.dropout(inputs=hidden1, rate=rate_hidden)
# output layer, input dimension = 256,  output dimension = 10
output = tf.layers.dense(inputs=dropout1, units=10, activation=tf.nn.relu, name='output')

# get histogram summaries for layer weights weights
with tf.variable_scope('hidden1', reuse=True):
    tf.summary.histogram("hidden_1", tf.get_variable('kernel'))
with tf.variable_scope('output', reuse=True):
    tf.summary.histogram("output", tf.get_variable('kernel'))

# optimization operations
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output))
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)

# define accuracy metric
accuracy = tf.metrics.accuracy(labels=tf.argmax(y, 1),
                               predictions=tf.argmax(output, 1),
                               name="accuracy")

# get scalar summary for loss and accuracy
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy[1])

# operation to initialize all variables
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

# execute Session
with tf.Session() as sess:
    # to view logs, run 'tensorboard --logdir=./tf_logs'
    train_writer = tf.summary.FileWriter("./tf_logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("./tf_logs/test")
    merged = tf.summary.merge_all()
    
    # initialize all variables
    sess.run(init_op)
    
    # Train the model
    for steps in range(EPOCHS):
        mini_batch = zip(range(0, NUM_OBS, BATCH_SIZE),
                   range(BATCH_SIZE, NUM_OBS+1, BATCH_SIZE))
        
        # train data in mini-batches
        for (start, end) in mini_batch:
            sess.run(optimizer, feed_dict = {X: trainX[start:end],
                                             y: trainY[start:end],
                                             rate_hidden: 0.5})
        # print accuracy after some steps 
        if (steps+1) % PRINT_FLAG == 0:
            accuracy_train = sess.run(accuracy, {X: trainX, y: trainY,
                                                rate_hidden: 1.0})
            accuracy_test = sess.run(accuracy, {X: testX, y: testY,
                                                rate_hidden: 1.0})
            print('Step {}: \tTraining Accuracy: {:.2f} \tTest Accuracy: {:.2f}'.format(
                    (steps+1), accuracy_train[1], accuracy_test[1]))

'Output':
Extracting ./tmp/mnist_data/train-images-idx3-ubyte.gz
Extracting ./tmp/mnist_data/train-labels-idx1-ubyte.gz
Extracting ./tmp/mnist_data/t10k-images-idx3-ubyte.gz
Extracting ./tmp/mnist_data/t10k-labels-idx1-ubyte.gz
Step 2:         Training Accuracy: 0.96         Test Accuracy: 0.96
Step 4:         Training Accuracy: 0.96         Test Accuracy: 0.96
Step 6:         Training Accuracy: 0.97         Test Accuracy: 0.97
Step 8:         Training Accuracy: 0.97         Test Accuracy: 0.97
Step 10:        Training Accuracy: 0.97         Test Accuracy: 0.97
```

From the preceding code listing, take note of the following steps and functions:
- Variable scopes using the `tf.variable_scope` method is used together with the `tf.get_variable()` method to retrieve the weights from a network Layer `tf.layers.dense`. The weight matrix of a `tf.layers.dense` layer is implicitly named `kernel`, and we can access it by prefixing the name assigned to the later. The variable scope method is used here to indicate that the variable named `hidden1/kernel` already exists, and we only want to reuse it for logging purposes.
- The method `tf.summary.histogram` is used to capture the distribution of a Tensor and display it as a Histogram in TensorBoard.
- The method `tf.summary.scalar` is used to capture a single scalar value as it changes with each iteration.
- The FileWriter class in `tf.summary.FileWriter` writes the data for the summaries to a file location on disk.
- The method `tf.summary.merge_all()` merges all summaries collected from the graph.
- Observed that it is the variable which contains the merged summaries that is evaluated using `sess.run()`.

To run TensorBoard, execute the code `tensorboard_pid = ml.TensorBoard.start('./tf_logs')` from the Datalab cell. Click on the provided link to open the TensorBoard interface on your browser. Figure 9 shows the visualization dashboards for the various metrics captures by TensorFlow. From left to right they include the graphs, scalar, distribution and histogram dashboards.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/tensorboard.png">
    <div class="figcaption" style="text-align: center;">
        Figure 9: TensorBoard visualization dashboards. Top left: Graph Dashboard - showing visual diagram of the computational graph. Top right: Scalars dashboard - illustrates the changes in the value of the scalar variables as the network trains. Bottom left: The Distribution dashboard. Bottom right: The Histogram dashboard.
    </div>
</div>

### Running TensorFlow with GPUs
GPU is short for Graphics Processing Unit. It is a specialized processor designed for carrying out simple calculations on massive datasets, which is often the case when building models with deep learning techniques. TensorFlow supports processing on both the CPUs and GPUs.

Certain TensorFlow operations (like matrix multiplication `tf.matmul`) are optimized to run on both CPUs and GPUs. When such operations are called, TensorFlow attempts to run them first on the systems GPU, if no GPUs are present, then TensorFlow will run on the CPU. To know which device is exeuting a particular operation set the attribute `log_device_placement` to `True` when creating the Session object. As an example:

```python
# A simple graph
mul_2 = tf.constant([2, 4, 6, 8, 10, 12, 14, 16, 18], shape=[3, 3], name='mul_2')
mul_3 = tf.constant([3, 6, 12, 15, 18, 21, 24, 27], shape=[3, 3], name='mul_3')
result = tf.matmul(mul_2, mul_3)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(result))

'Output':
[[180. 216. 252.]
 [396. 486. 576.]
 [612. 756. 900.]]
Device mapping: no known devices.
MatMul: (MatMul): /job:localhost/replica:0/task:0/cpu:0
mul_3: (Const): /job:localhost/replica:0/task:0/cpu:0
mul_2: (Const): /job:localhost/replica:0/task:0/cpu:0
```

TensorFlow can be manually assigned to execute an operation on a specific device using the `tf.device()` method.
```python
# perform execution on a CPU device
with tf.device('/cpu:0'):
    mul_2 = tf.constant([2, 4, 6, 8, 10, 12, 14, 16, 18], shape=[3, 3], name='mul_2')
    mul_3 = tf.constant([3, 6, 12, 15, 18, 21, 24, 27], shape=[3, 3], name='mul_3')
    res = tf.add(mul_2, mul_3)

# perform execution on a GPU device
with tf.device('/gpu:0'):
    mul_2 = tf.constant([2, 4, 6, 8, 10, 12, 14, 16, 18], shape=[3, 3], name='mul_2')
    mul_3 = tf.constant([3, 6, 12, 15, 18, 21, 24, 27], shape=[3, 3], name='mul_3')
    res = tf.add(mul_2, mul_3)
```

If multiple devices exits on the system, for example multiple GPUs, set the attribute `allow_soft_placement` to `True` when creating a Session object to allow TensorFlow to select the available and supported device on the system, if the specifc one assigned is not present.

```python
with tf.device('/gpu:0'):
    mul_2 = tf.constant([2, 4, 6, 8, 10, 12, 14, 16, 18], shape=[3, 3], name='mul_2')
    mul_3 = tf.constant([3, 6, 12, 15, 18, 21, 24, 27], shape=[3, 3], name='mul_3')
    res = tf.add(mul_2, mul_3)

with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True)) as sess:
    print(sess.run(result))
```

Where there are multiple GPUs on the machine, TensorFlow doesn't automatically choose where to run the code, it has to be manually assigned when working with low/medium level Tensor APIs. Using Estimators is a different proposition, which is why it is generally a preferred option for using TensorFlow as we will see later.

TensorFlow can leverage processing on multiple GPUs to greatly speed up the speed of computation, especially when training a complex network architecture. To take advantage of this parallel processing, a replica of the network architecture resides on each GPU machine and trains a subset of the data. However, for synchronous updates, the model parameters from each tower (or GPU machines) are stored and updated on a CPU to speed up processing. Moreso this type of mean or averaging is what CPUs are generally good at. Moreso, it is a known case that GPUs take a performance hit when moving data from one tower to another. A diagram of this operation is shown in Figure 10.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/multiple_GPUs.png">
    <div class="figcaption" style="text-align: center;">
        Figure 10: Framework for training on Multiple GPUs.
    </div>
</div>

Here is an example in code of executing subsets of the data on different GPU devices:

```python
import tensorflow as tf

# clear graph (if any) before running
tf.reset_default_graph()

data = [2, 4, 6, 8, 10, 12, 14, \
         16, 18, 20, 22, 24, 26, \
         28, 30, 32, 34, 36, 38, \
         40, 42, 44, 46, 48, 50]

global_average = []
with tf.device('/device:GPU:1'):
    sum_op_1 = tf.Variable(0, dtype=tf.float32)
    for value in data[:int(len(data)/2)]:
        sum_op_1 = tf.assign_add(sum_op_1, value)
    global_average.append(sum_op_1)
         
with tf.device('/device:GPU:2'):
    sum_op_2 = tf.Variable(0, dtype=tf.float32)
    for value in data[int(len(data)/2):]:
        sum_op_2 = tf.assign_add(sum_op_2, value)
    global_average.append(sum_op_2)

with tf.device('/cpu:0'):
    average_op = tf.add_n(global_average)/len(data)

sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True))
sess.run(tf.global_variables_initializer())
print('Running average: {}'.format(sess.run(average_op)))
sess.close()

'Output':
Running average: 26.0
```

### Convolutional Neural Networks
In this example, we will build a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. CIFAR-10 is another standard image classification dataset to classify a coloured 32 x 32 pixel image data into 10 image classes namely, airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. In keeping with the simplicity of this text, we will work with the version of this dataset that comes prepacked with Keras `tf.keras.datasets` instead of writing the code to download and import the dataset from <a href="https://www.cs.toronto.edu/~kriz/cifar.html">https://www.cs.toronto.edu/~kriz/cifar.html</a> into Python. Moreso, the focus of this section is exclusively on using TensorFlow functions to build a CNN classifier.

The CNN model architecture implemented loosely mirrors the Krizhevsky's architecture. The network architecture has the following layers:
- Convolution layer: kernel_size => [5 x 5]
- Local response normalization
- Max pooling: pool size => [2 x 2]
- Convolution layer: kernel_size => [5 x 5]
- Local response normalization
- Max pooling: pool size => [2 x 2]
- Convolution layer: kernel_size => [3 x 3]
- Convolution layer: kernel_size => [3 x 3]
- Max Pooling: pool size => [2 x 2]
- Dense Layer: units => [512]
- Dense Layer: units => [512]
- Output Layer: units => [10]

The details of the code are provided after the listing.

```python
import tensorflow as tf
import google.datalab.ml as ml

# clear graph (if any) before running
tf.reset_default_graph()

# method to download and load data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# convert targets to one-hot vectors
num_classes = 10
sess = tf.Session()
y_train_one_hot = tf.one_hot(y_train, num_classes, axis=1)
y_train = sess.run(tf.reshape(y_train_one_hot, [-1,num_classes]))

y_test_one_hot = tf.one_hot(y_test, num_classes, axis=1)
y_test = sess.run(tf.reshape(y_test_one_hot, [-1,num_classes]))
sess.close()

# parameters
BATCH_SIZE = 1000
EPOCHS = 100
NUM_OBS_TRAIN = x_train.shape[0] # ==> 50000
NUM_OBS_TEST = x_test.shape[0] # ==> 10000
PRINT_FLAG = 10

with tf.device('/device:GPU:2'):
    # build the computational graph
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None, 10])
    
    # convolutional layer 1
    conv1 = tf.layers.conv2d(inputs=X, filters=64,
                             kernel_size=[5,5], padding='same',
                             activation=tf.nn.relu, name='conv1')
    # local response normalization 1
    lrn1 = tf.nn.local_response_normalization(conv1, name='lrn1')
    # max pool 1
    maxpool1 = tf.layers.max_pooling2d(inputs=lrn1, pool_size=[2,2],
                                       strides=2, padding='same',
                                       name='maxpool1')
    # convolutional layer 2
    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=64,
                             kernel_size=[5,5], padding='same',
                             activation=tf.nn.relu, name='conv2')
    # local response normalization 2
    lrn2 = tf.nn.local_response_normalization(conv2, name='lrn2')
    # max pool 2
    maxpool2 = tf.layers.max_pooling2d(inputs=lrn2, pool_size=[2,2],
                                       strides=2, padding='same',
                                       name='maxpool2')
    # convolutional layer 3
    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=32,
                             kernel_size=[3,3], padding='same',
                             activation=tf.nn.relu, name='conv3')
    # convolutional layer 4
    conv4 = tf.layers.conv2d(inputs=conv3, filters=32,
                             kernel_size=[3,3], padding='same',
                             activation=tf.nn.relu, name='conv4')
    # max pool 3
    maxpool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2],
                                       strides=2, padding='same',
                                       name='maxpool3')
    # flatten the Tensor
    flatten = tf.layers.flatten(maxpool3, name='flatten')
    # dense fully-connected layer 1
    dense1 = tf.layers.dense(inputs=flatten, units=512,
                             activation=tf.nn.relu, name='dense1')
    # dense fully-connected layer 2
    dense2 = tf.layers.dense(inputs=dense1, units=512,
                             activation=tf.nn.relu, name='dense2')
    # output layer
    output = tf.layers.dense(inputs=dense2, units=10,
                             activation=tf.nn.relu, name='output')
    
    # optimization operations
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,\
                            logits=output))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=1e-04,
                                               global_step=global_step,
                                               decay_steps=100000,
                                               decay_rate=0.96,
                                               staircase=True,
                                               name='exp_decay_lr')
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss=loss,
                                      global_step=global_step)
    
    # define accuracy metric
    accuracy = tf.metrics.accuracy(labels=tf.argmax(y, 1),
                                   predictions=tf.argmax(output, 1),
                                   name="accuracy")
    precision = tf.metrics.precision(labels=tf.argmax(y, 1),
                                     predictions=tf.argmax(output, 1),
                                   name="precision")
    recall = tf.metrics.recall(labels=tf.argmax(y, 1),
                               predictions=tf.argmax(output, 1),
                               name="recall")
    f1 = 2 * accuracy[1] * recall[1] / ( precision[1] + recall[1] )
    
    # get scalar summary for loss and accuracy
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy[1])
    tf.summary.scalar('precision', precision[1])
    tf.summary.scalar('recall', recall[1])
    tf.summary.scalar('f1_score', f1)
    
    # operation to initialize all variables
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

# execute Session
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, \
                  log_device_placement=True)) as sess:
    # to view logs, run 'tensorboard --logdir=./cnn_logs'
    train_writer = tf.summary.FileWriter("./cnn_logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("./cnn_logs/test")
    merged = tf.summary.merge_all()
    
    # initialize all variables
    sess.run(init_op)
    
    # Train the model
    for steps in range(EPOCHS):
        mini_batch_train = zip(range(0, NUM_OBS_TRAIN, BATCH_SIZE),
                   range(BATCH_SIZE, NUM_OBS_TRAIN+1, BATCH_SIZE))
        
        # train data in mini-batches
        for (start, end) in mini_batch_train:
            sess.run(optimizer, feed_dict = {X: x_train[start:end],
                                             y: y_train[start:end]})
        # print accuracy after some steps 
        if (steps+1) % PRINT_FLAG == 0:
            print('Step: {}'.format((steps+1)))
            
            # evaluate on train dataset
            mini_batch_train = zip(range(0, NUM_OBS_TRAIN, BATCH_SIZE),
                   range(BATCH_SIZE, NUM_OBS_TRAIN+1, BATCH_SIZE))
            
            temp_acc_train = []
            temp_summ_train = None
            for (start, end) in mini_batch_train:                
                acc_train, summ_train = sess.run([accuracy, merged],
                                                 {X:x_train[start:end],
                                                  y:y_train[start:end]})
                temp_acc_train.append(acc_train[1])
                temp_summ_train = summ_train
                
            # write train summary
            train_writer.add_summary(temp_summ_train, (steps+1))
            print('Accuracy on training data: {}'.format(
                    sum(temp_acc_train)/float(len(temp_acc_train))))
            
            # Evaluate on test dataset
            mini_batch_test = zip(range(0, NUM_OBS_TEST, BATCH_SIZE),
                   range(BATCH_SIZE, NUM_OBS_TEST+1, BATCH_SIZE))
            
            temp_acc_test = []
            temp_summ_test = None
            for (start, end) in mini_batch_test:                
                acc_test, summ_test = sess.run([accuracy, merged],
                                               {X:x_test[start:end],
                                                y:y_test[start:end]})
                temp_acc_test.append(acc_test[1])
                temp_summ_test = summ_test
            
            # write test summary
            test_writer.add_summary(temp_summ_test, (steps+1))
            print('Accuracy on testing data: {}'.format(
                    sum(temp_acc_test)/float(len(temp_acc_test))))

'Output':
Step: 10
Accuracy on training data: 0.5450000166893005
Accuracy on testing data: 0.5362499952316284
Step: 20
Accuracy on training data: 0.5636666417121887
Accuracy on testing data: 0.5695000290870667
Step: 30
Accuracy on training data: 0.5882999897003174
Accuracy on testing data: 0.5915833115577698
Step: 40
Accuracy on training data: 0.606071412563324
Accuracy on testing data: 0.6075624823570251
Step: 50
Accuracy on training data: 0.617388904094696
Accuracy on testing data: 0.6176000237464905
Step: 60
Accuracy on training data: 0.626909077167511
Accuracy on testing data: 0.6275416612625122
Step: 70
Accuracy on training data: 0.6356538534164429
Accuracy on testing data: 0.6359285712242126
Step: 80
Accuracy on training data: 0.6431000232696533
Accuracy on testing data: 0.6427500247955322
Step: 90
Accuracy on training data: 0.6490588188171387
Accuracy on testing data: 0.6484166383743286
Step: 100
Accuracy on training data: 0.6539999842643738
Accuracy on testing data: 0.6526749730110168
```

From the preceding code listing, take note of the following steps and functions:

- `google.datalab.ml` is a library containing CloudML helper methods. We use the `google.datalab.ml.TensorBoard`  method to launch TensorBoard from Google Datalab.
- Observe how the computational graph is constructed used methods from `tf.layers`.
  - Convolutional layer `tf.layers.conv2d`
  - Max-Pooling `tf.layers.max_pooling2d`
  - Method to flatten the tensor while maintaining the batch axis before passing into a Dense layer `tf.layers.flatten`
  - Fully connected layer `tf.layers.dense`
  - The `tf.layers` package has many more methods for building different types of neural network layers.
- Note the `tf.nn` packgae which provides low-level computational neural network operations. This is another vital package for constructing neural networks.
- The `tf.nn.local_response_normalization` method implemnts a normalization technique from the famous AlexNet neural network architecture. We use it here, because the our architecture intentionally mirrors the AlexNet architecture. The local response normalization method is however not fancied in practice as its effects are neglibile on the network performance.
- Observe how scalar summaries are collected for display in TensorBoard using the `tf.summary` package.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/cnn_tensorboard.png">
    <div class="figcaption" style="text-align: center;">
        Figure 11: Visualization of CNN Graph with TensorBoard. Top left: Graph Dashboard - showing visual diagram of the CNN architecture. Top right: Scalars dashboard - illustrates the changes in the Accuracy and F1 score. Bottom left: Scalars dashboard - shows the changes in the Loss and Precision score. Bottom right: Scalars dashboard - screenshot of Precision and Recall score.
    </div>
</div>

### Save and Restore TensorFlow Graph Variables 
TensorFlow provides a mechanism to save and restore the state of trained Variables for an already constructed Graph without having to run the code to re-train the Variables each time the program listing is revisited. Let's see an example of this using this simple TensorFlow programme.

```python
import tensorflow as tf
tf.reset_default_graph()    

# create variables
a = tf.get_variable(name = "a", initializer = tf.constant(3.0, dtype=tf.float32))
b = tf.get_variable(name = "b", initializer = tf.constant(4.0, dtype=tf.float32))
c = tf.sqrt(tf.pow(a,2) + tf.pow(b,2))

# initialize variables
init_op = tf.global_variables_initializer()

# ops to save variables
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(c)
    save_path = saver.save(sess, './pythagoras/model.ckpt')
    print("Model saved in file: %s" % save_path)

# restore the variables
with tf.Session() as sess:
    sess.run(init_op)
    # Restore variables from disk.
    saver.restore(sess, './pythagoras/model.ckpt')
    print("a: %s" % a.eval())
    print("b: %s" % b.eval())
    print("c: %s" % c.eval())
  
    print("Variable restored.")

'Output':
INFO:tensorflow:Restoring parameters from ./pythagoras/model.ckpt
a: 3.0
b: 4.0
c: 5.0
Variable restored.
```

From the preceding code listing, take note of the following steps and functions:
- The object `tf.train.Saver()` initialized in the variable `saver` contains the methods `save` and `restore` for saving and restoring Graph variables.
- After running the Session instance to save the variables; the variables are saved in a folder called `pythagoras`. Within the folder, the name `model.ckpt`, is used as a prefix to store the checkpoint files.
- To restore the Graphs variables, build the graph and run the Session instance containing `saver.restore`.

### Recurrent Neural Networks
This section goes through brief examples of using Recurrent Neural Networks to predict the target of a univariate and multivariate timeseries dataset using TensorFlow.

#### Univariate Timeseries with RNN
The dataset for this example is the Nigeria power consumption data from January 1 - March 11 by Hipel and McLeod (1994). Retrieved from DataMarket.

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# data url
url = "https://raw.githubusercontent.com/dvdbisong/gcp-learningmodels-book/master/Chapter_12/nigeria-power-consumption.csv"

# load data
parse_date = lambda dates: pd.datetime.strptime(dates, '%d-%m')
data = pd.read_csv(url, parse_dates=['Month'], index_col='Month',
                   date_parser=parse_date,
                   engine='python', skipfooter=2)

# print column name
data.columns

# change column names
data.rename(columns={'Nigeria power consumption': 'power-consumption'},
            inplace=True)

# split in training and evaluation set
data_train, data_eval = train_test_split(data, test_size=0.2, shuffle=False)

# MinMaxScaler - center ans scale the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data_train = scaler.fit_transform(data_train)
data_eval = scaler.fit_transform(data_eval)

# adjust univariate data for timeseries prediction
def convert_to_sequences(data, sequence, is_target=False):
    temp_df = []
    for i in range(len(data) - sequence):
        if is_target:
            temp_df.append(data[(i+1): (i+1) + sequence])
        else:
            temp_df.append(data[i: i + sequence])
    return np.array(temp_df)

# parameters
time_steps = 20
inputs = 1
neurons = 100
outputs = 1

# create training and testing data
train_x = convert_to_sequences(data_train, time_steps, is_target=False)
train_y = convert_to_sequences(data_train, time_steps, is_target=True)

eval_x = convert_to_sequences(data_eval, time_steps, is_target=False)
eval_y = convert_to_sequences(data_eval, time_steps, is_target=True)

# model parameters
learning_rate = 0.001
epochs = 2500
batch_size = 50
length = train_x.shape[0]
display = 500

# clear graph (if any) before running
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, time_steps, inputs])
y = tf.placeholder(tf.float32, [None, time_steps, outputs])

# get single output value at each time step by wrapping the cell
# in an OutputProjectionWrapper
cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicRNNCell(num_units=neurons, activation=tf.nn.relu),
        output_size=outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# squared error loss or cost function for linear regression
loss = tf.losses.mean_squared_error(labels=y, predictions=outputs)
# optimizer to minimize cost
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# define root-mean-square-error (rmse) metric
rmse = tf.metrics.root_mean_squared_error(labels = y,
                                          predictions = outputs,
                                          name = "rmse")

# save the model
saver = tf.train.Saver()

# execute in Session
with tf.Session() as sess:
    # initialize all variables
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    
    # Train the model
    for steps in range(epochs):
        mini_batch = zip(range(0, length, batch_size),
                   range(batch_size, length+1, batch_size))
        
        # train data in mini-batches
        for (start, end) in mini_batch:
            sess.run(training_op, feed_dict = {X: train_x[start:end,:,:],
                                               y: train_y[start:end,:,:]})
    
        # print training performance 
        if (steps+1) % display == 0:
            # evaluate loss function
            loss_fn = loss.eval(feed_dict = {X: eval_x, y: eval_y})
            print('Step: {}  \tTraining loss (mse): {}'.format((steps+1), loss_fn))
             
        saver.save(sess, "power_model_folder/ng_power_model")
    
    # report rmse for training and test data
    print('\nTraining set (rmse): {:.2f}'.format(sess.run(rmse,
          feed_dict = {X: train_x, y: train_y})[1]))
    print('Test set (rmse): {:.2f}'.format(sess.run(rmse,
          feed_dict = {X: eval_x, y: eval_y})[1]))
    
    y_pred = sess.run(outputs, feed_dict={X: eval_x})
    
    plt.title("Model Testing", fontsize=12)
    plt.plot(eval_x[0,:,0], "b--", markersize=10, label="training instance")
    plt.plot(eval_y[0,:,0], "g--", markersize=10, label="targets")
    plt.plot(y_pred[0,:,0], "r--", markersize=10, label="model prediction")
    plt.legend(loc="upper left")
    plt.xlabel("Time")
    
# use model to predict sequences using training data as seed
with tf.Session() as sess:
    saver.restore(sess, "power_model_folder/ng_power_model")
    rnn_data = list(data_train[:20])
    for i in range(len(data_train) - time_steps):
        batch = np.array(rnn_data[-time_steps:]).reshape(1, time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: batch})
        rnn_data.append(y_pred[0, -1, 0])
    
    plt.title("RNN vs. Original series", fontsize=12)
    plt.plot(data_train, "b--", markersize=10, label="Original series")
    plt.plot(rnn_data, "g--", markersize=10, label="RNN generated series")
    plt.legend(loc="upper left")
    plt.xlabel("Time")
    
    # inverse to normal scale and plot
    data_train_inverse = scaler.inverse_transform(data_train.reshape(-1, 1))
    rnn_data_inverse = scaler.inverse_transform(np.array(rnn_data).reshape(-1, 1))
    
    plt.title("RNN vs. Original series with normal scale", fontsize=12)
    plt.plot(data_train_inverse, "b--", markersize=10, label="Original series")
    plt.plot(rnn_data_inverse, "g--", markersize=10, label="RNN generated series")
    plt.legend(loc="upper left")
    plt.xlabel("Time")
```
```bash
'Output':
Step: 500       Training loss (mse): 0.03262907266616821
Step: 1000      Training loss (mse): 0.03713105246424675
Step: 1500      Training loss (mse): 0.03985978662967682
Step: 2000      Training loss (mse): 0.041361670941114426
Step: 2500      Training loss (mse): 0.041800327599048615
INFO:tensorflow:Restoring parameters from power_model_folder/ng_power_model

Training set (rmse): 0.08
Test set (rmse): 0.10
```

From the preceding code listing, take note of the following steps and functions:
- The dataset is pre-processed for timeseries modeling using Recurrent Neural Networks by converting the the data input and otuputs into sequences using the method `convert_to_sequences`. This method splits the dataset into rolling sequences consiting of 20 rows (or timesteps) using a window of `1`. In Figure 12, the example univariate dataset is converted into sequences of 5 time steps, where output sequence is one step ahead of the input sequence. Each sequence contains 5 rows (determined by the `time_steps` variable) and in this univariate case, 1 column.
<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/univariate_sequences.png">
    <div class="figcaption" style="text-align: center;">
        Figure 12: Converting a Univariate series into sequences for prediction with RNNs. Left: Sample univariate dataset. Center: Input sequence. Right: Output sequence.
    </div>
</div> 
- When modelling using RNNs it is important to scale the dataset to have values within the same range.
- The method `tf.contrib.rnn.BasicRNNCell` implements a vanilla RNN cell.
- The method `tf.contrib.rnn.OutputProjectionWrapper` is used to add a Dense or fully connected layer to the RNN cell. However while this is a simple solution, this method is not the most efficient. In future examples, we will pass the outputs of the recurrent network in a Dense layer using `tf.layers.dense` after reshaping.
- The method `tf.nn.dynamic_rnn` creates a recurrent neural network by dynamically unrolling the RNN cell.
- The first plot is the model predictions also showing the targets and the training instance.
<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/rnn_ts_model_testing.png" width="70%" height="70%">
    <div class="figcaption" style="text-align: center;">
        Figure 13: RNN Model Testing.
    </div>
</div> 
- The next two plots shows the original series and the RNN generated series in both the scaled and normal values.
<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/rnn_ts_vs_original.png" width="70%" height="70%">
    <img src="/assets/seminar_IEEE/rnn_ts_vs_original_normal_scale.png" width="70%" height="70%">
    <div class="figcaption" style="text-align: center;">
        Figure 14: Original series vs. RNN generated series. Left: Scaled data values. Right: Normal data values
    </div>
</div> 

#### Deep RNN
The example in this section updates sections of the previous example and code, and modifies the RNN cell to become a Deep RNN network. This network is more powerful and can learn more complex sequences. In this example, we use another type of RNN cell called the GRU or Gated Recurrent Unit.

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# data url
url = "https://raw.githubusercontent.com/dvdbisong/gcp-learningmodels-book/master/Chapter_12/nigeria-power-consumption.csv"

# load data
parse_date = lambda dates: pd.datetime.strptime(dates, '%d-%m')
data = pd.read_csv(url, parse_dates=['Month'], index_col='Month',
                   date_parser=parse_date,
                   engine='python', skipfooter=2)

# print column name
data.columns

# change column names
data.rename(columns={'Nigeria power consumption': 'power-consumption'},
            inplace=True)

# split in training and evaluation set
data_train, data_eval = train_test_split(data, test_size=0.2, shuffle=False)

# MinMaxScaler - center ans scale the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data_train = scaler.fit_transform(data_train)
data_eval = scaler.fit_transform(data_eval)

# adjust univariate data for timeseries prediction
def convert_to_sequences(data, sequence, is_target=False):
    temp_df = []
    for i in range(len(data) - sequence):
        if is_target:
            temp_df.append(data[(i+1): (i+1) + sequence])
        else:
            temp_df.append(data[i: i + sequence])
    return np.array(temp_df)

# parameters
time_steps = 20
inputs = 1
neurons = 50
outputs = 1

# create training and testing data
train_x = convert_to_sequences(data_train, time_steps, is_target=False)
train_y = convert_to_sequences(data_train, time_steps, is_target=True)

eval_x = convert_to_sequences(data_eval, time_steps, is_target=False)
eval_y = convert_to_sequences(data_eval, time_steps, is_target=True)

# model parameters
layers_num = 3
learning_rate = 0.001
epochs = 2500
batch_size = 50
length = train_x.shape[0]
display = 500

# clear graph (if any) before running
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, time_steps, inputs])
y = tf.placeholder(tf.float32, [None, time_steps, outputs])

# Deep RNN Cell
layers = [tf.contrib.rnn.GRUCell(num_units=neurons,activation=tf.nn.relu) \
          for layer in range(layers_num)]

stacked_cells = tf.contrib.rnn.MultiRNNCell(layers)
multi_output, states = tf.nn.dynamic_rnn(stacked_cells, X, dtype=tf.float32)

# pass into Dense layer
stacked_outputs = tf.reshape(multi_output, [-1, neurons])
dense_output = tf.layers.dense(inputs=stacked_outputs, units=outputs)
outputs = tf.reshape(dense_output, [-1, time_steps, outputs])

# squared error loss or cost function for linear regression
loss = tf.losses.mean_squared_error(labels=y, predictions=outputs)
# optimizer to minimize cost
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# save the model
saver = tf.train.Saver()

# execute in Session
with tf.Session() as sess:
    # initialize all variables
    tf.global_variables_initializer().run()
    
    # Train the model
    for steps in range(epochs):
        mini_batch = zip(range(0, length, batch_size),
                   range(batch_size, length+1, batch_size))
        
        # train data in mini-batches
        for (start, end) in mini_batch:
            sess.run(training_op, feed_dict = {X: train_x[start:end,:,:],
                                               y: train_y[start:end,:,:]})
    
        # print training performance 
        if (steps+1) % display == 0:
            # evaluate loss function
            loss_fn = loss.eval(feed_dict = {X: eval_x, y: eval_y})
            print('Step: {}  \tTraining loss (mse): {}'.format((steps+1), loss_fn))
             
        saver.save(sess, "power_model_folder_deep/deep_ng_power_model")
    
    y_pred = sess.run(outputs, feed_dict={X: eval_x})
    
    plt.figure(1)
    plt.title("Deep GRU RNN - Model Testing", fontsize=12)
    plt.plot(eval_x[0,:,0], "b--", markersize=10, label="training instance")
    plt.plot(eval_y[0,:,0], "g--", markersize=10, label="targets")
    plt.plot(y_pred[0,:,0], "r--", markersize=10, label="model prediction")
    plt.legend()
    plt.xlabel("Time")
    
# use model to predict sequences using training data as seed
with tf.Session() as sess:
    saver.restore(sess, "power_model_folder_deep/deep_ng_power_model")
    rnn_data = list(data_train[:20])
    for i in range(len(data_train) - time_steps):
        batch = np.array(rnn_data[-time_steps:]).reshape(1, time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: batch})
        rnn_data.append(y_pred[0, -1, 0])
    
    plt.figure(2)
    plt.title("Deep GRU RNN vs. Original series", fontsize=12)
    plt.plot(data_train, "b--", markersize=10, label="Original series")
    plt.plot(rnn_data, "g--", markersize=10, label="Deep GRU RNN generated series")
    plt.legend(loc="upper left")
    plt.xlabel("Time")
    
    # inverse to normal scale and plot
    data_train_inverse = scaler.inverse_transform(data_train.reshape(-1, 1))
    rnn_data_inverse = scaler.inverse_transform(np.array(rnn_data).reshape(-1, 1))
    
    plt.figure(3)
    plt.title("Deep GRU RNN vs. Original series with normal scale", fontsize=12)
    plt.plot(data_train_inverse, "b--", markersize=10, label="Original series")
    plt.plot(rnn_data_inverse, "g--", markersize=10, label="Deep GRU RNN generated series")
    plt.legend(loc="upper left")
    plt.xlabel("Time")    

'Output':
Step: 500       Training loss (mse): 0.13439016044139862
Step: 1000      Training loss (mse): 0.15328247845172882
Step: 1500      Training loss (mse): 0.1200922280550003
Step: 2000      Training loss (mse): 0.12217740714550018
Step: 2500      Training loss (mse): 0.12361736595630646
INFO:tensorflow:Restoring parameters from power_model_folder_deep/deep_ng_power_model
```

From the preceding code listing, take note of the following steps and functions:
- The method `tf.contrib.rnn.GRUCell` implements a GRU cell.
- Observe how multiple GRU cells are created using list comprehensions in place of a for-loop.
- The method `tf.contrib.rnn.MultiRNNCell` puts together the deep recurrent cells.
- Also in this example, in place of the `tf.contrib.rnn.OutputProjectionWrapper` method, the outputs of the recurrent network layer are reshaped into a 1-D tensor and passed to a Dense layer using `tf.layers.dense`. The ouput of the Dense layer is then resahped back into a 3-D tensor.
- The first plot is the model predictions which also shows the targets and the training instance.
<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/deep_GRU_rnn_ts_model_testing.png" width="70%" height="70%">
    <div class="figcaption" style="text-align: center;">
        Figure 15: Deep GRU RNN Model Testing.
    </div>
</div> 
- The next two plots are the original series and the Deep GRU RNN generated series shown in both the scaled and normal values.
<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/deep_GRU_rnn_ts_vs_original.png" width="70%" height="70%">
    <img src="/assets/seminar_IEEE/deep_GRU_rnn_ts_vs_original_normal_scale.png" width="70%" height="70%">
    <div class="figcaption" style="text-align: center;">
        Figure 16: Original series vs. Deep GRU RNN generated series. Left: Scaled data values. Right: Normal data values.
    </div>
</div> 

#### Multivariate Timeseries with RNN
The dataset for this example is the Dow Jones Index Data Set from the famous UCI Machine Learning Repository. In this stock dataset, each row contains the stock price record for a week including the percentage of return that stock has in the following week `percent_change_next_weeks_price`. For this example, the record for the previous week is used to predict the percent change in price for the next two weeks for Bank of America, BAC stock prices.

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# data url
url = "https://raw.githubusercontent.com/dvdbisong/gcp-learningmodels-book/master/Chapter_12/dow_jones_index.data"

# load data
data = pd.read_csv(url, parse_dates=['date'], index_col='date')

# print column name
data.columns

# print column datatypes
data.dtypes

# parameters
time_steps = 1
inputs = 10
outputs = 1
stock ='BAC'  # Bank of America

def clean_dataset(data):
    # strip dollar sign from `object` type columns
    col = ['open', 'high', 'low', 'close', 'next_weeks_open', 'next_weeks_close']
    data[col] = data[col].replace({'\$': ''}, regex=True)
    # drop NaN
    data.dropna(inplace=True)
    # rearrange columns
    columns = ['quarter', 'stock', 'open', 'high', 'low', 'close', 'volume',
       'percent_change_price', 'percent_change_volume_over_last_wk',
       'previous_weeks_volume', 'next_weeks_open', 'next_weeks_close',
       'days_to_next_dividend', 'percent_return_next_dividend',
       'percent_change_next_weeks_price']
    data = data[columns]
    return data

def data_transform(data):
    # select stock data belonging to Bank of America
    data = data[data.stock == stock]
    # adjust target(t) to depend on input (t-1)
    data.percent_change_next_weeks_price = data.percent_change_next_weeks_price.shift(-1)
    # remove nans as a result of the shifted values
    data = data.iloc[:-1,:]
    # split quarter 1 as training data and quarter 2 as testing data
    train_df = data[data.quarter == 1]
    test_df = data[data.quarter == 2]   
    return (np.array(train_df), np.array(test_df))

def normalize_and_scale(train_df, test_df):
    # remove string columns and convert to float
    train_df = train_df[:,2:].astype(float,copy=False)
    test_df = test_df[:,2:].astype(float,copy=False)
    # MinMaxScaler - center and scale the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))    
    train_df_scale = scaler.fit_transform(train_df[:,2:])
    test_df_scale = scaler.fit_transform(test_df[:,2:])  
    return (scaler, train_df_scale, test_df_scale)

# clean the dataset
data = clean_dataset(data)

# select Dow Jones stock and split into training and test sets
train_df, test_df = data_transform(data)

# scale the data
scaler, train_df_scaled, test_df_scaled = normalize_and_scale(train_df, test_df)

# split train/ test
train_X, train_y = train_df_scaled[:, :-1], train_df_scaled[:, -1]
test_X, test_y = test_df_scaled[:, :-1], test_df_scaled[:, -1]

# reshape inputs to 3D array
train_X = train_X[:,None,:]
test_X = test_X[:,None,:]

# reshape outputs
train_y = np.reshape(train_y, (-1,outputs))
test_y = np.reshape(test_y, (-1,outputs))

# model parameters
learning_rate = 0.002
epochs = 1000
batch_size = int(train_X.shape[0]/5)
length = train_X.shape[0]
display = 100
layers_num = 3
neurons = 150

# clear graph (if any) before running
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, time_steps, inputs])
y = tf.placeholder(tf.float32, [None, outputs])

# Deep LSTM Cell
layers = [tf.contrib.rnn.BasicLSTMCell(num_units=neurons,activation=tf.nn.relu) \
          for layer in range(layers_num)]

stacked_cells = tf.contrib.rnn.MultiRNNCell(layers)
multi_output, states = tf.nn.dynamic_rnn(stacked_cells, X, dtype=tf.float32)

# pass into Dense layer
stacked_outputs = tf.reshape(multi_output, [-1, neurons])
dense_output = tf.layers.dense(inputs=stacked_outputs, units=outputs)

# squared error loss or cost function for linear regression
loss = tf.losses.mean_squared_error(labels=y, predictions=dense_output)

# optimizer to minimize cost
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# execute in Session
with tf.Session() as sess:
    # initialize all variables
    tf.global_variables_initializer().run()
    
    # Train the model
    for steps in range(epochs):
        mini_batch = zip(range(0, length, batch_size),
                   range(batch_size, length+1, batch_size))
        
        # train data in mini-batches
        for (start, end) in mini_batch:
            sess.run(training_op, feed_dict = {X: train_X[start:end,:,:],
                                               y: train_y[start:end,:]})
    
        # print training performance 
        if (steps+1) % display == 0:
            # evaluate loss function
            loss_fn = loss.eval(feed_dict = {X: test_X, y: test_y})
            print('Step: {}  \tTraining loss (mse): {}'.format((steps+1), loss_fn))
            
    # Test model
    y_pred = sess.run(dense_output, feed_dict={X: test_X})
    
    plt.figure(1)
    plt.title("LSTM RNN Model Testing for '{}' stock".format(stock), fontsize=12)
    plt.plot(test_y, "g--", markersize=10, label="targets")
    plt.plot(y_pred, "r--", markersize=10, label="model prediction")
    plt.legend()
    plt.xlabel("Time")

'Output':
Step: 100       Training loss (mse): 0.1169147714972496
Step: 200       Training loss (mse): 0.10158639401197433
Step: 300       Training loss (mse): 0.10255051404237747
Step: 400       Training loss (mse): 0.10251575708389282
Step: 500       Training loss (mse): 0.09951873868703842
Step: 600       Training loss (mse): 0.09933555126190186
Step: 700       Training loss (mse): 0.10997023433446884
Step: 800       Training loss (mse): 0.0974557027220726
Step: 900       Training loss (mse): 0.09203604608774185
Step: 1000      Training loss (mse): 0.09838250279426575
```

From the preceding code listing, take note of the following steps and functions:
- The method named `clean_dataset` carries out some rudimentary clean-up of the dataset to make it suitable for modeling. The actions taken on this particular dataset involves removing the dollar sign from certain of the data columns, removing missing values and rearranging the data columns so target attribute `percent_change_next_weeks_price` is the last column.
- The method named `data_transform` subselects the stock records belonging to 'Bank of America', and the target attribute is adjusted so that the previous week record is used to predict the percent change in price for the next two weeks. Also, the dataset is split into training and testing sets.
- The method named `normalize_and_scale` removes the non-numeric columns and scales the dataset attributes.
- Observe how multiple LSTM cells `tf.contrib.rnn.BasicLSTMCell` are created using list comprehensions.
- Also in this example, in place of the `tf.contrib.rnn.OutputProjectionWrapper` method, the outputs of the recurrent network layer are reshaped into a 1-D tensor and passed to a Dense layer using `tf.layers.dense`. Note that in this example, we did not resahpe the output back into a 3-D tensor because our original target is a 1-D value and not a sequence.
- The output plot is the model predictions showing the targets and the training instance.
<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/lstm_rnn_ts_model_testing.png" width="70%" height="70%">
    <div class="figcaption" style="text-align: center;">
        Figure 17: LSTM RNN Model Testing for Bank of America stock.
    </div>
</div>

### Autoencoders
The code example in the section shows how to implement an Autoencoder network using TensorFlow. For simplicity, the MNIST handwriting dataset is used to create reconstructions of the original images. In this example, a Stacked Autoencoder is implemented. The code listing is presented below and corresponding notes on the code is shown thereafter.

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# parameters
batch_size = 128
learning_rate = 0.002
epoch = 1000
display = 100

# get MNIST data
data = input_data.read_data_sets('./MNIST', one_hot=False)

# split to test and train iages
test_x = data.test.images
test_y = data.test.labels

# input placeholder
input_X = tf.placeholder(tf.float32, [None, 784])  # 28 * 28 pixel images

# encoder
encoder_layer_1 = tf.layers.dense(input_X, 512, tf.nn.relu)
encoder_layer_2 = tf.layers.dense(encoder_layer_1, 128, tf.nn.relu)
encoder_layer_3 = tf.layers.dense(encoder_layer_2, 64, tf.nn.relu)
coding_layer = tf.layers.dense(encoder_layer_3, 4)

# decoder
decoder_layer_1 = tf.layers.dense(coding_layer, 64, tf.nn.relu)
decoder_layer_2 = tf.layers.dense(decoder_layer_1, 128, tf.nn.relu)
decoder_layer_3 = tf.layers.dense(decoder_layer_2, 512, tf.nn.relu)
decoded_output = tf.layers.dense(decoder_layer_3, 784)

loss = tf.losses.mean_squared_error(labels=input_X, predictions=decoded_output)
training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# execute autoencoder graph in session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(epoch):
    batch_x, batch_y = data.train.next_batch(batch_size)
    _, loss_ = sess.run([training_op, loss], {input_X: batch_x})
    
    # print loss
    if step % display == 0:
        print("loss: ", loss_)
    
# visualize reconstruction
sample_size = 6
test_image = data.test.images[:sample_size]
# reconstruct test samples
test_reconstruction = sess.run(decoded_output, feed_dict={input_X: test_image})

plt.figure(figsize = (8,25))
plt.suptitle('Stacked Autoencoder Reconstruction', fontsize=16)
for i in range(sample_size):
    plt.subplot(sample_size, 2, i*2+1)
    plt.title('Original image')
    plt.imshow(test_image[i].reshape((28, 28)), cmap="Greys", interpolation="nearest", aspect='auto')
    plt.subplot(sample_size, 2, i*2+2)
    plt.title('Reconstructed image')
    plt.imshow(test_reconstruction[i].reshape((28, 28)), cmap="Greys", interpolation="nearest", aspect='auto')

# close Session
sess.close()

'Output':
loss:  0.11387876
loss:  0.047890253
loss:  0.037955094
loss:  0.03574044
loss:  0.035101265
loss:  0.032688208
loss:  0.031728305
loss:  0.03372544
loss:  0.029142274
loss:  0.031097852
```

From the preceding code listing, take note of the following steps and functions:
- Observe closely the arrangement of the encoder layers, the network "codings" and the decoder layers of the Stacked Autoencoder. Specifically note how the corresponding layer arrangement of the encoder and the decoder have the same number of neurons.
- The loss error measures the squared difference between the inputs into the Autoencoder network and the decoder output.
- The plot constrasts the reconstructed images from the Autoencoder network with the original images in the dataset
<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/stacked_autoencoder.png" width="50%" height="50%">
    <div class="figcaption" style="text-align: center;">
        Figure 18: Stacked Autoencoder Reconstruction. Left: Original image. Right: Reconstructed image
    </div>
</div>

### Building Efficient Input Pipelines with the Dataset API
The Dataset API `tf.Data` offers an efficient mechanism for building robust input pipelines for passing data into a TensorFlow programme. The process includes:
- Using the Dataset package `tf.data.Dataset` to create a data instance.
- Using the Iterator package `tf.data.Iterator` to define how the records or elements are retrieved from the dataset.

This section will use the popular Iris dataset to illustrate working with the Dataset API methods for building data input pipelines in a TensorFlow graph.

```python
# import packages
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# clear graph (if any) before running
tf.reset_default_graph()

# load dataset
data = datasets.load_iris()

# separate features and target
X = data.data
y = data.target

# apply one-hot encoding to targets
one_hot_encoder = OneHotEncoder()
encode_categorical = y.reshape(len(y), 1)
y = one_hot_encoder.fit_transform(encode_categorical).toarray()

# split in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

# parameters
learning_rate = 0.003
epochs = 1000
display = 100
train_batch_size = 30
tain_dataset_size = X_train.shape[0]
test_dataset_size = X_test.shape[0]

# construct the data place-holders
input_X = tf.placeholder(name='input_X', dtype=X_train.dtype, shape=[None, X_train.shape[1]])
input_y = tf.placeholder(name='input_y', dtype=y_train.dtype, shape=[None, y_train.shape[1]])
batch_size = tf.placeholder(name='batch_size', dtype=tf.int64)

# construct data input pipelines
dataset = tf.data.Dataset.from_tensor_slices((input_X, input_y))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()

# build the model
features, labels = iterator.get_next()
hidden = tf.layers.dense(inputs=features, units=20, activation=tf.nn.relu)
prediction = tf.layers.dense(inputs=hidden, units=y_train.shape[1])

# softmax cross entropy loss or cost function for logistic regression
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=prediction))

# optimizer to minimize cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# define accuracy metric
accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                               predictions=tf.argmax(prediction, 1),
                               name="accuracy")

# execute in Session
with tf.Session() as sess:
    # initialize all variables
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
     
    for steps in range(epochs):
        # initialize the data pipeline with train data
        sess.run(iterator.initializer, feed_dict = {input_X: X_train,
                                                    input_y: y_train,
                                                    batch_size: train_batch_size})
        
        # train the model
        train_loss = 0
        n_batches = 0
        while True:
            try:
                _, loss_op, fea = sess.run((training_op, loss, features))
                train_loss += loss_op
                n_batches += 1
            except tf.errors.OutOfRangeError:
                break
        
        if steps % display == 0:
            # evaluate the loss
            train_loss = train_loss/n_batches            
            
            # initialize the data pipeline with test data
            sess.run(iterator.initializer, feed_dict = {input_X: X_test,
                                                        input_y: y_test,
                                                        batch_size: test_dataset_size})
            test_loss, fea_test = sess.run((loss, features))
            
            print('Step: {} - Train loss: {:.2f} \t Test loss: {:.2f}'.format(
                    steps, train_loss, test_loss))
    
    # report accuracy for training and test data
    
    ## initialize the data pipeline with train data
    sess.run(iterator.initializer, feed_dict = {input_X: X_train,
                                                input_y: y_train,
                                                batch_size: tain_dataset_size})
    acc, train_fea = sess.run((accuracy, features))
    print('\nTraining set (accuracy): {}'.format(acc[1]))
    print(train_fea.shape)
    
    ## initialize the data pipeline with test data
    sess.run(iterator.initializer, feed_dict = {input_X: X_test,
                                                input_y: y_test,
                                                batch_size: test_dataset_size})
    acc, test_fea = sess.run((accuracy, features))
    print('Test set (accuracy): {}'.format(acc[1]))
    print(test_fea.shape)

'Output':
Step: 0 - Train loss: 2.30       Test loss: 1.75
Step: 100 - Train loss: 0.47     Test loss: 0.50
Step: 200 - Train loss: 0.37     Test loss: 0.39
Step: 300 - Train loss: 0.31     Test loss: 0.33
Step: 400 - Train loss: 0.26     Test loss: 0.27
Step: 500 - Train loss: 0.23     Test loss: 0.23
Step: 600 - Train loss: 0.21     Test loss: 0.21
Step: 700 - Train loss: 0.18     Test loss: 0.18
Step: 800 - Train loss: 0.17     Test loss: 0.16
Step: 900 - Train loss: 0.15     Test loss: 0.15

Training set (accuracy): 0.9910714030265808
(112, 4)
Test set (accuracy): 0.9866666793823242
(38, 4)
```

From the preceding code listing, take note of the following steps and functions:
- Observe the code for creating a pipeline using the Dataset API. The method `tf.data.Dataset.from_tensor_slices()` is used to create a Dataset from Tensor elements.
- The Dataset method `shuffle()` shuffles the Dataset at each epoch.
- The Dataset mehtod `batch()` feeds the Data elements in mini-batches.
- The batch size of the Dataset pipeline is controlled by the `batch_size` placeholder which is modified at runtime through the `feed_dict` attribute.
- This example of working with the Dataset API makes use of an initializable iterator `make_initializable_iterator()` to dynamically change the source of the Dataset at runtime. Hence, when running in Session we can initialize the iterator to create a Dataset pipeline off the training or evaluation dataset.

### TensorFlow High-Level APIs: Using Estimators
This section will provide examples of using the High-Level TensorFlow Estimator API both with the premade Estimators as well as writing a custom Estimator. Estimators are the preffered means for building a TensorFlow model primarily because the code can easily run on CPUs, GPUs or TPUs without any model modification. In working with Estimators, the data input pipeline is separated from the model architecture, this way it is easier to carry out experimentation on the dataset input.

#### Using the Pre-Made or Canned Estimator
The following steps are typically followed when working with a Premade Estimators:
1. Write the `input_fn` to handle the data pipeline.
2. Define the type of data attributes into the model using feature columns `tf.feature_column`.
3. Instantiate one of the pre-made Estimators by passing in the feature columns and other relevant attributes.
4. Use the `train()`, `evaluate()` and `predict()` methods to train, evaluate the model on evaluation dataset and use the model to make prediction/ inference.

Let's see a simple example of working with a TensorFlow premade Estimator again using the Iris dataset as in the previous example.

```python
# import packages
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

# load dataset
data = datasets.load_iris()

# separate features and target
X = data.data
y = data.target

# split in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

# create an input_fn
def input_fn(X, y, batch_size=30, training=True):
    # convert to dictionary
    X = {'sepal_length': X[:,0],
         'sepal_width':  X[:,1],
         'petal_length': X[:,2],
         'petal_width':  X[:,3]}
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()    
    return features, labels

# use feature columns to define the attributes to the model
sepal_length = tf.feature_column.numeric_column('sepal_length')
sepal_width = tf.feature_column.numeric_column('sepal_width')
petal_length = tf.feature_column.numeric_column('petal_length')
petal_width = tf.feature_column.numeric_column('petal_width')

feature_columns = [sepal_length, sepal_width, petal_length, petal_width]

# instantiate a DNNLinearCombinedClassifier Estimator
estimator = tf.estimator.DNNLinearCombinedClassifier(
        dnn_feature_columns=feature_columns,
        dnn_optimizer='Adam',
        dnn_hidden_units=[20],
        dnn_activation_fn=tf.nn.relu,
        n_classes=3
    )

# train model
estimator.train(input_fn=lambda:input_fn(X_train, y_train), steps=2000)
# evaluate model
metrics = estimator.evaluate(input_fn=lambda:input_fn(X_test, y_test, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**metrics))

'Output':
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into /var/folders/gh/mqsbbqy55bb4z763xxgddw780000gn/T/tmpop_v6qgk/model.ckpt.
INFO:tensorflow:loss = 46.064186, step = 1
INFO:tensorflow:global_step/sec: 605.83
INFO:tensorflow:loss = 26.527525, step = 101 (0.166 sec)
INFO:tensorflow:global_step/sec: 968.946
INFO:tensorflow:loss = 20.685375, step = 201 (0.103 sec)
INFO:tensorflow:global_step/sec: 1039.1
INFO:tensorflow:loss = 16.046446, step = 301 (0.096 sec)
...
INFO:tensorflow:global_step/sec: 1083.04
INFO:tensorflow:loss = 3.4552207, step = 1801 (0.092 sec)
INFO:tensorflow:global_step/sec: 1062.24
INFO:tensorflow:loss = 4.109541, step = 1901 (0.094 sec)
INFO:tensorflow:Saving checkpoints for 2000 into /var/folders/gh/mqsbbqy55bb4z763xxgddw780000gn/T/tmpop_v6qgk/model.ckpt.
INFO:tensorflow:Loss for final step: 2.134757.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2018-08-13-11:49:42
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from /var/folders/gh/mqsbbqy55bb4z763xxgddw780000gn/T/tmpop_v6qgk/model.ckpt-2000
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2018-08-13-11:49:42
INFO:tensorflow:Saving dict for global step 2000: accuracy = 1.0, average_loss = 0.07314519, global_step = 2000, loss = 1.3897586
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 2000: /var/folders/gh/mqsbbqy55bb4z763xxgddw780000gn/T/tmpop_v6qgk/model.ckpt-2000

Test set accuracy: 1.000
```

#### Building a Custom Estimator
The difference between the premade Estimator and the custom Estimator is that in the former, the `model_fn` that encapsulates methods for constructing the training/ inference graphs have already been created whereas in the latter the `model_fn` will have to be written. To work with Custom Estimators we provide:
1. An input data pipeline `input_fn`, and
2. A model function `model_fn` with modes to implement the:
   - train(),
   - evaluate(), and
   - predict() methods

Let's see an example of a Custom Estimator. The MLP MNIST example provided earlier in this text is converted to work with the Estimator API.

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# clear graph (if any) before running
tf.reset_default_graph()

# download data into current directory
data = input_data.read_data_sets('./tmp/mnist_data', one_hot=True)

# split data int training and evaluation sets
trainX = data.train.images
trainY = data.train.labels
testX = data.test.images
testY = data.test.labels

# create an input_fn
def input_fn(X, y, batch_size=100, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()    
    return features, labels

# create model_fn
def _mlp_model(features, labels, mode, params):
    # hidden layer
    hidden1 = tf.layers.dense(inputs=features, units=params['hidden_unit'],
                              activation=tf.nn.relu, name='hidden1')
    # dropout layer
    if mode == tf.estimator.ModeKeys.TRAIN:
        dropout1 = tf.layers.dropout(inputs=hidden1, rate=params['dropout_rate'])
    else:
        dropout1 = tf.layers.dropout(inputs=hidden1, rate=1.0)
    # output layer
    logits = tf.layers.dense(inputs=dropout1, units=10,
                             activation=tf.nn.relu, name='output')    
    return logits

def mlp_custom_estimator(features, labels, mode, params):
    logits = _mlp_model(features, labels, mode, params)
    
    predictions = {
      "classes": tf.argmax(input=logits, axis=1), # result classes
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor") # class probabilities
    }
    
    # Prediction Mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Loss Function
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                           logits=logits)
    
    # Learning Rate Decay (Exponential)
    learning_rate = tf.train.exponential_decay(learning_rate=1e-04,
                                               global_step=tf.train.get_global_step(),
                                               decay_steps=10000, 
                                               decay_rate=0.96, 
                                               staircase=True,
                                               name='lr_exp_decay')
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    
    # Training mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # Evaluation mode
    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels,1),
                                   predictions=predictions['classes'])
    precision = tf.metrics.precision(labels=tf.argmax(labels,1),
                                     predictions=predictions['classes'])
    recall = tf.metrics.recall(labels=tf.argmax(labels,1),
                               predictions=predictions['classes'])
    
    eval_metric_ops = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    }
    
    # TensorBoard Summary
    tf.summary.scalar('Accuracy', accuracy[1])
    tf.summary.scalar('Precision', precision[1])
    tf.summary.scalar('Recall', recall[1])
    tf.summary.histogram('Probabilities', predictions['probabilities'])
    tf.summary.histogram('Classes', predictions['classes'])
    
    summary_hook = tf.train.SummarySaverHook(summary_op=tf.summary.merge_all(),
                                             save_steps=1)
    
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions['classes'],
                                      train_op = train_op,
                                      loss = loss, 
                                      eval_metric_ops = eval_metric_ops,
                                      training_hooks=[summary_hook])

# build model    
estimator = tf.estimator.Estimator(mlp_custom_estimator,
                            model_dir='./mlp_custom_estimator',
                            config=tf.estimator.RunConfig(save_summary_steps=1),
                            params = {'hidden_unit': 256,
                                      'dropout_rate': 0.5}
                            )

# train model
estimator.train(input_fn=lambda:input_fn(trainX, trainY), steps=2000)

# evaluate model
metrics = estimator.evaluate(input_fn=lambda:input_fn(testX, testY, training=False))

print('\nTest set Accuracy score: {Accuracy:0.3f}'.format(**metrics))
print('Precision score: {Precision:0.3f}'.format(**metrics))
print('Recall score: {Recall:0.3f}'.format(**metrics))

'Output':
INFO:tensorflow:Using config: {'_model_dir': './mlp_custom_estimator', '_tf_random_seed': None, '_save_summary_steps': 1, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x1c2c411dd8>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into ./mlp_custom_estimator/model.ckpt.
INFO:tensorflow:loss = 2.3436866, step = 1
INFO:tensorflow:global_step/sec: 7.27846
INFO:tensorflow:loss = 1.3396419, step = 101 (13.743 sec)
INFO:tensorflow:global_step/sec: 318.577
INFO:tensorflow:loss = 0.76794, step = 201 (0.307 sec)
INFO:tensorflow:global_step/sec: 334.916
...
INFO:tensorflow:global_step/sec: 356.958
INFO:tensorflow:loss = 0.24392787, step = 1901 (0.280 sec)
INFO:tensorflow:Saving checkpoints for 2000 into ./mlp_custom_estimator/model.ckpt.
INFO:tensorflow:Loss for final step: 0.24159643.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2018-08-14-15:59:41
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from ./mlp_custom_estimator/model.ckpt-2000
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2018-08-14-15:59:43
INFO:tensorflow:Saving dict for global step 2000: Accuracy = 0.9362, Precision = 0.99821746, Recall = 0.9933481, global_step = 2000, loss = 0.22996733
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 2000: ./mlp_custom_estimator/model.ckpt-2000

Test set Accuracy score: 0.936
Precision score: 0.998
Recall score: 0.993
```

### Eager Execution
Eager Execution enables instant evaluation of TensorFlow operations as opposed to first constructing a computational graph and executing it in a Session. This feature allows for rapid development, experimentation and debugging of TensorFlow models. To enable Eager execution, run the following code at the beginning of the program or console session:

```python
import tensorflow as tf

# enable eager execution
tf.enable_eager_execution()

a = tf.constant(6, name='a')
b = tf.constant(3, name='b')

print('Addition: a + b = {}'.format(tf.add(a, b)))
'Output': Addition: a + b = 9

print('Multiply: a x b = {}'.format(tf.multiply(a, b)))
'Output': Multiply: a x b = 18

print('Divide: a / b = {}'.format(tf.divide(a, b)))
'Output': Divide: a / b = 2.0
```

It is important to note that once Eager execution is enabled, it cannot be disabled except the terminal or session is restarted. More about modeling with Eager Execution will be discussed in the Keras Chapter.