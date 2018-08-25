---
layout: page-seminar
title: 'Keras'
permalink: gcp-ml-seminar/keras/
---

Keras provides a high-level API abstraction for developing deep neural network models. It uses as of this time of writing, TensorFlow, CNTK and Theano at the backend. The API was initially separate from TensorFlow and only provided a simple interface for model building with TensorFlow as one of the frameworks running at the backend. However, from TensorFlow 1.2, Keras is now an integral part of the TensorFlow codebase as a high-level API.

The Keras API version internal to TensorFlow is available from the `tf.keras` package. Whereas, the broader Keras API blueprint that is not tied to a specific backend will remain available from the `keras` package, and can be installed by running:

```bash
# from the terminal
pip install keras
```
or,

```bash
# on a Datalab notebook
!pip install keras
```

In summary, when working with the `keras` package, the backend can run with either TensorFlow, Microsoft CNTK or Theano. On the other hand, working with `tf.keras` provides a TensorFlow only version which is tightly integrated and compatible with the all of the functionality of the core TensorFlow library.

In this Chapter, we will work exclusively with the `tf.keras` TensorFlow package.

### The Anatomy of a Keras Program
ABCD

### Multilayer Perceptron (MLP) with Keras
ABCD

### Model Visualization with Keras
ABCD

### Saving and Loading Models with Keras
ABCD

### Checkpointing to Select Best Models
ABCD

### Convolutional Neural Networks (CNNs) with Keras
ABCD

### Recurrent Neural Networks (RNNs) with Keras
ABCD

#### Stacked LSTM
ABCD

#### CNN LSTM
ABCD

#### Encoder-Decoder LSTM
ABCD

#### Bidirectional LSTM
ABCD
