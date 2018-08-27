---
layout: page-seminar
title: 'Keras'
permalink: gcp-ml-seminar/keras/
---

Table of contents:

- [The Anatomy of a Keras Program](#the-anatomy-of-a-keras-program)
- [Multilayer Perceptron (MLP) with Keras](#multilayer-perceptron-mlp-with-keras)
    - [Create a Dataset pipeline](#create-a-dataset-pipeline)
- [Model Visualization with Keras](#model-visualization-with-keras)
- [Saving and Loading Models with Keras](#saving-and-loading-models-with-keras)
- [Checkpointing to Select Best Models](#checkpointing-to-select-best-models)
- [Convolutional Neural Networks (CNNs) with Keras](#convolutional-neural-networks-cnns-with-keras)
- [Recurrent Neural Networks (RNNs) with Keras](#recurrent-neural-networks-rnns-with-keras)
    - [Stacked LSTM](#stacked-lstm)
    - [CNN LSTM](#cnn-lstm)
    - [Encoder-Decoder LSTM](#encoder-decoder-lstm)
    - [Bidirectional LSTM](#bidirectional-lstm)

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
The Keras `Model` forms the core of a Keras programme. A `Model` is first constructed, then it is compiled. Next, the compiled model is trained and evaluated using their respective training and evaluation datasets. Upon successful evaluation using the relevant metrics, the model is then used for making predictions on previously unseen data samples. Figure 1 shows the program flow for modeling with Keras.

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/keras-program.png"> <!--width="50%" height="50%"-->
    <div class="figcaption" style="text-align: center;">
        Figure 1: The Anatomy of a Keras Program
    </div>
</div>

As shown in Figure 1, the Keras `Model` can be constructued using the Sequential API `tf.keras.Sequential` or the Keras Functional API which defines a model instance `tf.keras.Model`. The Sequential model is the simplest method for creating a linear stack of Neural Networks layers. The Functional model is used if a more complex graph is desired.


### Multilayer Perceptron (MLP) with Keras
In this section, we examine a motivating example by building a Keras MLP model using both the sequential and functional APIs.

In doing so, we'll go through the following steps:
- Import and transform the dataset.
- Build and Compile the Model
- Train the data using `Model.fit()`
- Evaluate the Model using `Model.evaluate()`
- Predict on unseen data using `Model.predict()`

The dataset used for this example is the Fashion-MNIST database of fashion articles. Similar to the MNIST handwriting dataset, this dataset contains 60,000 28x28 pixel grayscale images. This dataset and more can be found in the `tf.keras.datasets` package.

```python
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

# import dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# flatten the 28*28 pixel images into one long 784 pixel vector
x_train = np.reshape(x_train, (-1, 784)).astype('float32')
x_test = np.reshape(x_test, (-1, 784)).astype('float32')

# scale dataset from 0 -> 255 to 0 -> 1
x_train /= 255
x_test /= 255

# one-hot encode targets
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
 
# create the model
def model_fn():
    model = keras.Sequential()
    # Adds a densely-connected layer with 256 units to the model:
    model.add(keras.layers.Dense(256, activation='relu', input_dim=784))
    # Add Dropout layer
    model.add(keras.layers.Dropout(0.2))
    # Add another densely-connected layer with 64 units:
    model.add(keras.layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # compile the model
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# build model
model = model_fn()

# train the model
model.fit(x_train, y_train, epochs=10,
          batch_size=100, verbose=1,
          validation_data=(x_test, y_test))

# evaluate the model
score = model.evaluate(x_test, y_test, batch_size=100)
print('Test loss: {:.2f} \nTest accuracy: {:.2f}%'.format(score[0], score[1]*100))

'Output':
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
60000/60000 [==============================] - 254s 4ms/step - loss: 0.5429 - acc: 0.8086 - val_loss: 0.4568 - val_acc: 0.8352
Epoch 2/10
60000/60000 [==============================] - 4s 72us/step - loss: 0.3910 - acc: 0.8566 - val_loss: 0.3884 - val_acc: 0.8568
Epoch 3/10
60000/60000 [==============================] - 4s 67us/step - loss: 0.3557 - acc: 0.8700 - val_loss: 0.3909 - val_acc: 0.8506
Epoch 4/10
60000/60000 [==============================] - 4s 65us/step - loss: 0.3324 - acc: 0.8773 - val_loss: 0.3569 - val_acc: 0.8718
Epoch 5/10
60000/60000 [==============================] - 5s 77us/step - loss: 0.3202 - acc: 0.8812 - val_loss: 0.3504 - val_acc: 0.8684
Epoch 6/10
60000/60000 [==============================] - 5s 75us/step - loss: 0.3035 - acc: 0.8877 - val_loss: 0.3374 - val_acc: 0.8776
Epoch 7/10
60000/60000 [==============================] - 4s 66us/step - loss: 0.2950 - acc: 0.8908 - val_loss: 0.3337 - val_acc: 0.8786
Epoch 8/10
60000/60000 [==============================] - 4s 68us/step - loss: 0.2854 - acc: 0.8939 - val_loss: 0.3320 - val_acc: 0.8801
Epoch 9/10
60000/60000 [==============================] - 4s 65us/step - loss: 0.2771 - acc: 0.8965 - val_loss: 0.3238 - val_acc: 0.8842
Epoch 10/10
60000/60000 [==============================] - 4s 73us/step - loss: 0.2680 - acc: 0.8986 - val_loss: 0.3252 - val_acc: 0.8853
10000/10000 [==============================] - 0s 21us/step

Test loss: 0.33 
Test accuracy: 88.35%
```

#### Create a Dataset pipeline


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
