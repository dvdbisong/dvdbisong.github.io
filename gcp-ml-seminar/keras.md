---
layout: page-seminar
title: 'Keras'
permalink: gcp-ml-seminar/keras/
---

Table of contents:

- [The Anatomy of a Keras Program](#the-anatomy-of-a-keras-program)
- [Multilayer Perceptron (MLP) with Keras](#multilayer-perceptron-mlp-with-keras)
- [Using the Dataset API with tf.keras](#using-the-dataset-api-with-tfkeras)
- [Model Visualization with Keras](#model-visualization-with-keras)
- [TensorBoard with Keras](#tensorboard-with-keras)
- [Checkpointing to Select Best Models](#checkpointing-to-select-best-models)
- [Convolutional Neural Networks (CNNs) with Keras](#convolutional-neural-networks-cnns-with-keras)
- [Recurrent Neural Networks (RNNs) with Keras](#recurrent-neural-networks-rnns-with-keras)
  - [Stacked LSTM](#stacked-lstm)
- [Long-term Recurrent Convolutional Network (CNN LSTM)](#long-term-recurrent-convolutional-network-cnn-lstm)
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

As shown in Figure 1, the Keras `Model` can be constructed using the Sequential API `tf.keras.Sequential` or the Keras Functional API which defines a model instance `tf.keras.Model`. The Sequential model is the simplest method for creating a linear stack of Neural Networks layers. The Functional model is used if a more complex graph is desired.


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
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 256 units to the model:
    model.add(tf.keras.layers.Dense(256, activation='relu', input_dim=784))
    # Add Dropout layer
    model.add(tf.keras.layers.Dropout(0.2))
    # Add another densely-connected layer with 64 units:
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
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

From the code above, observe the following key methods:
- A Keras Sequential Model is built by calling the `tf.keras.Sequential()` method from which layers are then added to the model.
- After constructing the model layers, the model is compiled by calling the method `.compile()`.
- The model is trained by calling the `.fit()` method which receives the training features `x_train` and corresponding labels `y_train` dataset. The attribute `validation_data` used the evaluation training and test split to show the performance of the training algorithm at every training epoch.
- The method `.evaluate()` is used to get the final metric estimate and the loss score of the model after training.
- The optimizer `tf.train.AdamOptimizer()` is an example of using a TensorFlow optimizer and frankly other TensorFlow functions with Keras.

### Using the Dataset API with tf.keras
In this section will use the TensorFlow Dataset API to build a data pipeline for feeding data into a Keras Model. Using the Dataset API is the preferred mechanism for feeding data as it can easily scale to handle very large datasets and better utilize the machine resources. Again, here, we use the Fashion MNIST dataset as an example, but this time a data pipeline is constructed using the Dataset API and the Model is constructed using the Keras Functional API.

```python
import tensorflow as tf
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

# create dataset pipeline
def input_fn(features, labels, batch_size, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()    
    return features, labels

# parameters
batch_size = 100
training_steps_per_epoch = int(np.ceil(x_train.shape[0] / float(batch_size)))  # ==> 600
eval_steps_per_epoch = int(np.ceil(x_test.shape[0] / float(batch_size)))  # ==> 100
epochs = 10

# create the model
def model_fn(input_fn):
    
    (features, labels) = input_fn
    
    # Model input
    model_input = tf.keras.layers.Input(tensor=features)
    # Adds a densely-connected layer with 256 units to the model:
    x = tf.keras.layers.Dense(256, activation='relu', input_dim=784)(model_input)
    # Add Dropout layer:
    x = tf.keras.layers.Dropout(0.2)(x)
    # Add another densely-connected layer with 64 units:
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    # Add a softmax layer with 10 output units:
    predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    # the model
    model = tf.keras.Model(inputs=model_input, outputs=predictions)
    
    # compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  target_tensors=[tf.cast(labels,tf.float32)])
    return model

# build train model
model = model_fn(input_fn(x_train, y_train, batch_size=batch_size, training=True))

# print train model summary
model.summary()
```
```bash
'Output':
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 784)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               200960    
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 64)                16448     
_________________________________________________________________
dense_5 (Dense)              (None, 10)                650       
=================================================================
Total params: 218,058
Trainable params: 218,058
Non-trainable params: 0
_________________________________________________________________
```
```python
# train the model
history = model.fit(epochs=epochs,
                    steps_per_epoch=training_steps_per_epoch)
```
```bash
'Output':
Epoch 1/10
600/600 [==============================] - 3s 5ms/step - loss: 0.2634 - acc: 0.9005
Epoch 2/10
600/600 [==============================] - 4s 6ms/step - loss: 0.2522 - acc: 0.9050
Epoch 3/10
600/600 [==============================] - 4s 7ms/step - loss: 0.2471 - acc: 0.9070
Epoch 4/10
600/600 [==============================] - 4s 6ms/step - loss: 0.2439 - acc: 0.9082
Epoch 5/10
600/600 [==============================] - 4s 7ms/step - loss: 0.2388 - acc: 0.9090
Epoch 6/10
600/600 [==============================] - 4s 6ms/step - loss: 0.2330 - acc: 0.9112
Epoch 7/10
600/600 [==============================] - 3s 6ms/step - loss: 0.2253 - acc: 0.9140
Epoch 8/10
600/600 [==============================] - 3s 6ms/step - loss: 0.2252 - acc: 0.9157
Epoch 9/10
600/600 [==============================] - 3s 6ms/step - loss: 0.2209 - acc: 0.9166
Epoch 10/10
600/600 [==============================] - 3s 5ms/step - loss: 0.2146 - acc: 0.9177
```
```python
# store trained model weights
model.save_weights('/tmp/mlp_weight.h5')

# build evaluation model
eval_model = model_fn(input_fn(x_test, y_test, batch_size=batch_size, training=False))
# load saved weights to evaluation model
eval_model.load_weights('/tmp/mlp_weight.h5')

# evaluate the model
score = eval_model.evaluate(steps=eval_steps_per_epoch)
print('Test loss: {:.2f} \nTest accuracy: {:.2f}%'.format(score[0], score[1]*100))
```
```bash
'Output':
Test loss: 0.32 
Test accuracy: 89.30%
```

From the code block above, observe the following steps:
- The Keras Functional API is used to construct the model in the custom `model_fn` function. Note how the layers are constructed from the Input `tf.keras.layers.Input` to the output. Also observe how the model is built using `tf.keras.Model`.
- After training the model using `model.fit()`, the weights of the model are saved to a Hierarchical Data Format (HDF5) file with the extension `.h5` by calling `model.save_weights(/save_path)`.
- Using the model function `model_fn`, we create a new model instance, this time using the evaluation dataset. The weights of the trained model are loaded into the evaluation model by calling `eval_model.load_weights(/save_path)`.
- The variable `history` stores the metrics for the model at each time epoch returned by the callback function of the method `.fit()`.

### Model Visualization with Keras
With Keras, it is quite easy and straightforward to plot the metrics of the model to have a better graphical perspective as to how the model is performing for every training epoch. This view is also useful for dealing with issues of bias or variance of the model.

A callback function of the `model.fit()` method returns the loss and evaluation score for each epoch. This information is stored in a variable and plotted.

In this example, to illustrate model visualization with Keras, we build an MLP network using the Keras Functional API to classify the MNIST handwriting dataset.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(128, activation='relu', input_dim=784)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    prediction = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=prediction)
    
    # compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# build model
model = model_fn()

# train the model
history = model.fit(x_train, y_train, epochs=10,
                    batch_size=100, verbose=1,
                    validation_data=(x_test, y_test))

# evaluate the model
score = model.evaluate(x_test, y_test, batch_size=100)
print('Test loss: {:.2f} \nTest accuracy: {:.2f}%'.format(score[0], score[1]*100))
```

```bash
'Output':
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
60000/60000 [==============================] - 36s 599us/step - loss: 0.3669 - acc: 0.8934 - val_loss: 0.1498 - val_acc: 0.9561
Epoch 2/10
60000/60000 [==============================] - 2s 27us/step - loss: 0.1572 - acc: 0.9532 - val_loss: 0.1071 - val_acc: 0.9661
Epoch 3/10
60000/60000 [==============================] - 1s 24us/step - loss: 0.1154 - acc: 0.9648 - val_loss: 0.0852 - val_acc: 0.9744
Epoch 4/10
60000/60000 [==============================] - 1s 24us/step - loss: 0.0949 - acc: 0.9707 - val_loss: 0.0838 - val_acc: 0.9724
Epoch 5/10
60000/60000 [==============================] - 1s 22us/step - loss: 0.0807 - acc: 0.9752 - val_loss: 0.0754 - val_acc: 0.9772
Epoch 6/10
60000/60000 [==============================] - 1s 22us/step - loss: 0.0721 - acc: 0.9766 - val_loss: 0.0712 - val_acc: 0.9774
Epoch 7/10
60000/60000 [==============================] - 1s 22us/step - loss: 0.0625 - acc: 0.9798 - val_loss: 0.0694 - val_acc: 0.9776
Epoch 8/10
60000/60000 [==============================] - 1s 23us/step - loss: 0.0575 - acc: 0.9808 - val_loss: 0.0692 - val_acc: 0.9782
Epoch 9/10
60000/60000 [==============================] - 1s 24us/step - loss: 0.0508 - acc: 0.9832 - val_loss: 0.0687 - val_acc: 0.9785
Epoch 10/10
60000/60000 [==============================] - 1s 23us/step - loss: 0.0467 - acc: 0.9844 - val_loss: 0.0716 - val_acc: 0.9786
10000/10000 [==============================] - 0s 9us/step
Test loss: 0.07 
Test accuracy: 97.86%
```
```python
# list metrics returned from callback function
history.history.keys()
```
```bash
'Output':
dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
```
```python
# plot loss metric
plt.figure(1)
plt.plot(history.history['loss'], '--')
plt.plot(history.history['val_loss'], '--')
plt.title('Model loss per epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'evaluation'])
```
<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/keras_visualize_loss_metric.png" width="70%" height="70%">
    <div class="figcaption" style="text-align: center;">
        Figure 2: Model loss per epoch.
    </div>
</div>

```python
# plot accuracy metric
plt.figure(2)
plt.plot(history.history['acc'], '--')
plt.plot(history.history['val_acc'], '--')
plt.title('Model accuracy per epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'evaluation'])
```
<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/keras_visualize_accuracy_metric.png" width="70%" height="70%">
    <div class="figcaption" style="text-align: center;">
        Figure 3 Model accuracy per epoch.
    </div>
</div>

### TensorBoard with Keras
To visualize models with TensorBoard, attach a TensorBoard callback `tf.keras.callbacks.TensorBoard()` to the `model.fit()` method before training the model. The model graph, scalars, histograms, and other metrics are stored as event files in the log directory.

For this example, we modify the MLP MNIST model to use TensorBoard.
```python
import tensorflow as tf
import numpy as np

# import dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(128, activation='relu', input_dim=784)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    prediction = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=prediction)
    
    # compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# build model
model = model_fn()

# checkpointing
best_model = tf.keras.callbacks.ModelCheckpoint('./tmp/mnist_weights.h5', monitor='val_acc',
                                                verbose=1, save_best_only=True, mode='max')

# tensorboard
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./tmp/logs_mnist_keras',
                                             histogram_freq=0, write_graph=True,
                                             write_images=True)

callbacks = [best_model, tensorboard]

# train the model
history = model.fit(x_train, y_train, epochs=10,
                    batch_size=100, verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks)

# evaluate the model
score = model.evaluate(x_test, y_test, batch_size=100)
print('Test loss: {:.2f} \nTest accuracy: {:.2f}%'.format(score[0], score[1]*100))
```

Execute the command to run TensorBoard.
```bash
# via terminal
tensorboard --logdir tmp/logs_mnist_keras
```
```python
# via Datalab cell
tensorboard_pid = ml.TensorBoard.start('tmp/logs_mnist_keras')

# After use, close the TensorBoard instance by running:
ml.TensorBoard.stop(tensorboard_pid)
```

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/keras_tensorboard.png">
    <div class="figcaption" style="text-align: center;">
        Figure 4: TensorBoard with Keras.
    </div>
</div>

### Checkpointing to Select Best Models
Checkpointing makes it possible to save the weights of the neural network model when there is an increase in the validation accuracy metric. This is achieved in Keras using the method `tf.keras.callbacks.ModelCheckpoint()`. The saved weights can then be loaded back into the model and used to make predictions. Using the MNIST dataset, we'll build a model that saves the weights to file only when there is an improvement in the validation set performance.

```python
import tensorflow as tf
import numpy as np

# import dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(128, activation='relu', input_dim=784)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    prediction = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=prediction)
    
    # compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# build model
model = model_fn()

# checkpointing
checkpoint = tf.keras.callbacks.ModelCheckpoint('mnist_weights.h5', monitor='val_acc',
                                                verbose=1, save_best_only=True, mode='max')
callbacks = [checkpoint]

# train the model
history = model.fit(x_train, y_train, epochs=10,
                    batch_size=100, verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks)

# evaluate the model
score = model.evaluate(x_test, y_test, batch_size=100)
print('Test loss: {:.2f} \nTest accuracy: {:.2f}%'.format(score[0], score[1]*100))
```
```bash
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
60000/60000 [==============================] - 39s 653us/step - loss: 0.3605 - acc: 0.8947 - val_loss: 0.1444 - val_acc: 0.9565

Epoch 00001: val_acc improved from -inf to 0.95650, saving model to mnist_weights.h5
Epoch 2/10
60000/60000 [==============================] - 2s 38us/step - loss: 0.1549 - acc: 0.9533 - val_loss: 0.1124 - val_acc: 0.9668

Epoch 00002: val_acc improved from 0.95650 to 0.96680, saving model to mnist_weights.h5
Epoch 3/10
60000/60000 [==============================] - 2s 40us/step - loss: 0.1178 - acc: 0.9639 - val_loss: 0.0866 - val_acc: 0.9721

Epoch 00003: val_acc improved from 0.96680 to 0.97210, saving model to mnist_weights.h5
Epoch 4/10
60000/60000 [==============================] - 2s 33us/step - loss: 0.0925 - acc: 0.9720 - val_loss: 0.0837 - val_acc: 0.9748

Epoch 00004: val_acc improved from 0.97210 to 0.97480, saving model to mnist_weights.h5
Epoch 5/10
60000/60000 [==============================] - 2s 35us/step - loss: 0.0825 - acc: 0.9744 - val_loss: 0.0751 - val_acc: 0.9765

Epoch 00005: val_acc improved from 0.97480 to 0.97650, saving model to mnist_weights.h5
Epoch 6/10
60000/60000 [==============================] - 2s 31us/step - loss: 0.0711 - acc: 0.9775 - val_loss: 0.0707 - val_acc: 0.9765

Epoch 00006: val_acc did not improve from 0.97650
Epoch 7/10
60000/60000 [==============================] - 2s 34us/step - loss: 0.0621 - acc: 0.9804 - val_loss: 0.0705 - val_acc: 0.9781

Epoch 00007: val_acc improved from 0.97650 to 0.97810, saving model to mnist_weights.h5
Epoch 8/10
60000/60000 [==============================] - 2s 31us/step - loss: 0.0570 - acc: 0.9810 - val_loss: 0.0709 - val_acc: 0.9781

Epoch 00008: val_acc improved from 0.97810 to 0.97810, saving model to mnist_weights.h5
Epoch 9/10
60000/60000 [==============================] - 2s 32us/step - loss: 0.0516 - acc: 0.9835 - val_loss: 0.0719 - val_acc: 0.9779

Epoch 00009: val_acc did not improve from 0.97810
Epoch 10/10
60000/60000 [==============================] - 2s 32us/step - loss: 0.0467 - acc: 0.9851 - val_loss: 0.0730 - val_acc: 0.9784

Epoch 00010: val_acc improved from 0.97810 to 0.97840, saving model to mnist_weights.h5
10000/10000 [==============================] - 0s 11us/step
Test loss: 0.07 
Test accuracy: 97.84%
```

### Convolutional Neural Networks (CNNs) with Keras
In this section, we use Keras to implement a Convolutional Neural Network. The network architecture is similar to the Krizhevsky architecture implemented in the TensorFlow chapter and consists of the following layers:
- Convolution layer: kernel_size => [5 x 5]
- Convolution layer: kernel_size => [5 x 5]
- Batch Normalization layer
- Convolution layer: kernel_size => [5 x 5]
- Max pooling: pool size => [2 x 2]
- Convolution layer: kernel_size => [5 x 5]
- Convolution layer: kernel_size => [5 x 5]
- Batch Normalization layer
- Max pooling: pool size => [2 x 2]
- Convolution layer: kernel_size => [5 x 5]
- Convolution layer: kernel_size => [5 x 5]
- Convolution layer: kernel_size => [5 x 5]
- Max pooling: pool size => [2 x 2]
- Dropout layer
- Dense Layer: units => [512]
- Dense Layer: units => [256]
- Dropout layer
- Dense Layer: units => [10]

The code listing is provided below.

```python
# import dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# change datatype to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# scale the dataset from 0 -> 255 to 0 -> 1
x_train /= 255
x_test /= 255

# one-hot encode targets
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# create dataset pipeline
def input_fn(features, labels, batch_size, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()    
    return features, labels

# parameters
batch_size = 100
training_steps_per_epoch = int(np.ceil(x_train.shape[0] / float(batch_size)))  # ==> 600
eval_steps_per_epoch = int(np.ceil(x_test.shape[0] / float(batch_size)))  # ==> 100
epochs = 10

# create the model
def model_fn(input_fn):
    
    (features, labels) = input_fn
    
    # Model input
    model_input = tf.keras.layers.Input(tensor=features)
    x = tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu')(model_input)
    x = tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    # the model
    model = tf.keras.Model(inputs=model_input, outputs=output)
    
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Nadam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  target_tensors=[tf.cast(labels,tf.float32)])
    return model

# build train model
model = model_fn(input_fn(x_train, y_train, batch_size=batch_size, training=True))

# print train model summary
model.summary()
```
```bash
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 32, 32, 3)         0         
_________________________________________________________________
conv2d (Conv2D)              (None, 32, 32, 64)        4864      
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 64)        102464    
_________________________________________________________________
batch_normalization (BatchNo (None, 32, 32, 64)        256       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 64)        102464    
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 64)        102464    
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 64)        102464    
_________________________________________________________________
batch_normalization_1 (Batch (None, 16, 16, 64)        256       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 32)          18464     
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 8, 8, 32)          9248      
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 8, 8, 32)          9248      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 32)          0         
_________________________________________________________________
dropout (Dropout)            (None, 4, 4, 32)          0         
_________________________________________________________________
flatten (Flatten)            (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 512)               262656    
_________________________________________________________________
dense_1 (Dense)              (None, 256)               131328    
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 848,746
Trainable params: 848,490
Non-trainable params: 256
_________________________________________________________________
```
```python
# train the model
history = model.fit(epochs=epochs,
                    steps_per_epoch=training_steps_per_epoch)
```
```bash
'Output':
Epoch 1/10
500/500 [==============================] - 485s 970ms/step - loss: 1.6918 - acc: 0.3785
Epoch 2/10
500/500 [==============================] - 473s 945ms/step - loss: 1.3357 - acc: 0.5217
Epoch 3/10
500/500 [==============================] - 480s 960ms/step - loss: 1.1156 - acc: 0.6092
Epoch 4/10
500/500 [==============================] - 491s 981ms/step - loss: 0.9740 - acc: 0.6623
Epoch 5/10
500/500 [==============================] - 482s 963ms/step - loss: 0.8796 - acc: 0.6977
Epoch 6/10
500/500 [==============================] - 482s 964ms/step - loss: 0.8153 - acc: 0.7187
Epoch 7/10
500/500 [==============================] - 469s 939ms/step - loss: 0.7583 - acc: 0.7409
Epoch 8/10
500/500 [==============================] - 466s 932ms/step - loss: 0.7126 - acc: 0.7576
Epoch 9/10
500/500 [==============================] - 473s 946ms/step - loss: 0.6749 - acc: 0.7699
Epoch 10/10
500/500 [==============================] - 472s 944ms/step - loss: 0.6459 - acc: 0.7798
```
```python
# store trained model weights
model.save_weights('./tmp/cnn_weight.h5')

# build evaluation model
eval_model = model_fn(input_fn(x_test, y_test, batch_size=batch_size, training=False))
eval_model.load_weights('./tmp/cnn_weight.h5')

# evaluate the model
score = eval_model.evaluate(steps=eval_steps_per_epoch)
print('Test loss: {:.2f} \nTest accuracy: {:.2f}%'.format(score[0], score[1]*100))
```
```bash
'Output':
100/100 [==============================] - 35s 349ms/step
Test loss: 0.83 
Test accuracy: 73.53%
```

From the code block above, observe how the network layers are implemented in Keras:
- Convolutional layer - `tf.keras.layers.Conv2D()`
- Batch Normalization - `tf.keras.layers.BatchNormalization()`
- Max Pooling layer - `tf.keras.layers.MaxPooling2D()`
- Dropout layer - `tf.keras.layers.Dropout()`
- Fully connected or Dense layer - `tf.keras.layers.Dense()`


### Recurrent Neural Networks (RNNs) with Keras
This section makes use of the Nigeria power consumption univariate timeseries dataset to implement a LSTM Recurrent Neural Network.

```python
import tensorflow as tf
import pandas as pd
import numpy as np
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
batch_size = 32

# create training and testing data
train_x = convert_to_sequences(data_train, time_steps, is_target=False)
train_y = convert_to_sequences(data_train, time_steps, is_target=True)

eval_x = convert_to_sequences(data_eval, time_steps, is_target=False)
eval_y = convert_to_sequences(data_eval, time_steps, is_target=True)

# Build model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2,
                               input_shape=train_x.shape[1:],
                               return_sequences=True))
model.add(tf.keras.layers.Dense(1))

# Compile the model
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])

# print model summary
model.summary()
```
```bash
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_2 (LSTM)                (None, 20, 128)           66560     
_________________________________________________________________
dense_2 (Dense)              (None, 20, 1)             129       
=================================================================
Total params: 66,689
Trainable params: 66,689
Non-trainable params: 0
_________________________________________________________________
```
```python
# Train the model
history = model.fit(train_x, train_y,
                    batch_size=batch_size,
                    epochs=20, shuffle=False,
                    validation_data=(eval_x, eval_y))
```
```bash
'Output':
Train on 78 samples, validate on 5 samples
Epoch 1/20
78/78 [==============================] - 2s 29ms/step - loss: 0.1068 - mean_squared_error: 0.1068 - val_loss: 0.1121 - val_mean_squared_error: 0.1121
Epoch 2/20
78/78 [==============================] - 0s 1ms/step - loss: 0.0522 - mean_squared_error: 0.0522 - val_loss: 0.0596 - val_mean_squared_error: 0.0596
Epoch 3/20
78/78 [==============================] - 0s 1ms/step - loss: 0.0264 - mean_squared_error: 0.0264 - val_loss: 0.0277 - val_mean_squared_error: 0.0277
...
Epoch 18/20
78/78 [==============================] - 0s 2ms/step - loss: 0.0259 - mean_squared_error: 0.0259 - val_loss: 0.0243 - val_mean_squared_error: 0.0243
Epoch 19/20
78/78 [==============================] - 0s 2ms/step - loss: 0.0296 - mean_squared_error: 0.0296 - val_loss: 0.0241 - val_mean_squared_error: 0.0241
Epoch 20/20
78/78 [==============================] - 0s 2ms/step - loss: 0.0156 - mean_squared_error: 0.0156 - val_loss: 0.0241 - val_mean_squared_error: 0.0241
```
```python
loss, mse = model.evaluate(eval_x, eval_y, batch_size=batch_size)
print('Test loss: {:.4f}'.format(loss))
print('Test mse: {:.4f}'.format(mse))

# predict
y_pred = model.predict(eval_x)

# plot predicted sequence
plt.title("Model Testing", fontsize=12)
plt.plot(eval_x[0,:,0], "b--", markersize=10, label="training instance")
plt.plot(eval_y[0,:,0], "g--", markersize=10, label="targets")
plt.plot(y_pred[0,:,0], "r--", markersize=10, label="model prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")
```

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/keras_rnn_ts_model_testing.png" width="70%" height="70%">
    <div class="figcaption" style="text-align: center;">
        Figure 5: Keras LSTM Model.
    </div>
</div>

From the Keras LSTM code listing, the method `tf.keras.layers.LSTM()` is used to implement the LSTM recurrent layer. The attribute `return_sequences` is set to `True` to return the full sequence in the output sequence.

#### Stacked LSTM
A Stacked LSTM is a deep RNN with multiple LSTM layers. This stacking of LSTM layers with memory cells makes the network more expressive, and can learn more complex long-running sequences. In this section, we use the Dow Jones Index dataset to show an example of building a deep LSTM network with Keras.

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
batch_size = int(train_X.shape[0]/5)
length = train_X.shape[0]

# Build model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2,
                               input_shape=train_X.shape[1:],
                               return_sequences=True))
model.add(tf.keras.layers.LSTM(100, return_sequences=True))
model.add(tf.keras.layers.LSTM(64))
model.add(tf.keras.layers.Dense(1))

# Compile the model
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])

# print model summary
model.summary()
```
```bash
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_50 (LSTM)               (None, 1, 128)            71168     
_________________________________________________________________
lstm_51 (LSTM)               (None, 1, 100)            91600     
_________________________________________________________________
lstm_52 (LSTM)               (None, 64)                42240     
_________________________________________________________________
dense_19 (Dense)             (None, 1)                 65        
=================================================================
Total params: 205,073
Trainable params: 205,073
Non-trainable params: 0
```
```python
# Train the model
history = model.fit(train_X, train_y,
                    batch_size=batch_size,
                    epochs=20, shuffle=False,
                    validation_data=(test_X, test_y))
```
```bash
Train on 11 samples, validate on 12 samples
Epoch 1/20
11/11 [==============================] - 10s 926ms/step - loss: 0.3354 - mean_squared_error: 0.3354 - val_loss: 0.1669 - val_mean_squared_error: 0.1669
Epoch 2/20
11/11 [==============================] - 0s 5ms/step - loss: 0.2917 - mean_squared_error: 0.2917 - val_loss: 0.1359 - val_mean_squared_error: 0.1359
Epoch 3/20
11/11 [==============================] - 0s 4ms/step - loss: 0.2355 - mean_squared_error: 0.2355 - val_loss: 0.0977 - val_mean_squared_error: 0.0977
...
Epoch 18/20
11/11 [==============================] - 0s 4ms/step - loss: 0.0436 - mean_squared_error: 0.0436 - val_loss: 0.0771 - val_mean_squared_error: 0.0771
Epoch 19/20
11/11 [==============================] - 0s 4ms/step - loss: 0.0378 - mean_squared_error: 0.0378 - val_loss: 0.0746 - val_mean_squared_error: 0.0746
Epoch 20/20
11/11 [==============================] - 0s 5ms/step - loss: 0.0405 - mean_squared_error: 0.0405 - val_loss: 0.0696 - val_mean_squared_error: 0.0696
12/12 [==============================] - 0s 933us/step
Test loss: 0.0696
Test mse: 0.0696
```
```python
loss, mse = model.evaluate(test_X, test_y, batch_size=batch_size)
print('Test loss: {:.4f}'.format(loss))
print('Test mse: {:.4f}'.format(mse))

# predict
y_pred = model.predict(test_X)

plt.figure(1)
plt.title("Keras - LSTM RNN Model Testing for '{}' stock".format(stock), fontsize=12)
plt.plot(test_y, "g--", markersize=10, label="targets")
plt.plot(y_pred, "r--", markersize=10, label="model prediction")
plt.legend()
plt.xlabel("Time")
```

<div class="fig figcenter fighighlight">
    <img src="/assets/seminar_IEEE/keras_lstm_rnn_ts_model_testing.png" width="70%" height="70%">
    <div class="figcaption" style="text-align: center;">
        Figure 6 Keras Stacked LSTM Model.
    </div>
</div>

### Long-term Recurrent Convolutional Network (CNN LSTM)

<!-- The CNN LSTM architecture involves using Convolutional Neural Network (CNN) layers for feature extraction on input data combined with LSTMs to support sequence prediction.

CNN LSTMs were developed for visual time series prediction problems and the application of generating textual descriptions from sequences of images (e.g. videos).

This architecture was originally referred to as a Long-term Recurrent Convolutional Network or LRCN model, although we will use the more generic name CNN LSTM to refer to LSTMs that use a CNN as a front end in this lesson.

This architecture is used for the task of generating textual descriptions of images. Key is the use of a CNN that is pre-trained on a challenging image classification task that is re-purposed as a feature extractor for the caption generating problem.

CNNs are used as feature extractors for the LSTMs on audio and textual input data.

We can define a CNN LSTM model to be trained jointly in Keras. A CNN LSTM can be defined by adding CNN layers on the front end followed by LSTM layers with a Dense layer on the output. -->

#### Encoder-Decoder LSTM
ABCD

#### Bidirectional LSTM
ABCD
