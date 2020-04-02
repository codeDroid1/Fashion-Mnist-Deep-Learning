# -*- coding: utf-8 -*-
"""Fashion Mnist.ipynb
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.datasets import fashion_mnist

(trainX, trainy),(testX, testy) = fashion_mnist.load_data()

"""### Label---------	Class
* 0 T-shirt/top
* 1	Trouser
* 2	Pullover
* 3	Dress
* 4	Coat
* 5	Sandal
* 6	Shirt
* 7	Sneaker
* 8	Bag
* 9	Ankle boot
"""

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f'Shape of the training data : {trainX.shape}')
print(f'Shape of the test data : {testX.shape}')

trainX = trainX/255.0
testX = testX/255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainX[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[trainy[i]])
plt.show()

"""Build The Model"""

from tensorflow.keras import layers
model = keras.Sequential([
                           layers.Flatten(input_shape=(28,28)),
                           layers.Dense(128,activation='relu'),
                           layers.Dense(10)
])

"""The First Layer in this network `(Flatten)`, transfor the format of the images from a two dimensional array (28 * 28) to a one dimensional array (28*28 = 784) pixels. think of this layer as unstacking rows of pixel in the image and lining them up.This layer has no parameters to learn. It only reformats the data.

After the Pixel are flattened, the network consits of a sequence of two `Dense` Lyaers.
These Layers are connected Densly or fully connected neural layers. The first layers has 128 nodes. the second layers returns the 10 nodes. 
Each Node contains a score that indicates the current image belongs to one the 10 classes.

## Complile The Model

Before the model is ready for training, it needs a few more settings. these are added during the model's compile steps;
- Loss Function : This measures how accurate the model is during training. yot want to minimize this functions to steer the model in the right directions.

- Optimizer : this is how the model is updated based on the data it sees and its loss function.

- Metrics : Used to moniter the traning and testing steps/ the following example uses accurac, the fraction of the image that are correctly classified.
"""

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics =['accuracy']
)

"""### Train The Model
Training the neural network model requires the following steps:
- Feed the training data to the model. In the example, the training data is the `trainX` and `Trainy`.
- The model learn to associate images and labels.
- you ask the model to make predictions about a test set in this set `testX` and `testy` are test Case.
"""

model.fit(trainX,trainy,epochs=10)

"""#### Evaluate Accuracy
Compare how the model performs on the test datasets
"""

test_loss, test_acc = model.evaluate(testX, testy, verbose=2)
print(f'Test Accuracy {test_acc}')

"""Test accuracy is less than train accuracy it represents the overfitting the model.
Overfitting happens when machine learning models performs worse on test or new unseen data.
An overfitted model "Memorize" the noise and details in the datasets to a point where it negatively impacts the performance of the model on the new data.

With the model trained, you can use it to make predictions about some images. Attach a softmax layer to convert the logits to probabilities, which are eaiser to interpret.
"""

probability_model = tf.keras.Sequential([
                                         model, layers.Softmax()
])

predictions = probability_model.predict(testX)

print(predictions[7])

print(np.argmax(predictions[7]))

"""Use the trained model"""

img = testX[1]
print(img.shape)

"""tf.keras models are optimized to make predictions on a batch, or collection, of examples at once. Accordingly, even though you're using a single image, you need to add it to a list:"""

img = (np.expand_dims(img,0))
print(img.shape)

"""Predict the correct label for this image"""

predictions_single = probability_model.predict(img)
print(predictions_single)
print(f'OutPut : {np.argmax(predictions_single[0])}')

model.summary()

#save Keras Model
model.save("my_model.h5") #using h5 extension
print("model save!!!")

#load Model 
from keras.models import load_model
new = load_model('my_model.h5')
new.summary()

