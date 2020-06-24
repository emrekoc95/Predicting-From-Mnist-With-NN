from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

#downloading mnist dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


#printing first ten training image
_, ax = plt.subplots(1, 10, figsize=(6.4,6.4))

for i in range(0, 10):
    ax[i].axis('off')
    ax[i].imshow(train_images[i], cmap=plt.cm.binary)
    
    
#building neural network
model = models.Sequential()
model.add(layers.Conv2D(32, (28, 28), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#choosing optimizer, loss function and metric
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])



#printing model specs
model.summary()

#preprocessing
#using the first 50,000 training images for training
#using the remaining 10,000 training images for validation
train_images = train_images.reshape((60000, 28, 28, 1))
train_images= train_images.astype('float32') / 255 # rescale pixel values from range [0, 255] to [0, 1]

test_images = test_images.reshape((10000, 28, 28, 1))
test_images= test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

validation_images = train_images[50000:]
validation_labels = train_labels[50000:]

train_images = train_images[:50000]
train_labels = train_labels[:50000]

#model training
history = model.fit(train_images, 
                    train_labels, epochs=5, 
                    batch_size=64, 
                    validation_data=(validation_images, validation_labels))

#finding accuracy and loss
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nAccuracy:', test_acc)
print('Loss: ', test_loss)


#plotting function for showing accuracy and loss graphs
def plotting(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

plotting(history)


#trying our model for specific values
preds = model.predict(test_images)
(_, _), (test_images, _) = mnist.load_data()

plt.axis('off')
plt.imshow(test_images[5], cmap=plt.cm.binary)

print('\nPredictions for numbers between 0-9 (highest value is the result)')
print(preds[5])
print('Result is :',np.argmax(preds[5]))
