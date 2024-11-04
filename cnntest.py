import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

model = models.Sequential() # this is creating a sequential cnn
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#this is using 5 layers with maxpooling2d and conv2d, two industry standard dense layers
#a dense layer is a way of modifying an image to make it readable
#it returns an output in the format (height, width, channels) a channel is a layer to each pixel, we start with 3(RGB) but the model adds it own qualities (I THINK THIS IS AARUSH AND I TEND TO GET STUFF WRONG)

model.add(layers.Flatten()) ## turning the 3D data into 1D
model.add(layers.Dense(64, activation='relu')) # Dense layers is like a standard nnet, with each neuron in a layer taking inputs for all the inputs in a previous layers
model.add(layers.Dense(10)) # adding 10 dense layers (I THINK)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, #this is training the model using the training data set in the database each epoch is a version of the data
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy') # training data accuracy
plt.plot(history.history['val_accuracy'], label = 'val_accuracy') # test data accuracy
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess the image
img_path = 'truck.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(32, 32))  # Resize to 32x32
img_array = image.img_to_array(img)  # Convert image to array
img_array = img_array / 255.0  # Normalize pixel values
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

predictions = model.predict(img_array)

predicted_class = np.argmax(predictions[0])
print("Predicted class:", class_names[predicted_class])


print(test_acc)