import os
import matplotlib.pyplot as plt


import keras 
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout




#Step 1: Data Preprocessing



# Define the data generators for training and validation
datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values from [0, 255] to [0, 1]
    rotation_range = 20,  # Randomly rotate images by up to 20 degrees
    width_shift_range=0.1,  # Randomly shift images horizontally by up to 20% of the width
    height_shift_range=0.2,  # Randomly shift images vertically by up to 20% of the height
    shear_range=0.2,  # Apply shear transformations, kind of like stretching
    zoom_range=0.3,  # Randomly zoom in or out on images
    horizontal_flip=True,  # Randomly flip images horizontally
    vertical_flip = True,
    fill_mode='nearest',  # Fill in new pixels with the nearest pixel values
    validation_split=0.2  # Use 20% of the data for validation
)

train_generator = datagen.flow_from_directory(
    '/Users/kylakim/Desktop/HarvardAI/dataset',
    target_size = (400,400),
    batch_size = 32,
    class_mode = 'categorical', # make sure that it labels with one hot, mutually exclusive
    subset = 'training'



)

validation_generator = datagen.flow_from_directory(
    '/Users/kylakim/Desktop/HarvardAI/dataset',
    target_size = (400,400),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'validation'
)



#Step 2: Initialization
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(400,400,3))) #is input layer (kernel size, (strides-how many times it skips over))
model.add(MaxPooling2D((2, 2))) #largest pixel value form each sector 
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))



#Step 3: Feed Forward
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_generator.class_indices), activation='softmax')) #last output layer, getting input from the train egnerator

#Step 4: Back Propagation
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


model.summary()

#Step 5: Training
history = model.fit(
    train_generator,
    epochs = 25,
    validation_data = validation_generator
)

model.save('textile_classifier.h5')
model = tf.keras.models.load_model('textile_classifier.h5')

