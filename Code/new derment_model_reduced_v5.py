#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:53:50 2018

"""


# PRELIMINARIES


# Import packages
import os
import shutil
import numpy as np


# Declare key directories

  # Set source direcotyr for images
base_source_directory = '/Users/jamesmanzi/Desktop/FoundryDC/Skin rash models/Reduced_Rash_Classes/'  

  # Set base target directory
base_target_directory = '/Users/jamesmanzi/Desktop/FoundryDC/Skin rash models/Reduced_Modeling_Data/'



## Create train and test directories
#
#  # Train
target_directory_train_name = base_target_directory + 'Train'
#os.mkdir(target_directory_train_name)
#
#  # Test
target_directory_test_name = base_target_directory + 'Test'
#os.mkdir(target_directory_test_name)
#
#
#
## Create class directories in train and test folders in modeling directory
#
#  # Get class list
#class_list = os.listdir(base_source_directory)
#class_list = [item for item in class_list if '.DS' not in item]
#
#  # Get list of source directories  
#source_dir_list = []
#for item in class_list:
#    source_directory = base_source_directory + item
#    source_dir_list.append([item, source_directory])
# 
#    
#  # Create test and train folders
#for source_dir_pair in source_dir_list:
#    
#    # Create target directory for training photos for this class
#    dest_train_name = target_directory_train_name + '/' + source_dir_pair[0]
#    os.mkdir(dest_train_name)
#    
#    # Create target directory for training photos for this class
#    dest_test_name = target_directory_test_name + '/' + source_dir_pair[0]
#    os.mkdir(dest_test_name)
#    
#    
#    
#    
## Split photos into test and validation sets       
#    
#  # Set train percentage
#train_percentage = 0.75
#    
#  # Extract and move train photos
#  
#for source_dir_pair in source_dir_list:  
#
#    for f in os.listdir(source_dir_pair[1]):
#        
#        dest_train_name = target_directory_train_name + '/' + source_dir_pair[0]
#        dest_test_name = target_directory_test_name + '/' + source_dir_pair[0]
#        
#        if np.random.rand(1) < train_percentage:
#            shutil.copy(source_dir_pair[1] + '/'+ f, dest_train_name + '/'+ f)
#            
#        else:
#            shutil.copy(source_dir_pair[1] + '/'+ f, dest_test_name + '/'+ f) 
    

# Get number of classes
#num_classes = len(source_dir_list)      
num_classes = 4











## MODELING SECTION
#
#
#
#
#
## Load packages
import numpy as np
import matplotlib.pyplot as plt
from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
#
#
#
#
#
#
# Identify data directories
train_dir = target_directory_train_name
validation_dir = target_directory_test_name
image_size = 224
#
#
#
#
#
#
#
###############    Experiment 1: Train the last 4 layers without data augmentation   ################
#
## BUILD THE MDOEL
#from keras.applications import VGG16
#
##Load the VGG model
#vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
#
## Freeze all the layers up to the trainable laters
#for layer in vgg_conv.layers[:-4]:
#    layer.trainable = False
#
## Check the trainable status of the individual layers
#for layer in vgg_conv.layers:
#    print(layer, layer.trainable)
#
#
from keras import models
from keras import layers
from keras import optimizers
#
## Create the model
#model = models.Sequential()
#
## Add the vgg convolutional base model
#model.add(vgg_conv)
#
## Add new layers
#model.add(layers.Flatten())
#model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(num_classes, activation='softmax'))
#
## Show a summary of the model. Check the number of trainable parameters
#model.summary()
#
#
## FIT THE MODEL
#
## No Data augmentation 
#train_datagen = ImageDataGenerator(rescale=1./255)
#validation_datagen = ImageDataGenerator(rescale=1./255)
#
## Change the batchsize according to your system RAM
#train_batchsize = 100
#val_batchsize = 25
#
## Data Generator for Training data
#train_generator = train_datagen.flow_from_directory(
#        train_dir,
#        target_size=(image_size, image_size),
#        batch_size=train_batchsize,
#        class_mode='categorical')
#
## Data Generator for Validation data
#validation_generator = validation_datagen.flow_from_directory(
#        validation_dir,
#        target_size=(image_size, image_size),
#        batch_size=val_batchsize,
#        class_mode='categorical',
#        shuffle=False)
#
## Compile the model
#model.compile(loss='categorical_crossentropy',
#              optimizer=optimizers.RMSprop(lr=1e-4),
#              metrics=['acc'])
#
## Train the Model
#history = model.fit_generator(
#      train_generator,
#      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
#      epochs=20,
#      validation_data=validation_generator,
#      validation_steps=validation_generator.samples/validation_generator.batch_size,
#      verbose=1)
#
## Save the Model
#model.save('last4_layers.h5')
#
#
## ANALYZE MODEL GOODNESS-OF-FIT
#
## Plot the accuracy and loss curves
#acc = history.history['acc']
#val_acc = history.history['val_acc']
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#epochs = range(len(acc))
#
#plt.plot(epochs, acc, 'b', label='Training acc')
#plt.plot(epochs, val_acc, 'r', label='Validation acc')
#plt.title('Training and validation accuracy')
#plt.legend()
#
#plt.figure()
#
#plt.plot(epochs, loss, 'b', label='Training loss')
#plt.plot(epochs, val_loss, 'r', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
#
#plt.show()
#
#
## Create a generator for prediction
#validation_generator = validation_datagen.flow_from_directory(
#        validation_dir,
#        target_size=(image_size, image_size),
#        batch_size=val_batchsize,
#        class_mode='categorical',
#        shuffle=False)
#
## Get the filenames from the generator
#fnames = validation_generator.filenames
#
## Get the ground truth from generator
#ground_truth = validation_generator.classes
#
## Get the label to class mapping from the generator
#label2index = validation_generator.class_indices
#
## Getting the mapping from class index to class label
#idx2label = dict((v,k) for k,v in label2index.items())
#
## Get the predictions from the model using the generator
#predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
#predicted_classes = np.argmax(predictions,axis=1)
#
#errors = np.where(predicted_classes != ground_truth)[0]
#print("No of errors = {}/{}".format(len(errors),validation_generator.samples))
#
## Show the errors
#for i in range(len(errors)):
#    pred_class = np.argmax(predictions[errors[i]])
#    pred_label = idx2label[pred_class]
#    
#    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#        fnames[errors[i]].split('/')[0],
#        pred_label,
#        predictions[errors[i]][pred_class])
#    
#    original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
#    plt.figure(figsize=[7,7])
#    plt.axis('off')
#    plt.title(title)
#    plt.imshow(original)
#    plt.show()






##############    Experiment 2: Train the last 4 layers with data augmentation   ################


## DEFINE MODEL ARCHITECTURE
#    
##from keras.applications import VGG16
##from keras.applications import inception_v3  
##from keras.applications.inception_v3 import InceptionV3  
##from keras.applications import VGG19
#
#
##Load the VGG model
##vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
#vgg_conv = VGG19(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
#
## Freeze all the layers
#for layer in vgg_conv.layers[:-4]:
#    layer.trainable = False
#
## Check the trainable status of the individual layers
#for layer in vgg_conv.layers:
#    print(layer, layer.trainable)
#
#
#from keras import models
#from keras import layers
#from keras import optimizers
#
## Create the model
#model = models.Sequential()
#
## Add the vgg convolutional base model
#model.add(vgg_conv)
#
## Add new layers
#model.add(layers.Flatten())
#model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(num_classes, activation='softmax'))
#
## Show a summary of the model. Check the number of trainable parameters
#model.summary()
#
#
## FIT THE MODEL
#
#train_datagen = ImageDataGenerator(
#      rescale=1./255,
#      rotation_range=20,
#      width_shift_range=0.2,
#      height_shift_range=0.2,
#      horizontal_flip=True,
#      fill_mode='nearest')
#
#validation_datagen = ImageDataGenerator(rescale=1./255)
#
## Change the batchsize according to your system RAM
#train_batchsize = 20
#val_batchsize = 10
#
## Data Generator for Training data
#train_generator = train_datagen.flow_from_directory(
#        train_dir,
#        target_size=(image_size, image_size),
#        batch_size=train_batchsize,
#        class_mode='categorical')
#
## Data Generator for Validation data
#validation_generator = validation_datagen.flow_from_directory(
#        validation_dir,
#        target_size=(image_size, image_size),
#        batch_size=val_batchsize,
#        class_mode='categorical',
#        shuffle=False)
#
## Compile the model
#model.compile(loss='categorical_crossentropy',
#              optimizer=optimizers.RMSprop(lr=1e-4),
#              metrics=['acc'])
#
## Train the Model
## NOTE that we have multiplied the steps_per_epoch by 2. This is because we are using data augmentation.
#history = model.fit_generator(
#      train_generator,
#      steps_per_epoch=2*train_generator.samples/train_generator.batch_size ,
#      epochs=10,
#      validation_data=validation_generator,
#      validation_steps=validation_generator.samples/validation_generator.batch_size,
#      verbose=1)
#
## Save the Model
#model.save('/Users/jamesmanzi/Desktop/da_last4_layers.h5')
#
## Plot the accuracy and loss curves
#acc = history.history['acc']
#val_acc = history.history['val_acc']
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#epochs = range(len(acc))
#
#plt.plot(epochs, acc, 'b', label='Training acc')
#plt.plot(epochs, val_acc, 'r', label='Validation acc')
#plt.title('Training and validation accuracy')
#plt.legend()
#
#plt.figure()
#
#plt.plot(epochs, loss, 'b', label='Training loss')
#plt.plot(epochs, val_loss, 'r', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
#
#plt.show()
#
## ANALYZE MODEL GOODNESS-OF-FIT
#
## Create a generator for prediction
#validation_generator = validation_datagen.flow_from_directory(
#        validation_dir,
#        target_size=(image_size, image_size),
#        batch_size=val_batchsize,
#        class_mode='categorical',
#        shuffle=False)
#
## Get the filenames from the generator
#fnames = validation_generator.filenames
#
## Get the ground truth from generator
#ground_truth = validation_generator.classes
#
## Get the label to class mapping from the generator
#label2index = validation_generator.class_indices
#
## Getting the mapping from class index to class label
#idx2label = dict((v,k) for k,v in label2index.items())
#
## Get the predictions from the model using the generator
#predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
#predicted_classes = np.argmax(predictions,axis=1)
#
#errors = np.where(predicted_classes != ground_truth)[0]
#print("No of errors = {}/{}".format(len(errors),validation_generator.samples))
#
## Show the errors
#for i in range(len(errors)):
#    pred_class = np.argmax(predictions[errors[i]])
#    pred_label = idx2label[pred_class]
#    
#    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#        fnames[errors[i]].split('/')[0],
#        pred_label,
#        predictions[errors[i]][pred_class])
#    
#    original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
#    plt.figure(figsize=[7,7])
#    plt.axis('off')
#    plt.title(title)
#    plt.imshow(original)
#    plt.show()
    
    
    
    
    

    
    
    
    
    
    ############    Experiment 3: Train the last 4 layers with data augmentation and Inception Netowrk   ################


# DEFINE MODEL ARCHITECTURE
    
# Import packages
from keras.applications import inception_v3  
from keras.applications.inception_v3 import InceptionV3  

from keras.applications import VGG16
from keras.applications import VGG19


from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import os.path


# Import base models
inception_model = InceptionV3(weights='imagenet', include_top=False)
#res_50_model = ResNet50(weights='imagenet', include_top=False)
#vgg_19_model = VGG19(weights='imagenet', include_top=False)
#vgg_16_model = VGG16(weights='imagenet', include_top=False)
#xception_model = Xception(weights='imagenet', include_top=False)

# create the base pre-trained model
base_model = inception_model

# add a global spatial average pooling layer
x = base_model.output

x = GlobalAveragePooling2D()(x)
# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)


# and a dense layer mapping to number of classes
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# Show a summary of the model. Check the number of trainable parameters
model.summary()





# FIT THE MODEL

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 30
val_batchsize = 10

# Data Generator for Training data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')

# Data Generator for Validation data
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the Model
# NOTE that we have multiplied the steps_per_epoch by 2. This is because we are using data augmentation.
history = model.fit_generator(
      train_generator,
      steps_per_epoch=2*train_generator.samples/train_generator.batch_size ,
      epochs=1,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)

# Save the Model
model.save('/Users/jamesmanzi/Desktop/Inception_da_last4_layers.h5')

# Plot the accuracy and loss curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# ANALYZE MODEL GOODNESS-OF-FIT

# Create a generator for prediction
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Get the filenames from the generator
fnames = validation_generator.filenames

# Get the ground truth from generator
ground_truth = validation_generator.classes

# Get the label to class mapping from the generator
label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

## Show the errors
#for i in range(len(errors)):
#    pred_class = np.argmax(predictions[errors[i]])
#    pred_label = idx2label[pred_class]
#    
#    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#        fnames[errors[i]].split('/')[0],
#        pred_label,
#        predictions[errors[i]][pred_class])
#    
#    original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
#    plt.figure(figsize=[7,7])
#    plt.axis('off')
#    plt.title(title)
#    plt.imshow(original)
#    plt.show()
    
    
    