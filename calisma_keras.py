# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 12:21:05 2019

@author: numan
"""
from PIL import Image
import os
import cv2
import numpy as np
train_negative_path = 'LSIFIR/Classification/Train/neg'
train_pozitive_path = 'LSIFIR/Classification/Train/pos'
test_negative_path = 'LSIFIR/Classification/Test/neg'
test_pozitive_path = 'LSIFIR/Classification/Test/pos'

#%% train_data_loading
data_pos_train = []
labels_pos_train = []
for i in os.listdir(train_pozitive_path):
    try:
        image = cv2.imread(train_pozitive_path+'/'+i)
        image_to_array = Image.fromarray(image,'RGB')
        image_size = image_to_array.resize((64,32))
        data_pos_train.append(np.array(image_size))
        labels_pos_train.append(0)
    except AttributeError:
        print('')
labels_neg_train = []
data_neg_train = []
for j in os.listdir(train_negative_path):
    try:
        image = cv2.imread(train_negative_path+'/'+j)
        image_to_array = Image.fromarray(image,'RGB')
        image_size = image_to_array.resize((64,32))
        data_neg_train.append(np.array(image_size))
        labels_neg_train.append(1)
    except AttributeError:
        print('')
#test_data_loading
data_pos_test = []
labels_pos_test = []
for i in os.listdir(test_pozitive_path):
    try:
        image = cv2.imread(test_pozitive_path+'/'+i)
        image_to_array = Image.fromarray(image,'RGB')
        image_size = image_to_array.resize((64,32))
        data_pos_test.append(np.array(image_size))
        labels_pos_test.append(0)
    except AttributeError:
        print('')
labels_neg_test = []
data_neg_test = []
for j in os.listdir(test_negative_path):
    try:
        image = cv2.imread(test_negative_path+'/'+j)
        image_to_array = Image.fromarray(image,'RGB')
        image_size = image_to_array.resize((64,32))
        data_neg_test.append(np.array(image_size))
        labels_neg_test.append(1)
    except AttributeError:
        print('')        
        

#%%
        
train_Y = np.concatenate((labels_pos_train,labels_neg_train),axis = 0)
train_X = np.concatenate((data_pos_train,data_neg_train),axis = 0)
test_Y = np.concatenate((labels_pos_test,labels_neg_test),axis = 0)
test_X = np.concatenate((data_pos_test,data_neg_test),axis = 0)

from keras.utils import to_categorical

train_Y = to_categorical(train_Y,num_classes=2)
test_Y = to_categorical(test_Y,num_classes=2)
#%% MODEl


from  keras.layers  import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,Activation
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


model = Sequential()
model.add(Conv2D(32,(3,3),input_shape = (32,64,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2)) # output
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])


#%% model_fit
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(train_X)

hist = model.fit_generator(datagen.flow(train_X,train_Y,batch_size=4000),validation_data=(test_X,test_Y)
,epochs=3,steps_per_epoch=len(train_X)/4000)        
#%% model_save
model.save_weights("deneme.h5")

#%% model_evaluation
print(hist.history.keys())
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["acc"], label = "Train acc")
plt.plot(hist.history["val_acc"], label = "Validation acc")
plt.legend()
plt.show()

#%% save_history
import json
with open("deneme.json","w") as f:
    json.dump(hist.history, f)
    
#%% load_history
import codecs
with codecs.open("LSIFIR.json", "r",encoding = "utf-8") as f:
    h = json.loads(f.read())
plt.plot(h["loss"], label = "Train Loss")
plt.plot(h["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(h["acc"], label = "Train acc")
plt.plot(h["val_acc"], label = "Validation acc")
plt.legend()
plt.show()   
        