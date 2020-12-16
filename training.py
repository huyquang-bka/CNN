import tensorflow as tf
import keras
import numpy as np
import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from IPython.display import display
from PIL import Image
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
import os
import random
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

import keras
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.models import model_from_json
import cv2
from skimage import transform
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

FAST_RUN = False
IMAGE_WIDTH=150
IMAGE_HEIGHT=150
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3 # RGB color

def CNN_Classification():
  model = Sequential()

  model.add(Conv2D(16, (11, 11), strides=(4, 4), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2) ))
  model.add(Dropout(0.25))

  model.add(Conv2D(20, (5, 5), activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(48, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Dense(2, activation='softmax'))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  #Save Model
  CarsClassification_json =model.to_json()
  with open('FileOld/modelC9f.json', "w") as json_file:
      json_file.write(CarsClassification_json)
      json_file.close()
  model.summary()
  print(model.summary())
  return model

def Load_Image(pos_dir, neg_dir):
    train_images = []
    train_labels = []

    for pos_path in pos_dir:
      Images_Positive=os.listdir(pos_path)
      for image in Images_Positive:
          path=os.path.join(pos_path,image)
          img = cv2.imread(path)
          train_images.append(transform.resize(img,(150,150,3)))
          l = [1,0]
          train_labels.append(l)
    for neg_path in neg_dir:
      Images_Negative=os.listdir(neg_path)
      for image in Images_Negative:
          path=os.path.join(neg_path,image)
          img = cv2.imread(path)
  #       print(path)
          train_images.append(transform.resize(img,(150,150,3)))
          l = [0,1]
          train_labels.append(l)
    np.save("train_image_hustpark",train_images)
    np.save("train_label_hustpark",train_labels)
    return np.array(train_images), np.array(train_labels)

def Train_Test_split(train_data, train_labels, fraction):
    idx = np.random.permutation(train_data.shape[0])
    index = int(len(train_data)*fraction)
    return train_data[:index], train_labels[:index], train_data[index:], train_labels[index:]

def Train_Model(pos_dir, neg_dir, fraction):
    Train_Image,Lable_Image=Load_Image(pos_dir, neg_dir)
    fraction = 0.8
    train_data, train_labels, test_data, test_labels = Train_Test_split(Train_Image,Lable_Image, fraction)
    print ("Train data size: ", len(train_data))
    print ("Test data size: ", len(test_data))
    CNN=CNN_Classification()
    print ("Train data shape: ", train_data.shape)
    print ("Test data shape: ", test_data.shape)
    idx = np.random.permutation(train_data.shape[0])
    CNN.fit(train_data[idx], train_labels[idx], batch_size = 8, epochs = 10)
    #Save weight
    CNN.save_weights('weightC9f.h5')
    predicted_test_labels = np.argmax(CNN.predict(test_data), axis=1)
    test_labels = np.argmax(test_labels, axis=1)
    score = CNN.evaluate(test_data, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print ("Actual test labels:", test_labels)
    print ("Predicted test labels:", predicted_test_labels)
    print ("Accuracy score:", accuracy_score(test_labels, predicted_test_labels))

def Class_object(img,Loaded_Model):
    img_test=[]
    img_test.append(transform.resize(img,(150,150,3)))
    Input_test=np.array(img_test)
    print("Pridict: ",Loaded_Model.predict(Input_test))
    if(Loaded_Model.predict(Input_test)[0][0]>0.8):
        print(1)
        return 1
    else:
        print(0)
        return 0


# import numpy as np
# data = np.load('train_label_hustpark.npy')
# print (data)

if __name__ == "__main__":
    #pos_dir= ["/content/drive/MyDrive/HUSTPark/C9_fix/C9_base/Train/busy/1", "/content/drive/MyDrive/HUSTPark/C9_fix/C9_base/Train/busy/2", "/content/drive/MyDrive/HUSTPark/C9_fix/C9_base/Train/busy/3", "/content/drive/MyDrive/HUSTPark/C9_fix/C9_base/Train/busy/4", "/content/drive/MyDrive/HUSTPark/C9_fix/C9_base/Train/busy/5","/content/drive/MyDrive/HUSTPark/C9_fix/C9_agps/Train/busy/1","/content/drive/MyDrive/HUSTPark/C9_fix/C9_agps/Train/busy/2","/content/drive/MyDrive/HUSTPark/C9_fix/C9_agps/Train/busy/3","/content/drive/MyDrive/HUSTPark/C9_fix/C9_agps/Train/busy/4","/content/drive/MyDrive/HUSTPark/C9_fix/C9_agps/Train/busy/5"]
    #neg_dir= ["/content/drive/MyDrive/HUSTPark/C9_fix/C9_base/Train/free/1","/content/drive/MyDrive/HUSTPark/C9_fix/C9_base/Train/free/2","/content/drive/MyDrive/HUSTPark/C9_fix/C9_base/Train/free/3","/content/drive/MyDrive/HUSTPark/C9_fix/C9_base/Train/free/4","/content/drive/MyDrive/HUSTPark/C9_fix/C9_base/Train/free/5","/content/drive/MyDrive/HUSTPark/C9_fix/C9_agps/Train/free/1","/content/drive/MyDrive/HUSTPark/C9_fix/C9_agps/Train/free/2","/content/drive/MyDrive/HUSTPark/C9_fix/C9_agps/Train/free/3","/content/drive/MyDrive/HUSTPark/C9_fix/C9_agps/Train/free/4","/content/drive/MyDrive/HUSTPark/C9_fix/C9_agps/Train/free/5"]
    #pos_dir=["/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_base/busy/1","/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_base/busy/2","/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_base/busy/3","/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_base/busy/4"]
    #pos_dir=["/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_agps/busy/1","/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_agps/busy/2","/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_agps/busy/3","/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_agps/busy/4"]
    #neg_dir=["/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_base/free/2","/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_base/free/3","/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_base/free/4"]
    #neg_dir=["/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_agps/free/1","/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_agps/free/2","/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_agps/free/3","/content/drive/MyDrive/HUSTPark/1234/Train_1234/C9_agps/free/4"]
    pos_dir=["full3/Train_full3/C9_agps/busy/3"]#,"/content/drive/MyDrive/HUSTPark/full3/Train_full3/C9_agps/busy/3"]
    neg_dir=["full3/Train_full3/C9_agps/free/3"]#,"/content/drive/MyDrive/HUSTPark/full3/Train_full3/C9_agps/free/3"]

    fraction= 0.8
    Train_Image,Lable_Image= Load_Image(pos_dir, neg_dir)
    Train_Image= np.load("FileOld/train_image_hustpark.npy")
    Lable_Image= np.load("FileOld/train_label_hustpark.npy")
    train_data, train_labels, test_data, test_labels = Train_Test_split(Train_Image,Lable_Image, fraction)
    print(len(train_data))
    CNN=CNN_Classification()
    print ("Train data shape: ", train_data.shape)
    print ("Test data shape: ", test_data.shape)
    idx = np.random.permutation(train_data.shape[0])
    ntrain = len(train_data)
    #CNN.fit(train_data[idx], train_labels[idx], batch_size = 64, step_ber_epoch = ntrain// batch_size, epochs = 10)
    CNN.fit(train_data[idx], train_labels[idx], batch_size = 8, epochs = 10)
    #Save weight
    CNN.save_weights('weightC9_agps.h5')
    predicted_test_labels = np.argmax(CNN.predict(test_data), axis=1)
    test_labels = np.argmax(test_labels, axis=1)
    print ("Actual test labels:", test_labels)
    print ("Predicted test labels:", predicted_test_labels)
    print ("Accuracy score:", accuracy_score(test_labels, predicted_test_labels))

# #Load Model
# json_file= open('modelC9f.json','r')
# loaded_model_json= json_file.read()
# json_file.close()
# Loaded_Model = model_from_json(loaded_model_json)
# #Load weights into new model
# Loaded_Model.load_weights("weightC9_base.h5")
# print("Loaded")

def Load_Image(test_dir):
    test_images = []
    for test_path in test_dir:
      Images_Positive=os.listdir(test_path)
      for image in Images_Positive:
          path=os.path.join(test_path,image)
          img = cv2.imread(path)
          test_images.append(transform.resize(img,(150,150,3)))
    return np.array(test_images)

#lấy từ đoạn này nhé.
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        print(filename)
        if img is not None:
            images.append(img);
    return images

# folder="full3/Test_full3/C9_base/free/3"
# Predict_Slot=[]
# filename_array = []
# for filename in os.listdir(folder):
#     images = []
#     img = cv2.imread(os.path.join(folder,filename))
#     images.append(transform.resize(img, (150,150,3)))
#     images= np.array(images)
#     if(Loaded_Model.predict(images)[0][0]>0.5):
#       Predict_Slot.append(1)
#       #print("BusySlot: %s" % (filename))
#       filename_array += ["BusySlot: %s" % (filename)]
#
#     else:
#       Predict_Slot.append(0)
#       #print("FreeSlot: %s" % (filename))
#       filename_array += ["FreeSlot: %s" % (filename)]
# with open('Check_Slotbk.txt', 'w+') as f:
#     for item in filename_array:
#         f.write("%s\n" % item)

# #Check Free Slot
# Index_Slot= []
# Count=0;
# for i in range(len(Predict_Slot)):
#   if(Predict_Slot[i]==0):
#     Count +=1;
#     Index_Slot.append(i)
#     print("FreeSlot: %s" % (filename), i+1)
# print("Total Number of Free Slot:", Count)
# print("Total: ",len(Predict_Slot))
# print("Accuracy:",Count/len(Predict_Slot))
# with open('Number_Slot.txt', 'w+') as f:
#         f.write("Total Number of Free Slot %s\n" % Count)
#
# #Check busy Slot
# Index_Slot= []
# Count=0;
# for i in range(len(Predict_Slot)):
#   if(Predict_Slot[i]==1):
#     Count +=1;
#     Index_Slot.append(i)
#     print("BusySlot: %s" % (filename), i+1)
# print("Total Number of Busy Slot:", Count)
# print("Total: ",len(Predict_Slot))
# print("Accuracy:",Count/len(Predict_Slot))
# with open('Number_Slot.txt', 'w+') as f:
#         f.write("Total Number of Busy Slot %s\n" % Count)