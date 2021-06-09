import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # for augment of image
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical  # for one hot encoding
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Dropout  # to have generic classes
from tensorflow.python.keras.layers import Flatten  # flatten layer
from tensorflow.python.keras.layers import Dense  # dense layer cosists of nodes

import pickle

######################################

path = 'Text Data'
testRatio = 0.2
valRatio = 0.2
imageDimensions = (32, 32, 3)
batch_size_val = 50
epochs_val = 1
steps_per_epoch_val = 20


#######################################

images = []
classNo = []
myList = os.listdir(path)
print("Total No of Classes", len(myList))
noOfClasses = len(myList)

print("Importing Classes....")

for x in range(0, noOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg,
                            (imageDimensions[0], imageDimensions[1]))  # resize will more efficien in training process
        images.append(curImg)
        classNo.append(x)
    print(x, end="")
print(" ")
print("Total no of images", len(images))
print("Total class for images", len(classNo))
images = np.array(images)
classNo = np.array(classNo)

############# Spliting Of Data
print("Splitting Data....")
X_train, X_test, y_train, y_test = train_test_split(images, classNo,
                                                    test_size=testRatio)  # for Test X_train has images and y_train has images id..

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)  # for validation
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

# to find no of images in each class
noOfSamples = []
for x in range(0, noOfClasses):
    # print(len(np.where(y_train==x)[0]))
    noOfSamples.append(len(np.where(y_train == x)[0]))
print(noOfSamples)

#### Plotiing
plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClasses), noOfSamples)
plt.title("No Of Images For Each Class")
plt.xlabel("Class ID")
plt.ylabel("NO of Images")
plt.show()

####### Preprocessing
print("Preprocessing of Images....")


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
    img = cv2.equalizeHist(img)  # equalize lighting to image evenly
    img = img / 255  # normalize our image value
    return img


X_train = np.array(
    list(map(preProcessing, X_train)))  # in map images in x train will move to preprocessing function one by one
# img = X_train[20]
# img = cv2.resize(img,(300,300))
# cv2.imshow("Preprocessed",img)
# cv2.waitKey(0)
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))
print(X_train.shape)
### Reshapping
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)  # add one to last
print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

################## Augmentation (To make our image generic)

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                             rotation_range=10)

dataGen.fit(X_train)

############ One Hot Encoding

y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)


########### Creating a model
def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500  # neuron

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0], imageDimensions[1], 1),
                      activation='relu')))
    model.add(Conv2D(noOfFilters, sizeOfFilter1, activation='relu'))
    model.add(MaxPool2D(pool_size=sizeOfPool))
    model.add(Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu'))
    model.add(Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu'))
    model.add(MaxPool2D(pool_size=sizeOfPool))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = myModel()
print(model.summary())


history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                              steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                              validation_data=(X_validation, y_validation), shuffle=1)

############################### PLOT
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# STORE THE MODEL AS A PICKLE OBJECT
model.save('Trained_model.h5')
cv2.waitKey(0)
