import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from sklearn.model_selection import train_test_split

def loadImages(path):
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        loadedImages.append(plt.imread(path + image))
    return np.array(loadedImages)

one1 = loadImages('./photo_small/1/') / 255
two2 = loadImages('./photo_small/2/') / 255
tree3 = loadImages('./photo_small/3/') / 255
photo = np.concatenate((one1, two2, tree3), axis=0)
label_first =  np.concatenate((np.zeros(160), np.zeros(160), np.ones(160)), axis=0)
label_second = np.concatenate((np.zeros(160), np.ones(160), np.zeros(160)), axis=0)
label_third =  np.concatenate((np.ones(160), np.zeros(160), np.zeros(160)), axis=0)
label_almost = np.vstack((label_first, label_second, label_third))
label = label_almost.swapaxes(1,0)
X_train, X_test, y_train, y_test = train_test_split(photo, label, test_size=0.1, random_state=42)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.fit(X_train, y_train,
          epochs=376,
          verbose=1,
          validation_data=(X_test, y_test))
		  
model.save('saved_model/my_model.h5')

