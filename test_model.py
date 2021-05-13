from skimage import transform
from PIL import Image
from PIL import ImageDraw
from keras.preprocessing import image
from skimage import transform
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

path = "D:\\TouristCam\\building_keras\\"

def load(filename):
	np_image = Image.open(filename)
	np_image = np.array(np_image).astype('float32')/255
	np_image = transform.resize(np_image, (200, 150, 3))
	np_image = np.expand_dims(np_image, axis=0)
	return np_image


# CIFAR-10 classes
categories = {
    0: "corp 6",
    1: "corp 8",
    2: "home"
}


# загрузим итоговую модель с весовыми коэффициентами
model = load_model("D:/TouristCam/building_keras/saved_model/my_model.h5")


# получить прогноз для этого изображения
data_sample = load(path + 'image.jpg')
sample_image = Image.open(path + 'image.jpg')
#sample_label = categories[data_sample[1].numpy()[0]]

pred = model.predict(data_sample)

#print(pred)

prediction = np.argmax(pred)
print("Predicted label:", categories[prediction])
#print("True label:", sample_label)

## show the image
#plt.axis('off')
#plt.imshow(sample_image)
#plt.show()

with open(path + "out.txt", "w") as file:
    file.write(categories[prediction])
	

	





























