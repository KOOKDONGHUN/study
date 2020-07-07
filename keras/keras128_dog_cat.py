from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img

img_cat = load_img('./Data/dog_cat/cat.png',target_size=(224,224))
img_dog = load_img('./Data/dog_cat/dog.png',target_size=(224,224))
img_suit = load_img('./Data/dog_cat/suit.png',target_size=(224,224))
img_yang = load_img('./Data/dog_cat/yang.jpg',target_size=(224,224))

plt.imshow(img_cat)
# plt.imshow(img_dog)
# plt.imshow(img_suit)
# plt.imshow(img_yang)

from keras.preprocessing.image import img_to_array

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_yang = img_to_array(img_yang)

print(arr_dog)
print(type(arr_dog))
print(arr_dog.shape)

# RGB -> BGR ??
from keras.applications.vgg16 import preprocess_input

arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_yang = preprocess_input(arr_yang)

print(arr_dog)
print(type(arr_dog))
print(arr_dog.shape)

# 이미지를 하나로 합친다. 
import numpy as np
arr_input = np.stack([arr_dog, arr_cat, arr_suit, arr_yang])

print(arr_input.shape)

# model
model = VGG16()
probs = model.predict(arr_input)

print(probs)
print(f'probs.shape : {probs.shape}')

# 이미지 결과
from keras.applications.vgg16 import decode_predictions

res = decode_predictions(probs)

for i in range(4):
    print('-'*40)
    print(res[i])