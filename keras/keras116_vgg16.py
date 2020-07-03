from keras.applications import VGG16, VGG19, Xception, ResNet101,ResNet101V2, ResNet152
# from keras.applications import ResNet152V2, ResNet50, ResNet50V2, InceptionV3, InceptionResNetV2
# from keras.applications import MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
# from keras.applications import NASNetLarge, NASNetMobile
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation

model = VGG19()
model = Xception()
model =  ResNet101()
model = ResNet101V2()
model = ResNet152()
# model = ResNet152V2()
# model = ResNet50()
# model = ResNet50V2()
# model = InceptionV3()
# model = InceptionResNetV2()
# model = MobileNet()
# model = MobileNetV2()
# model = DenseNet121()
# model = DenseNet169()
# model = DenseNet201()
# model = NASNetLarge()
# model = NASNetMobile()

vgg16 = VGG16()#weights='imagenet' , include_top=False)
# vgg16.trainable = False

# model.summary()

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.summary()