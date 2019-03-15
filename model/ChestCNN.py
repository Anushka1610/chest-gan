from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop, Adam
import pandas as pd
import numpy as np
traindf=pd.read_csv("~/Desktop/newDataTrain.csv")
testdf=pd.read_csv("~/Desktop/newDataTest.csv")

datagen=ImageDataGenerator(rescale=1/.255,validation_split=0.02)
shape=(64,64)
batch_size=50

train_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="/Volumes/Anushka/data/train/",
x_col="Image Index",
y_col="Finding Labels",
has_ext=True, subset="training",
batch_size=batch_size,
seed=42,
shuffle=True,
class_mode="categorical",
color_mode="grayscale",
target_size=shape)

valid_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="/Volumes/Anushka/data/train/",
x_col="Image Index",
y_col="Finding Labels",
has_ext=True,
subset="validation",
batch_size=batch_size,
seed=42,
shuffle=True,
class_mode="categorical",
color_mode="grayscale",
target_size=shape)
test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="/Volumes/Anushka/data/train/",
x_col="Image Index",
y_col=None,
has_ext=True,
batch_size=batch_size,
seed=42,
shuffle=False,
class_mode=None,
color_mode="grayscale",
target_size=shape)

input_shape = (64, 64,1)
kernel_dims = (3, 3)
dropout_rate = 0.2
num_classes = 5
epochs=10
batch_size=50

model=Sequential()
model.add(Conv2D(64, kernel_size=(kernel_dims), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(SpatialDropout2D(dropout_rate))

model.add(Conv2D(64, kernel_size=(kernel_dims), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(SpatialDropout2D(dropout_rate))

# lower-level filters don't need to drop entire feature maps
model.add(Conv2D(64, kernel_size=(kernel_dims), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(dropout_rate))

model.add(Flatten())
model.add(Dense(125, activation="relu"))
model.add(Dropout(dropout_rate))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

model.summary()
STEP_SIZE_TRAIN=2937//50
STEP_SIZE_VALID=1063//50
history = model.fit_generator(generator=train_generator,steps_per_epoch=STEP_SIZE_TRAIN,
                              epochs=epochs,validation_steps=STEP_SIZE_VALID,
                              validation_data=valid_generator,verbose=1)

score=model.evaluate_generator(generator=test_generator)

print("test loss",score[0])
print("test accuracy",score[1])
