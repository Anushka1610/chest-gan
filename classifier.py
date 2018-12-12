from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
traindf=pd.read_csv("~/Desktop/newDataTrain.csv")
testdf=pd.read_csv("~/Desktop/newDataTest.csv")

datagen=ImageDataGenerator(rescale=1/.255,validation_split=0.02)

train_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="/Volumes/Anushka/data/train/",
x_col="Image Index",
y_col="Finding Labels",
has_ext=True, subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(256,256))

valid_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="/Volumes/Anushka/data/train/",
x_col="Image Index",
y_col="Finding Labels",
has_ext=True,
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(256,256))
test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="/Volumes/Anushka/data/train/",
x_col="Image Index",
y_col=None,
has_ext=True,
batch_size=32,
seed=42,
shuffle=False,
class_mode=None,
target_size=(256,256))