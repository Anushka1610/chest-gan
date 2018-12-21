from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import pandas as pd
import cv2

import matplotlib.pyplot as plt

import sys

import numpy as np


# Input shape
img_rows = 128
img_cols = 128
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100

optimizer = Adam(0.0001, 0.8)


def build_generator(latent_dim1):

    model = Sequential()

    model.add(Dense(1024 * 4 * 4, activation="relu", input_dim=latent_dim1))
    model.add(Reshape((4, 4, 1024)))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(512, kernel_size=3, strides=(1, 1), dilation_rate=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(256, kernel_size=3, strides=(1, 1), dilation_rate=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(128, kernel_size=3, strides=(1, 1), dilation_rate=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(64, kernel_size=3, strides=(1, 1), dilation_rate=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(128, kernel_size=3, strides=(1, 1), dilation_rate=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(1, kernel_size=3, strides=(1, 1), dilation_rate=2, padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)


def build_discriminator(img_shape1):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=(2, 2), input_shape=img_shape1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=(2, 2), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=(2, 2), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=(2, 2), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=(2, 2), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    img = Input(shape=img_shape1)
    validity = model(img)

    return Model(img, validity)


# Build and compile the discriminator
discriminator = build_discriminator(img_shape) #multi_gpu_model(self.build_discriminator())
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])

# Build the generator
generator = build_generator(latent_dim) #multi_gpu_model(self.build_generator())

# The generator takes noise as input and generates imgs
z = Input(shape=(latent_dim,))
img = generator(z)

discriminator.trainable = False

# The discriminator takes generated images as input and determines validity
valid = discriminator(img)

# The GAN model  (stacked generator and discriminator)
# Trains the generator to fool the discriminator
gan = Model(z, valid)
gan.compile(loss='binary_crossentropy', optimizer=optimizer)


def load_xrays(epochs=100, batch_size=32):
    (img_x, img_y) = 128, 128
    train_path = "/Users/anushkagupta/Desktop/newDataTrain.csv"

    class_name = sys.argv[1] #['Atelectasis', 'No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax']
    num_classes = 5

    # Load training data
    dataTrain = pd.read_csv(train_path)

    x_train = []
    # prepare label binarizer
    from sklearn import preprocessing
    image_path = "/Volumes/Anushka/data/train/"

    count = 0
    for index, row in dataTrain[dataTrain["Finding Labels"] == class_name].iterrows():
        img1 = image_path + row["Image Index"]
        image1 = cv2.imread(img1)  # Image.open(img).convert('L')
        image1 = image1[:, :, 0]
        arr1 = cv2.resize(image1, (img_x, img_y))
        arr1 = arr1.astype('float32')
        arr1 /= 255.0
        arr1 = arr1 - np.mean(arr1)
        # DEBUG
        # print("shape of image: {}".format(arr1.shape))
        x_train.append(arr1)
        count += 1


        # DEBUG
    print("shape of x train: {}".format(len(x_train)))
    x_train = np.asarray(x_train)

    x_train = x_train.reshape(count, img_y, img_x, 1)

    valid1 = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half of images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]

        # Sample noise and generate a batch of new images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        # Train the discriminator (real classified as ones and generated as zeros)
        d_loss_real = discriminator.train_on_batch(imgs, valid1)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator (wants discriminator to mistake images as real)
        g_loss = gan.train_on_batch(noise, valid1)

        # Plot the progress
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))


"""
        # If at save interval => save generated image samples
        #if epoch % save_interval == 0:
            #self.save_imgs(epoch)

def save_imgs(self, epoch):
    r, c = 100, 10
    noise = np.random.normal(0, 1, (r * c, self.latent_dim))
    gen_imgs = self.generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("../images1/xrays_%d.png" % epoch)
    plt.close()
"""


if __name__ == '__main__':

    load_xrays(epochs=4, batch_size=32)
    generator.save('dcgen.h5')
    discriminator.save('dcdis.h5')

    from sklearn import preprocessing

    lb = preprocessing.LabelEncoder()  # Binarizer()

    classes = ['Atelectasis', 'No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax']

    OHE_labels = lb.fit_transform(classes)

    # at the end, loop per class, per 1000 images
    cnt = 0
    fig, ax = plt.subplots()
    for label in OHE_labels:
        for num in range(1000):
            #nlab = np.asarray([label]).reshape(-1, 1)
            noise1 = np.random.normal(0, 1, (1, 128))  # cgan.latent_dim))
            # noise1 = np.zeros((1, 10000))
            # labels1 = np.tile(labels, 1000)
            img = dcgan.generator.predict(noise1)  # labels1])
            plt.imshow(img[cnt, :, :, 0], cmap='gray')
            # cnt+=1
            fig.savefig("/Users/anushkagupta/Desktop/class1" + str(label) + "-" + str(num) + ".png")
            plt.clf()


