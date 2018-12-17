from __future__ import print_function, division

from keras.layers import Input, Dense, Flatten, Dropout, Reshape, Concatenate
from keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

from keras.datasets import cifar10
import keras.backend as K

import matplotlib.pyplot as plt

import sys
import os
import numpy as np
import pandas as pd
import cv2



def get_generator(input_layer, condition_layer):
    merged_input = Concatenate()([input_layer, condition_layer])

    hid = Dense(128 * 32 * 32, activation='relu')(merged_input)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    hid = Reshape((32, 32, 128))(hid)

    hid = Conv2D(128, kernel_size=4, strides=1, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    hid = Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    hid = Conv2D(128, kernel_size=5, strides=1, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    hid = Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    hid = Conv2D(128, kernel_size=5, strides=1, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    hid = Conv2D(128, kernel_size=5, strides=1, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    hid = Conv2D(1, kernel_size=5, strides=1, padding="same")(hid)
    out = Activation("tanh")(hid)

    model = Model(inputs=[input_layer, condition_layer], outputs=out)
    model.summary()

    return model, out


def get_discriminator(input_layer, condition_layer):
    hid = Conv2D(128, kernel_size=3, strides=1, padding='same')(input_layer)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    hid = Conv2D(256, kernel_size=4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    hid = Flatten()(hid)

    merged_layer = Concatenate()([hid, condition_layer])
    hid = Dense(512, activation='relu')(merged_layer)
    # hid = Dropout(0.4)(hid)
    out = Dense(1, activation='sigmoid')(hid)

    model = Model(inputs=[input_layer, condition_layer], outputs=out)

    model.summary()

    return model, out

from keras.preprocessing import image

def one_hot_encode(y):
  z = np.zeros((len(y), 5))
  idx = np.arange(len(y))
  z[idx, y] = 1
  return z

def generate_noise(n_samples, noise_dim):
  X = np.random.normal(0, 1, size=(n_samples, noise_dim))
  return X

def generate_random_labels(n):
  y = np.random.choice(5, n)
  y = one_hot_encode(y)
  return y


tags = ['Atelectasis', 'No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax']


def show_samples(batchidx):
    fig, axs = plt.subplots(5, 6, figsize=(10, 6))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    # fig, axs = plt.subplots(5, 6)
    # fig.tight_layout()
    for classlabel in range(5):
        row = int(classlabel / 2)
        coloffset = (classlabel % 2) * 3
        lbls = one_hot_encode([classlabel] * 3)
        noise = generate_noise(1, 100)
        gen_imgs = generator.predict([noise, lbls])

        for i in range(3):
            # Dont scale the images back, let keras handle it
            img = image.array_to_img(gen_imgs[i], scale=True)
            axs[row, i + coloffset].imshow(img)
            axs[row, i + coloffset].axis('off')
            if i == 1:
                axs[row, i + coloffset].set_title(tags[classlabel])
    plt.show()
    plt.close()

# GAN creation
img_input = Input(shape=(128,128,1))

disc_condition_input = Input(shape=(5,))

discriminator, disc_out = get_discriminator(img_input, disc_condition_input)
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False

noise_input = Input(shape=(100,))
gen_condition_input = Input(shape=(5,))
generator, gen_out = get_generator(noise_input, gen_condition_input)
print(generator)
print(gen_out)
gan_input = Input(shape=(100,))
x = generator([gan_input, gen_condition_input])
print(x)
gan_out = discriminator([x, disc_condition_input])
gan = Model(inputs=[gan_input, gen_condition_input, disc_condition_input], output=gan_out)
gan.summary()

gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

BATCH_SIZE = 52

# # Get training images
(img_x, img_y) = 128,128
train_path = "trainData.csv"

classes = ['Atelectasis', 'No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax']
num_classes = len(classes)

# Load training data
dataTrain = pd.read_csv(train_path)

x_train = []
y_train = []
# prepare label binarizer
from sklearn import preprocessing
# OHE
lb = preprocessing.LabelEncoder()#Binarizer()
lb.fit(classes)

prev = np.zeros((img_x, img_y))
count = 0
for index, row in dataTrain.iterrows():
    img1 = os.path.join("../images/", row["Image Index"])
    image1 = cv2.imread(img1)  # Image.open(img).convert('L')
    image1 = image1[:, :, 0]
    arr1 = cv2.resize(image1, (img_x, img_y))
    arr1 = arr1.astype('float32')
    arr1 /= 255.0
    arr1 = arr1 - np.mean(arr1)
    # DEBUG
    # print("shape of image: {}".format(arr1.shape))
    x_train.append(arr1)
    # not yet one-hot encoded
    label = lb.transform([row["Finding Labels"]])
    # STARTHERE
    y_train.append(np.asarray(label, dtype=np.uint8))
    # y_train.append(lb.transform([row["Finding Labels"]]).flatten().T)
    count += 1

    # 1hot encode labels
    #y_train = lb.fit_transform(y_train)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
print("Label shape (should be (X, 1): {}".format(y_train.shape))
print("Label element shape (should be (1,): {}".format(y_train[0].shape))
y_train = one_hot_encode(y_train[:,0])
x_train = x_train.reshape(count, img_y, img_x, 1)
print("Training shape: {}".format(x_train.shape))




num_batches = int(x_train.shape[0]/BATCH_SIZE)

# Array to store samples for experience replay
exp_replay = []

N_EPOCHS = 200
for epoch in range(N_EPOCHS):

    cum_d_loss = 0.
    cum_g_loss = 0.

    for batch_idx in range(num_batches):
        # Get the next set of real images to be used in this iteration
        images = x_train[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]
        labels = y_train[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]

        noise_data = generate_noise(BATCH_SIZE, 100)
        random_labels = generate_random_labels(BATCH_SIZE)
        # We use same labels for generated images as in the real training batch
        generated_images = generator.predict([noise_data, labels])

        # Train on soft targets (add noise to targets as well)
        noise_prop = 0.05  # Randomly flip 5% of targets

        # Prepare labels for real data
        true_labels = np.zeros((BATCH_SIZE, 1)) + np.random.uniform(low=0.0, high=0.1, size=(BATCH_SIZE, 1))
        flipped_idx = np.random.choice(np.arange(len(true_labels)), size=int(noise_prop * len(true_labels)))
        true_labels[flipped_idx] = 1 - true_labels[flipped_idx]

        # Train discriminator on real data
        d_loss_true = discriminator.train_on_batch([images, labels], true_labels)

        # Prepare labels for generated data
        gene_labels = np.ones((BATCH_SIZE, 1)) - np.random.uniform(low=0.0, high=0.1, size=(BATCH_SIZE, 1))
        flipped_idx = np.random.choice(np.arange(len(gene_labels)), size=int(noise_prop * len(gene_labels)))
        gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]

        # Train discriminator on generated data
        d_loss_gene = discriminator.train_on_batch([generated_images, labels], gene_labels)

        # Store a random point for experience replay
        r_idx = np.random.randint(BATCH_SIZE)
        exp_replay.append([generated_images[r_idx], labels[r_idx], gene_labels[r_idx]])

        # If we have enough points, do experience replay
        if len(exp_replay) == BATCH_SIZE:
            generated_images = np.array([p[0] for p in exp_replay])
            labels = np.array([p[1] for p in exp_replay])
            gene_labels = np.array([p[2] for p in exp_replay])
            expprep_loss_gene = discriminator.train_on_batch([generated_images, labels], gene_labels)
            exp_replay = []
            break

        d_loss = 0.5 * np.add(d_loss_true, d_loss_gene)
        cum_d_loss += d_loss

        # Train generator
        noise_data = generate_noise(BATCH_SIZE, 100)
        random_labels1 = generate_random_labels(BATCH_SIZE)
        g_loss = gan.train_on_batch([noise_data, random_labels, random_labels1], np.zeros((BATCH_SIZE, 1)))
        cum_g_loss += g_loss

    print('\tEpoch: {}, Generator Loss: {}, Discriminator Loss: {}'.format(epoch + 1, cum_g_loss / num_batches,
                                                                           cum_d_loss / num_batches))
    show_samples("epoch" + str(epoch))

    gan.save('/Users/anushkagupta/Desktop/gen.h5')
    discriminator.save('/Users/anushkagupta/Desktop/dis.h5')

    for classlabel in range(5):
        lbls = one_hot_encode([classlabel] * 4)
        noise = generate_noise(4, 100)
        gen_imgs = generator.predict([noise, lbls])

        fig, axs = plt.subplots(500, 500)
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        count = 0
        for i in range(500):
            for j in range(500):
                # Dont scale the images back, let keras handle it
                img = image.array_to_img(gen_imgs[count], scale=True)
                axs[i, j].imshow(img)
                axs[i, j].axis('off')
                plt.suptitle('Label: ' + str(classlabel))
                count += 1
                fig.savefig("/Users/anushkagupta/Desktop/imageconv/" + str(classlabel) + "-" + str([i,j]) + ".png")
                plt.clf()
