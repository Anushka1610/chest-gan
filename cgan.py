from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding, Multiply, Concatenate, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import multiply
import cv2
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys

import numpy as np


class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 5
        self.latent_dim = 128

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
	# is here
	
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))

        img = self.generator([noise, label])
	# was here
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 32 * 32, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((32, 32, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        # lower-cased it
        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

	img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='float32')

        #label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        #flat_img = Flatten()(img)
	"""
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
	"""
	c1 = Conv2D(32, kernel_size=3, strides=2, padding="same")(img)
	lr1 = LeakyReLU(alpha=0.2)(c1)
	d1 = Dropout(0.25)(lr1)
	c2 = Conv2D(64, kernel_size=3, strides=2, padding="same")(d1)
	zp1 = ZeroPadding2D(padding=((0, 1), (0, 1)))(c2)
	bn1 = BatchNormalization(momentum=0.8)(zp1)
	lr2 = LeakyReLU(alpha=0.2)(bn1)
	d2 = Dropout(0.25)(lr2)
	c3 = Conv2D(128, kernel_size=3, strides=2, padding="same")(d2)
	bn2 = BatchNormalization(momentum=0.8)(c3)
	lr3 = LeakyReLU(alpha=0.2)(bn2)
	d3 = Dropout(0.25)(lr3)
	c4 = Conv2D(256, kernel_size=3, strides=1, padding="same")(d3)
	bn3 = BatchNormalization(momentum=0.8)(c4)
	lr4 = LeakyReLU(alpha=0.2)(bn3)
	d4 = Dropout(0.25)(lr4)
	f = Flatten()(d4)
	concat = Concatenate()([f, label])
	hid = Dense(512, activation='relu')(concat)
	out = Dense(1, activation='sigmoid')(concat)
        #img = Input(shape=self.img_shape)
        #label = Input(shape=(1,), dtype='int32')

        #label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        #flat_img = Flatten()(img)

        # lower-cased it
        #model_input = multiply([flat_img, label_embedding])

        #validity = model(model_input)

        #return Model([img, label], validity)
	return Model([img, label], out)

    def load_xrays(self, epochs=10, batch_size=128, save_interval=50):
        (img_x, img_y) = 256, 256
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

	count = 0
        for index, row in dataTrain.iterrows():
            img1 = "../images/" + row["Image Index"]
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
	    label = lb.transform([row["Finding Labels"]])[0]
	    # STARTHERE
	    y_train.append(label)
            #y_train.append(lb.transform([row["Finding Labels"]]).flatten().T)
	    count += 1
            # OHE the y data
	# DEBUG DEBUG DEBUG transpose
        # y_train = lb.fit_transform(y_train)
        # finalize data

	# DEBUG 
	print("shape of x train: {}".format(len(x_train)))
        x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)
        x_train = x_train.reshape(count, img_y, img_x, 1)
        #y_train = y_train.reshape(count, num_classes)
	print("Y SHAPE BEFORE RESHAPING: {}".format(y_train.shape))
	y_train = y_train.reshape(-1, 1)
	# DEBUG
	print("Y SHAPE: {}".format(y_train.shape))

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

	# new experience relay to avoid mode collapse
	exp_replay = []

	for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs, labels = x_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 128))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
	    
	    # relay stuff
	    noise_prop = 0.05
	    gene_labels = np.ones((batch_size, 1)) - np.random.uniform(low=0.0, high=0.1, size=(batch_size, 1))
	    flipped_idx = np.random.choice(np.arange(len(gene_labels)), size=int(noise_prop*len(gene_labels)))
	    gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]
	    
	    # Store a random point for experience replay
	    r_idx = np.random.randint(batch_size)
	    exp_replay.append([gen_imgs[r_idx], labels[r_idx], gene_labels[r_idx]])
	    
	    #If we have enough points, do experience replay
	    if len(exp_replay) == batch_size:
	      generated_images = np.array([p[0] for p in exp_replay])
	      labels = np.array([p[1] for p in exp_replay])
	      gene_labels = np.array([p[2] for p in exp_replay])
	      expprep_loss_gene = discriminator.train_on_batch([generated_images, labels], gene_labels)
	      exp_replay = []
	      break
    
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
	    

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 5, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            #if epoch % save_interval == 0:
                #self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 128))
        sampled_labels = np.arange(0, 5).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.load_xrays(epochs=100, batch_size=128, save_interval=50)
    cgan.generator.save('models/gen.h5')
    cgan.discriminator.save('models/disc.h5')
    # Generate one-hot-encoded labels
    # prepare label binarizer

    from sklearn import preprocessing
    lb = preprocessing.LabelEncoder()#Binarizer()

    classes = ['Atelectasis', 'No Finding', 'Cardiomegaly', 'Effusion', 'Pneumothorax']

    OHE_labels = lb.fit_transform(classes)

    # at the end, loop per class, per 1000 images
    cnt = 0
    fig, ax = plt.subplots()
    for label in OHE_labels:
        for num in range(1):
	    nlab = np.asarray([label]).reshape(-1, 1)
	    noise1 = np.random.normal(0, 1, (1, 128))#cgan.latent_dim))
	    #noise1 = np.zeros((1, 10000))
	    #labels1 = np.tile(labels, 1000)
	    img = cgan.generator.predict([noise1, nlab])#labels1])
	    plt.imshow(img[cnt,:,:,0], cmap='gray')
            #cnt+=1
	    fig.savefig("images-strong-conv/" + str(label) + "-" + str(num) + ".png")
	    plt.clf()
