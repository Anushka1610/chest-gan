import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model

import sys

class_name = sys.argv[1]
latent_dim = 100

generator = load_model("dcgen-" + class_name + ".h5")

# at the end, loop per class, per 1000 images
cnt = 0
fig, ax = plt.subplots()
for num in range(2):
    noise1 = np.random.normal(0, 1, (1, 100))  # cgan.latent_dim))
    # noise1 = np.zeros((1, 10000))
    # labels1 = np.tile(labels, 1000)
    img = generator.predict(noise1)  # labels1])
    plt.imshow(img[cnt, :, :, 0], cmap='gray')
    # cnt+=1
    fig.savefig("../images4000/class1-" + str(num) + ".png")
    plt.clf()
