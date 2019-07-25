"""
Code for creating the model using ACGAN

https://towardsdatascience.com/gangogh-creating-art-with-gans-8d087d8f74a1
"""
from keras.datasets import mnist, cifar10

import os
import sys
import json
import itertools
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sculpture_gan.acgan import Acgan

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

plt.style.use("ggplot")

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

TRAIN_IMAGES = 2988
TEST_IMAGES = 1005
IMG_DIM = (128, 128)
BATCH_SIZE = 100
EPOCHS = 100


"""
train_datagen = ImageDataGenerator(
    shear_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# We just rescale the test ones
test_datagen = ImageDataGenerator()


train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(FILE_PATH, '../../sculpture_data/model_data/gan/train'),
        target_size=IMG_DIM,
        batch_size=1,
        color_mode="rgb",
        seed=42,
        shuffle=True,
        class_mode="binary")

test_generator = test_datagen.flow_from_directory(
        directory=os.path.join(FILE_PATH, '../../sculpture_data/model_data/gan/test'),
        target_size=IMG_DIM,
        batch_size=1,
        color_mode="rgb",
        seed=42,
        class_mode="binary")
"""

def generate_imgs(labels):
    """
    Generate & save a number of images

    :param list classes: List containing the numeric indicator for the class to generate

    :return None
    """
    gen = load_model(os.path.join(FILE_PATH, "..", "models/", "acgan_generator.h5"))
    class_indices = [i for i in range(10)]

    noise = np.random.normal(0, 1, (len(labels), 110))
    generated_images = gen.predict([noise, np.array(labels)])

    for i in range(len(generated_images)):
        fixed_img = (127.5 * (generated_images[i] + 1)).astype(np.uint8)   # From -1 to 1 to 0-255
        img = Image.fromarray(fixed_img)
        img.save(os.path.join(FILE_PATH, "..", "generated_images/", f"image_{class_indices[i]}.png"))


def plot_loss():
    """
    Training & Testing Loss
    """
    with open(os.path.join(FILE_PATH, "..", "models/", "loss_history.json"), "r") as f:
        loss_history = json.load(f)

    ##############
    # TRAIN
    ##############
    plt.figure()

    #plt.plot(np.arange(1, EPOCHS+1), loss_history['d_train'], label=f"discriminator")
    #plt.plot(np.arange(1, EPOCHS+1), loss_history['g_train'], label=f"generator")
    plt.plot(np.arange(1, EPOCHS+1), loss_history['real_gen'], label=f"real_gen")
    plt.plot(np.arange(1, EPOCHS+1), loss_history['fake_gen'], label=f"fake_gen")
    plt.plot(np.arange(1, EPOCHS+1), loss_history['real_cls'], label=f"real_cls")
    plt.plot(np.arange(1, EPOCHS+1), loss_history['fake_cls'], label=f"fake_cls")

    plt.title("Train Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(FILE_PATH, "..", "viz/", "train_plot.png"))

    ##############
    # TEST
    ##############
    plt.figure()

    d_test = [loss_history['d_test'][key] for key in loss_history['d_test']][:-1]
    g_test = [loss_history['g_test'][key] for key in loss_history['g_test']][:-1]

    plt.plot([i for i in range(5, EPOCHS+5, 5)], d_test, label=f"discriminator")
    plt.plot([i for i in range(5, EPOCHS+5, 5)], g_test, label=f"generator")

    plt.title("Test Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(FILE_PATH, "..", "viz/", "test_plot.png"))



def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #x_train = np.expand_dims(x_train, axis=-1)
    #x_test = np.expand_dims(x_test, axis=-1)

    if len(sys.argv) > 1:
        if sys.argv[1] == "img":
            generate_imgs(list(range(10)))
        if sys.argv[1] == "plot":
            plot_loss()
        return 

    a = Acgan(10, [x_train, y_train], test_sets=[x_test, y_test], input_shape=(32, 32, 3), data_gen=False)
    results = a.train(len(x_train) // BATCH_SIZE, 
                      epochs=EPOCHS, 
                      batch_size=BATCH_SIZE,
                      save_checkpoints=True,
                      save_path=os.path.join(FILE_PATH, "..", "models/"))

    a.save_state(loss_history=results, path=os.path.join(FILE_PATH, "..", "models/"))
    plot_loss()
    


if __name__ == "__main__":
    main()




