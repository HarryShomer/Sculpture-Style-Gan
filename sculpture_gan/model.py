"""
Code for creating the model using ACGAN
"""
import os
import json
import itertools
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sculpture_gan.acgan import Acgan

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

TRAIN_IMAGES = 2988
TEST_IMAGES = 1005
IMG_DIM = (128, 128)
BATCH_SIZE = 16
EPOCHS = 50


train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# We just rescale the test ones
test_datagen = ImageDataGenerator(
    rescale=1./255
)


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


def generate_imgs(labels):
    """
    Generate & save a number of images

    :param list classes: List containing the numeric indicator for the class to generate

    :return None
    """
    gen = load_model(os.path.join(FILE_PATH, "..", "models/", "acgan_generator.h5"))

    noise = np.random.uniform(-1, 1, (len(labels), 110))
    generated_images = gen.predict([noise, np.array(labels)])

    for i in range(len(generated_images)):
        fixed_img = (127.5 * (generated_images[i] + 1)).astype(np.uint8)   # From -1 to 1 to 0-255
        img = Image.fromarray(fixed_img)
        img.save(os.path.join(FILE_PATH, "..", "generated_images/", f"image_{i}.png"))


def main():
    a = Acgan(12, train_generator, test_sets=test_generator)
    results = a.train(TRAIN_IMAGES // BATCH_SIZE, epochs=EPOCHS, batch_size=BATCH_SIZE)

    a.save_model(os.path.join(FILE_PATH, "..", "models/"))
    with open("results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()




