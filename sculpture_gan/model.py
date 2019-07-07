import os
import json
import itertools
from keras.preprocessing.image import ImageDataGenerator
from sculpture_gan.acgan import Acgan

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

train_images = 2988
test_images = 1005
img_dimensions = (128, 128, 3)


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
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
        target_size=img_dimensions,
        batch_size=1,
        color_mode="rgb",
        seed=42,
        shuffle=True,
        class_mode="categorical")

test_generator = test_datagen.flow_from_directory(
        directory=os.path.join(FILE_PATH, '../../sculpture_data/model_data/gan/test'),
        target_size=img_dimensions,
        batch_size=1,
        color_mode="rgb",
        seed=42,
        class_mode="categorical")


def main():
    batch_size = 16
    a = Acgan(12, train_generator, test_sets=test_generator)
    results = a.train(train_images // batch_size, epochs=50, batch_size=batch_size)

    a.save_model()
    with open("results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()




