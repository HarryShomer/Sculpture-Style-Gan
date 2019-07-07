"""
Implementation of the Generative Adversarial Network ACGAN

ACGAN Paper: https://arxiv.org/abs/1610.09585

Two other implementations that were helpful in constructing this one:
    1. https://github.com/keras-team/keras/blob/master/examples/mnist_acgan.py
    2. https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py
"""

# 1. If you don't want to use plaidml just get rid of these 2 lines
# 2. If you don't have plaidml set up, follow the installation instructions here - https://github.com/plaidml/plaidml
import plaidml.keras
plaidml.keras.install_backend()

import os
import PIL
import numpy as np
from tqdm import trange
from keras import layers
from keras.optimizers import Adam
from keras.models import Sequential, Model
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
np.random.seed(42)


class Acgan:
    def __init__(self, num_classes, train_sets, test_sets=None, input_shape=(128, 128, 3), noise_dim=110):
        """
        :param ImageDataGenerator train_sets: (imgs, labels)
        :param ImageDataGenerator test_sets: (imgs, labels) - If not provided will skip
        :param num_classes: # of distinct labels
        """
        if len(input_shape) != 3:
            raise ValueError("Input shape must be 3 args (height, width, length")

        if input_shape[0] % 2 != 0 or input_shape[1] % 2 != 0:
            raise ValueError("The first two dimensions must be even. Don't complicate things")

        self._num_classes = num_classes
        self._input_shape = input_shape
        self._noise_dim = noise_dim  # Length of initial random vector for generator

        # For both the order is - if real and then class
        self._loss_function = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Params noted in paper
        self._optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)

        # See here for more details - https://arxiv.org/pdf/1606.03498.pdf
        self._one_sided_labels = [0, .95]

        self._generator = self._construct_generator()
        self._discriminator = self._construct_discriminator()
        self._combined_model = self._construct_combined_model()

        if not train_sets:
            raise ValueError("You need to provide some training data")

        self._train = train_sets
        self._test = test_sets
        self._num_test = len(test_sets) if test_sets else 0

        print("Num test", self._num_test)


    def save_model(self, path=None):
        """
        Save the model (discriminator, generator, and combined)

        :param path: Where you want them saved. If none then place in cwd

        :return None
        """
        if path is None:
            path = os.getcwd()

        self._generator.save(os.path.join(path, "acgan_generator.h5"))
        self._discriminator.save(os.path.join(path, "acgan_discriminator.h5"))
        self._combined_model.save(os.path.join(path, "acgan_model.h5"))


    def generate_imgs(self, ):
        """
        Generate a number of images

        :param list classes: List containing the numeric indicator for the class to generate

        :return 
        """
        pass


    def _test_discriminator_epoch(self):
        """
        Test the discriminator model for the epoch

        :return Test loss
        """
        # Sample noise & classes as generator input and create new images
        noise = np.random.uniform(-1, 1, (self._num_test, self._noise_dim))
        rand_labels = np.random.randint(0, self._num_classes, self._num_test)
        generated_imgs = self._generator.predict([noise, rand_labels])

        test_imgs, test_labels = self._get_test_data()

        # TODO: Get test data
        x = np.concatenate((test_imgs, generated_imgs))

        y_real = np.array([1] * self._num_test + [0] * self._num_test)
        y_labels = np.concatenate((test_labels, rand_labels))

        test_loss = self._discriminator.evaluate(x, [y_real, y_label], verbose=False)

        return float(test_loss[0])


    def _test_generator_epoch(self):
        """
        Test the generator model for the epoch

        :return Test loss
        """
        noise = np.random.uniform(-1, 1, (2 * self._num_test, self._noise_dim))
        rand_labels = np.random.randint(0, self._num_classes, 2 * self._num_test)
        real = np.ones(2 * self._num_test)

        test_loss = self._combined_model.evaluate([noise, rand_labels], [real, rand_labels], verbose=False)

        return float(test_loss[0])


    def _train_discriminator_batch(self, batch_size):
        """
        Train a single batch for the discriminator.

        Steps:
        1. Get the data X/Y data for the batch
        2. Generate fake images using the generator (random noise + labels)
        3. Combine the Real + Fake Images & the Real + Fake Labels
        4. Get one-sided labels - Instead of 0/1 use 0/.95
        5. Train on the batch

        :param batch_size: Size of the batch for training

        :return discriminator loss on batch
        """
        batch_imgs, batch_labels = self._get_batch_data(batch_size)

        # Sample noise & classes as generator input and create new images
        noise = np.random.uniform(-1, 1, (batch_size, self._noise_dim))
        rand_labels = np.random.randint(0, self._num_classes, batch_size)
        generated_imgs = self._generator.predict([noise, rand_labels])

        # Combine the real + fake images/labels for training the discriminator
        X_batch = np.concatenate((batch_imgs, generated_imgs))
        Y_batch = np.concatenate((batch_labels, rand_labels))

        # One sided label-smoothing
        # This is instead of just 0/1 labels for indicating if Fake/Real
        one_sided = np.array([self._one_sided_labels[1]] * batch_size + [self._one_sided_labels[0]] * batch_size)

        disc_loss = self._discriminator.train_on_batch(X_batch, [one_sided, Y_batch])

        return disc_loss


    def _train_generator_batch(self, batch_size):
        """
        Train a single batch for the combined model (really the generator) 

        :param batch_size: Size of the batch for training

        :return combined loss on batch
        """
        # Generate noise & random labels
        # Do 2 * batch_size to get the same # as discriminator (which is real+fake so do 2*fake)
        noise = np.random.uniform(-1, 1, (2 * batch_size, self._noise_dim))
        rand_labels = np.random.randint(0, self._num_classes, 2 * batch_size)

        # Instead of 1
        # Since they are actually fake we are attempting to 'Trick' the model
        one_sided = np.ones(2 * batch_size) * self._one_sided_labels[1]

        # Generator take noise & associated random labels
        # The discriminator should then hpefully guess they are real with the correct label
        gen_loss = self._combined_model.train_on_batch([noise, rand_labels], [one_sided, rand_labels])

        return gen_loss

    
    def train(self, steps_per_epoch, epochs=50, batch_size=16):
        """
        Train for the # of epochs batch by batch

        :param steps_per_epoch: How many batches to generate/draw
        :param epochs: Epochs to train model for
        :param batch_size: batch size to use for training

        :return loss_history dict
        """
        loss_history = {'d_train': [], 'g_train': [], 'd_test': [], 'g_test': []}

        for epoch in range(1, epochs+1):
            epoch_gen_loss = []
            epoch_disc_loss = []

            for batch in trange(steps_per_epoch, desc=f"Training Epoch {epoch}/{epochs}"):
                epoch_disc_loss.append(self._train_discriminator_batch(batch_size))
                epoch_gen_loss.append(self._train_generator_batch(batch_size))
  
            loss_history['d_train'].append(float(np.mean(np.array(epoch_disc_loss))))
            loss_history['g_train'].append(float(np.mean(np.array(epoch_gen_loss))))

            print(f"\nDisriminator train loss: {loss_history['d_train'][-1]}", end=" ", flush=True)
            print(f"Generator train loss: {loss_history['g_train'][-1]}", end=" ", flush=True)

            if self._test:
                loss_history['d_test'].append(self._test_discriminator_epoch())
                loss_history['g_test'].append(self._test_generator_epoch())

                print(f"Discriminator test loss: {loss_history['d_test'][-1]}", end=" ", flush=True)
                print(f"Generator test loss: {loss_history['g_test'][-1]}")

            print("")

        return loss_history


    def _construct_combined_model(self):
        """
        Create the combined model of the two

        :return The combined model of gen/disc
        """
        self._discriminator.compile(loss=self._loss_function, optimizer=self._optimizer)

        # We define the input and the output for the generator
        noise = layers.Input(shape=(self._noise_dim,))
        label = layers.Input(shape=(1,))
        generated_img = self._generator([noise, label])

        # For the combo model we will only train the generator
        self._discriminator.trainable = False

        # Discriminator takes an img and outputs real/fake & pred_class
        is_real, pred_class = self._discriminator(generated_img)

        # The combined model of both the generator and the discriminator
        # Input: noise, and label to the generator
        # Generator uses this to create an image
        # Then the discriminator takes an image 
        # The discriminator then ouputs if real and the predicted class
        combo = Model([noise, label], [is_real, pred_class])
        combo.compile(loss=self._loss_function, optimizer=self._optimizer)

        return combo


    def _construct_generator(self):
        """
        See Tables 1 and 2 in the paper linked at the top

        Deconvolve Steps:
        (, 110)
        (, 110, x/16 * y/16 * 768)
        (, 8, 8, 768)
        (, 16, 16, 384)
        (, 32, 32, 256)
        (, 64, 64, 192)
        (, 128, 128, 3)

        :return The generator model
        """
        # Divide by 16 since we upsample by a factor of 16 (2^4, 2=stride, 4=deconv layers)
        initial_dims = (int(self._input_shape[0] / 16), int(self._input_shape[1] / 16), 768)

        model = Sequential()

        model.add(layers.Dense(int(np.product(initial_dims)), input_dim=self._noise_dim, activation='relu'))
        model.add(layers.Reshape(initial_dims))

        model.add(layers.Conv2DTranspose(384, (5,5), strides=(2,2), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2DTranspose(192, (5,5), strides=(2,2), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', activation='tanh'))

        # Create both inputs for generator
        noise = layers.Input(shape=(self._noise_dim, ))
        img_class = layers.Input(shape=(1,))

        # Instead of one-hot encoding (think of a bow representation)
        classifier = layers.Embedding(self._num_classes, self._noise_dim)(img_class)

        # Element wise 
        # This combines the noise with the class label
        h = layers.multiply([noise, classifier])

        fake_image = model(h)

        # Takes in noise and class - returns a generated image
        return Model([noise, img_class], fake_image)


    def _construct_discriminator(self):
        """
        See Tables 1 and 2 in the paper linked at the top

        :return The discriminator model
        """
        model = Sequential()

        model.add(layers.Conv2D(16, (3, 3), strides=2, padding='same', input_shape=self._input_shape))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(64, (3, 3), strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(256, (3, 3), strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.5))

        # NOTE: Could get of this one if I want....see how long it takes to run
        model.add(layers.Conv2D(512, (3, 3), strides=1, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())

        img_dims = layers.Input(shape=self._input_shape)

        # Doing it this way allows us to generate answers for both seeing if fake & correct class
        # Just means the first layer takes our input shape and passes it to the above Sequential
        # So features is actually the ouput from the above Flatten
        # Looking at the final Model summary shows what I mean
        features = model(img_dims)

        # For both checking if fake & correct class
        fake = layers.Dense(1, activation='sigmoid', name='generation')(features)
        classifier = layers.Dense(self._num_classes, activation='softmax', name='classifier')(features)

        return Model(img_dims, [fake, classifier])


    def _get_test_data(self):
        """
        Get the test data
        """
        imgs, labels = [], []

        for _ in range(self._num_test):
            datum = next(self._test)

            # Go from 0 to 1 -> -1 to 1
            imgs.append((datum[0][0] - .5) / .5)
            labels.append(int(datum[1][0]))

        return imgs, labels


    def _get_batch_data(self, batch_size):
        """
        Get the data for a single batch

        :param batch_size: Size of batch

        :return tuple - imgs, labels
        """
        imgs, labels = [], []

        for _ in range(batch_size):
            datum = next(self._train)

            # Go from 0 to 1 -> -1 to 1
            imgs.append((datum[0][0] - .5) / .5)
            labels.append(int(datum[1][0]))

        return imgs, labels


    def _get_dummies(self, labels):
        """
        One-hot encode a list of labels

        E.g. [1, 0 ,2] - > [[0, 1, 0], [1, 0, 0], [0, 0, 1]]

        :param labels: list of labels

        :return one-hot encoded labels
        """
        num_labels = len(labels)
        empty = np.zeros((num_labels, self._num_classes))
        empty[np.arange(num_labels), np.array(labels)] = 1

        return empty








