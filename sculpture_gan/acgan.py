"""
Implementation of the Generative Adversarial Network ACGAN

ACGAN Paper: https://arxiv.org/abs/1610.09585

Two other implementations that were helpful in constructing this one:
    1. https://github.com/keras-team/keras/blob/master/examples/mnist_acgan.py
    2. https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py
"""
import os
import json
import numpy as np
from PIL import Image
from tqdm import trange, tqdm
from keras import layers
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.initializers import RandomNormal
from keras.layers.noise import GaussianNoise
import warnings

from .minibatch import MinibatchDiscrimination

tqdm.monitor_interval = 0
warnings.simplefilter(action='ignore', category=UserWarning)
np.random.seed(42)


class Acgan:
    def __init__(self, num_classes, train_sets, test_sets=None, input_shape=(128, 128, 3), noise_dim=110, data_gen=True):
        """
        :param ImageDataGenerator train_sets: (imgs, labels)
        :param ImageDataGenerator test_sets: (imgs, labels) - If not provided will skip
        :param num_classes: # of distinct labels
        """
        if len(input_shape) != 3:
            raise ValueError("Input shape must be 3 args (height, width, length")

        if input_shape[0] % 2 != 0 or input_shape[1] % 2 != 0:
            raise ValueError("The first two dimensions must be even. Don't complicate things")

        self._data_gen = data_gen
        self._num_classes = num_classes
        self._input_shape = input_shape
        self._noise_dim = noise_dim  # Length of initial random vector for generator

        self._weight_init = 'glorot_normal'

        # For both the order is - if real and then class
        # Sparse since using integers & not one-hot or not exclusive
        self._loss_function = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Params noted in paper
        self._gen_optimizer = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999)
        self._disc_optimizer = Adam(lr=0.0004, beta_1=0.5, beta_2=0.999)

        # Label smoothing - see here for more https://arxiv.org/pdf/1606.03498.pdf
        self._label_smoothing = .9

        self._generator = self._construct_generator()
        self._discriminator = self._construct_discriminator()
        self._combined_model = self._construct_combined_model()

        if not train_sets:
            raise ValueError("You need to provide some training data")

        self._train = train_sets
        self._test = test_sets
        self._num_test = len(test_sets) if test_sets else 0


    def save_state(self, loss_history=None, path=None, suffix=None):
        """
        Save the current model (discriminator, generator, and combined) and the
        current training/testing loss history

        :param loss_history: dict of training/testing loss
        :param path: Where you want them saved. If none then place in cwd
        :param suffix: To identify unique model
    
        :return None
        """
        if path is None:
            path = os.getcwd()

        suffix = "" if suffix is None else f"_epoch{suffix}"

        # Save all 3 models
        self._generator.save(os.path.join(path, f"acgan_generator{suffix}.h5"))
        #self._discriminator.save(os.path.join(path, f"acgan_discriminator{suffix}.h5"))
        #self._combined_model.save(os.path.join(path, f"acgan_model{suffix}.h5"))

        if loss_history is not None:
            with open(os.path.join(path, f"loss_history{suffix}.json"), "w") as f:
                json.dump(loss_history, f, indent=4)


    def _test_discriminator_epoch(self):
        """
        Test the discriminator model for the epoch

        :return loss
        """
        test_imgs, test_labels = self._get_test_data()

        # Sample noise & classes as generator input and create new images
        noise = self._generate_noise(self._num_test)
        rand_labels = np.random.randint(0, self._num_classes, self._num_test)
        generated_imgs = self._generator.predict([noise, rand_labels])

        x = np.concatenate((test_imgs, generated_imgs))

        y_real = np.array([1] * self._num_test + [0] * self._num_test)
        y_labels = np.concatenate((test_labels, rand_labels))

        test_loss = self._discriminator.evaluate(x, [y_real, y_labels], verbose=False)

        return float(test_loss[0])


    def _test_generator_epoch(self):
        """
        Test the generator model for the epoch

        :return loss
        """
        noise = self._generate_noise(2 * self._num_test)
        rand_labels = np.random.randint(0, self._num_classes, 2 * self._num_test)
        real = np.ones(2 * self._num_test)

        test_loss = self._combined_model.evaluate([noise, rand_labels], [real, rand_labels], verbose=False)

        return float(test_loss[0])


    def _test_epoch(self, epoch, loss_history):
        """
        Test the generator & classifier after an epoch. Only called when data given

        :param epoch: # epoch or 'final' if finished training
        :param loss_history: Dict of training/loss info for each epoch

        :return None
        """
        # Manually create progress bar. We'll update it manually
        pbar = tqdm(total=2, desc="Testing the model")

        loss_history['d_test'][str(epoch)] = self._test_discriminator_epoch()
        pbar.update(1)

        loss_history['g_test'][str(epoch)] = self._test_generator_epoch()
        pbar.update(1)

        print("\nTest Loss:",
              f"discriminator = {round(loss_history['d_test'][str(epoch)], 2)},", 
              f"generator = {round(loss_history['g_test'][str(epoch)], 2)}")


    def _train_discriminator_batch(self, batch_size, batch_num=None):
        """
        Train a single batch for the discriminator.

        Steps:
        1. Get the data X/Y data for the batch
        2. Generate fake images using the generator (random noise + labels)
        3. Combine the Real + Fake Images & the Real + Fake Labels
        4. Get one-sided labels - Instead of 0/1 use 0/.95
        5. Train on the batch

        :param batch_size: Size of the batch for training
        :param batch_num: batch in epoch

        :return loss
        """
        batch_imgs, batch_labels = self._get_batch_data(batch_size, batch_num=batch_num)

        # Sample noise & classes as generator input and create new images
        noise = self._generate_noise(batch_size)
        rand_labels = np.random.randint(0, self._num_classes, batch_size)
        generated_imgs = self._generator.predict([noise, rand_labels])

        # One sided label-smoothing
        # Also noisy labels
        y_real = self._noisy_labels(self._label_smoothing * np.ones(batch_size))
        y_fake = self._noisy_labels(np.zeros(batch_size))

        #foo = np.concatenate((y_real, y_fake))
        #bar = np.concatenate((batch_labels, rand_labels))
        #baz = np.concatenate((batch_imgs, generated_imgs))

        # Train generated & real separately 
        real_loss = self._discriminator.train_on_batch(batch_imgs, [y_real, batch_labels])
        fake_loss = self._discriminator.train_on_batch(generated_imgs, [y_fake, rand_labels])
        #total_loss = real_loss[0] + fake_loss[0]

        # Want generated and real to be counted equally
        #disc_real_weight = np.ones(2 * len(batch_imgs))
        # We want to focus more on the real images
        #disc_aux_weight = np.concatenate((1.5* np.ones(len(batch_imgs)), .5 * np.ones(len(batch_imgs))))
                              
        #total_loss = self._discriminator.train_on_batch(baz, [foo, bar], sample_weight=[disc_real_weight, disc_aux_weight])

        #return float(total_loss[0])
        return real_loss, fake_loss


    def _train_generator_batch(self, batch_size):
        """
        Train a single batch for the combined model (really the generator) 

        :param batch_size: Size of the batch for training

        :return loss
        """
        # Do 2 * batch_size to get the same # as discriminator (which is real+fake so do 2*fake)
        noise = self._generate_noise(2 * batch_size)
        rand_labels = np.random.randint(0, self._num_classes, 2 * batch_size)

        # Since they are actually fake we are attempting to 'Trick' the model
        trick = np.ones(2* batch_size)

        # Generator take noise & associated random labels
        # The discriminator should then hpefully guess they are real with the correct label
        gen_loss = self._combined_model.train_on_batch([noise, rand_labels], [trick, rand_labels])

        return gen_loss

    
    def train(self, steps_per_epoch, epochs=50, batch_size=16, save_checkpoints=False, save_path=None):
        """
        Train for the # of epochs batch by batch

        :param steps_per_epoch: How many batches to generate/draw
        :param epochs: Epochs to train model for
        :param batch_size: batch size to use for training

        :return loss_history dict
        """
        # We'll record the training for every epoch
        # For testing we'll note the epoch and if it's the final one
        #loss_history = {'d_train': [], 'g_train': [], 'd_test': {}, 'g_test': {}}
        loss_history = {'real_gen': [], 'fake_gen': [], 'real_cls': [], 'fake_cls': [],
                        'd_train': [], 'g_train': [], 'd_test': {}, 'g_test': {}}

        checkpoints = [i for i in range(int(epochs/4), epochs+int(epochs/4), int(epochs/4))]

        for epoch in range(1, epochs+1):
            epoch_gen_loss = []
            epoch_disc_loss = []
            intra_loss = {'real_cls': [], 'fake_cls': [], 'real_gen': [], 'fake_gen': []}

            for batch in trange(steps_per_epoch, desc=f"Training Epoch {epoch}/{epochs}"):
                r, f = self._train_discriminator_batch(batch_size, batch_num=batch)
                
                intra_loss['real_gen'].append(r[1])
                intra_loss['real_cls'].append(r[2])
                intra_loss['fake_gen'].append(f[1])
                intra_loss['fake_cls'].append(f[2])
                epoch_disc_loss.append(.5*(r[0]+f[0]))

                #epoch_disc_loss.append(self._train_discriminator_batch(batch_size, batch_num=batch))
                epoch_gen_loss.append(self._train_generator_batch(batch_size))
  
            loss_history['d_train'].append(float(np.mean(np.array(epoch_disc_loss))))
            loss_history['g_train'].append(float(np.mean(np.array(epoch_gen_loss))))
            for key in intra_loss:
                loss_history[key].append(float(np.mean(np.array(intra_loss[key]))))
            
            print("Train Loss:", 
                  f"discriminator = {round(loss_history['d_train'][-1], 2)},", 
                  f"generator = {round(loss_history['g_train'][-1], 2)}")

            # I like every 5
            if self._test and epoch % 5 == 0:
                self._test_epoch(epoch, loss_history)

            if save_checkpoints and epoch in checkpoints:                
                self.save_state(loss_history=loss_history, path=save_path, suffix=str(epoch))

        if self._test:
            # Don't bother rerunning if we are saving checkpoints (use last epoch)
            if save_checkpoints:
                loss_history['d_test']["final"] = loss_history['d_test'][max(loss_history['d_test'])]
                loss_history['g_test']["final"] = loss_history['g_test'][max(loss_history['g_test'])]
            else:
                self._test_epoch("final", loss_history)

        return loss_history


    def _construct_combined_model(self):
        """
        Create the combined model of the two

        :return The combined model of gen/disc
        """
        # We are optimizing the addition of both losses
        self._discriminator.compile(loss=self._loss_function, optimizer=self._disc_optimizer)

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

        combo.compile(loss=self._loss_function, optimizer=self._gen_optimizer)

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
        model = Sequential()
        
        # Divide by 16 since we upsample by a factor of 16 (2^4, 2=stride, 4=deconv layers)
        initial_dims = (int(self._input_shape[0] / 8), int(self._input_shape[1] / 8), 384)

        model.add(layers.Dense(int(np.product(initial_dims)), input_dim=self._noise_dim, activation='relu'))
        model.add(layers.Reshape(initial_dims))

        #model.add(layers.Conv2DTranspose(384, (5,5), strides=2, padding='same', kernel_initializer=self._weight_init, activation='relu'))
        #model.add(layers.Dropout(0.5))

        #############################
        model.add(GaussianNoise(0.15))
        #############################

        model.add(layers.Conv2DTranspose(192, (5,5), strides=2, padding='same', kernel_initializer=self._weight_init, activation='relu'))
        model.add(layers.BatchNormalization())
        #model.add(layers.Dropout(0.2))

        #############################
        model.add(GaussianNoise(0.15))
        #############################

        model.add(layers.Conv2DTranspose(96, (5,5), strides=2, padding='same', kernel_initializer=self._weight_init, activation='relu'))
        model.add(layers.BatchNormalization())
        #model.add(layers.Dropout(0.2))
        
        #############################
        model.add(GaussianNoise(0.15))
        #############################

        model.add(layers.Conv2DTranspose(3, (5,5), strides=2, padding='same', kernel_initializer=self._weight_init, activation='tanh'))

        # Create both inputs for generator
        noise = layers.Input(shape=(self._noise_dim, ))
        img_class = layers.Input(shape=(1,))

        # Instead of one-hot encoding (think of a bow representation)
        classifier = layers.Embedding(self._num_classes, self._noise_dim, 
                                      embeddings_initializer=self._weight_init)(img_class)

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

        ##########################################
        model.add(GaussianNoise(0.2, input_shape=self._input_shape))
        ##########################################

        #model.add(layers.Conv2D(16, (3, 3), strides=2, padding='same', kernel_initializer=self._weight_init, input_shape=self._input_shape))
        model.add(layers.Conv2D(16, (3, 3), strides=2, padding='same', kernel_initializer=self._weight_init))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', kernel_initializer=self._weight_init))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(64, (3, 3), strides=2, padding='same', kernel_initializer=self._weight_init))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', kernel_initializer=self._weight_init))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(256, (3, 3), strides=2, padding='same', kernel_initializer=self._weight_init))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(512, (3, 3), strides=1, padding='same', kernel_initializer=self._weight_init))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())

        ##########################################
        #model.add(MinibatchDiscrimination(50, 30))
        ##########################################

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


    def _generate_noise(self, num_imgs):
        """
        Generate noise for num of images 

        :param num_labels: # of labels to generate

        :return num of images
        """
        return np.random.normal(0, 1, (num_imgs, self._noise_dim))


    def _noisy_labels(self, labels, prob=.05):
        """
        Randomly flip the labels when training the discriminator

        :param labels: list of labels
        :param prob: Probability of flipping

        :return flipped labels
        """
        for l in range(len(labels)):
            # When 0 (which expect `prob` % of time - then flip)
            if np.random.choice(np.arange(0, 2), p=[prob, 1-prob]) == 0:
                # If generated greated than True smooth label then generate for lower & vice-versa
                labels[l] = 0 if labels[l] == self._label_smoothing else self._label_smoothing

        return np.array(labels)


    def _get_test_data(self):
        """
        Get the test data
        """
        imgs, labels = [], []

        for i in range(self._num_test):
            if self._data_gen:  
                datum = next(self._test)
                datum = datum[0][0], datum[1][0]
            else:
                datum = self._test[0][i], self._test[1][i]

            # Go from 0 to 1 -> -1 to 1
            imgs.append((datum[0] - 127.5) / 127.5)
            labels.append(int(datum[1]))

        return imgs, labels


    def _get_batch_data(self, batch_size, batch_num=None):
        """
        Get the data for a single batch

        :param batch_size: Size of batch

        :return tuple - imgs, labels
        """
        imgs, labels = [], []

        for i in range(batch_size):
            if self._data_gen:  
                datum = next(self._train)
                datum = datum[0][0], datum[1][0]
            else:
                offset = batch_num*batch_size
                datum = self._train[0][offset+i], self._train[1][offset+i]

            # Go from 0 to 1 -> -1 to 1
            imgs.append((datum[0] - 127.5) / 127.5)
            labels.append(int(datum[1]))

        return np.array(imgs), np.array(labels)



