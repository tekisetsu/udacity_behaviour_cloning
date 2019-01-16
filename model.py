import csv
import math
import numpy as np
import os

from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import misc
from sklearn.utils import shuffle


# Hyperparameters and constants
INPUT_IMAGE_SHAPE = (80, 320, 3)
DROPOUT_RATE = 0.8
BATCH_SIZE = 512
EPOCH = 5
VALIDATION_DIR = "validation_data"
TRAINING_DIR = "training_data"

def image_preprocessing(img):
    """
    preprocesses the image, cropping ...
    :param img: np.array
    :return: np.array
    """

    # Removing parasite data (sky, trees and front of the car)
    return img[60:-20, :, :]


def image_augmentation(img):
    """
    generates other input images from a source image 
    :param img: source image
    :return: 
    """
    return np.fliplr(img)


def normalize(img):
    """
    normalizes an image, i took this step out of preprocessing because after this step we can no longer visualize the
    output image
    :param img: np.array
    :return: matrix which values are between [-1,1] and have 0 mean
    """

    def normalize_pixel(x):
        return (x - 128) / 128

    normalize_vector = np.vectorize(normalize_pixel)
    return normalize_vector(img)


def get_img_and_angle(line, data_dir):
    source_path = line[0]
    filename = source_path.split('/')[-1]
    angle = float(line[3])

    current_dir = os.getcwd()
    img = misc.imread(os.path.join(current_dir, "data", data_dir, "IMG", filename))
    return img, angle



def batch_generator(directory):
    """
    GENERATOR : generates images that can go to a size of (2xBATCH_SIZE) for the training of the model
    :returns:
    """
    lines = []
    with open('./data/{}/driving_log.csv'.format(directory)) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)
    number_of_images = len(lines)

    while True:
            for offset in range(0, number_of_images, BATCH_SIZE):
                batch_images = []
                batch_labels = []

                for i in range(BATCH_SIZE):
                    if offset + i < number_of_images:
                        img, angle = get_img_and_angle(lines[offset+i], directory)

                        # image processing
                        img = normalize(image_preprocessing(img))

                        # Adding the original image
                        batch_images.append(img)
                        batch_labels.append(angle)

                        # image augmentation, we augment only the images that have an angle superior to 0.05
                        batch_images.append(image_augmentation(img))
                        batch_labels.append(-angle)
                    else:
                        break

                yield (np.array(batch_images), np.array(batch_labels))


# def training_generator():
#     """
#     GENERATOR : generates images that can go to a size of (2xBATCH_SIZE) for the training of the model
#     :returns:
#     """
#     lines = []
#     with open('./data/training_data/driving_log.csv') as csv_file:
#         reader = csv.reader(csv_file)
#         for line in reader:
#             lines.append(line)
#     number_of_images = len(lines)
#
#     while True:
#             for offset in range(0, number_of_images, BATCH_SIZE):
#                 batch_images = []
#                 batch_labels = []
#
#                 for i in range(BATCH_SIZE):
#                     if offset + i < number_of_images:
#                         img, angle = get_img_and_angle(lines[offset+i], TRAINING_DIR)
#
#                         # image processing
#                         img = normalize(image_preprocessing(img))
#
#                         # Adding the original image
#                         batch_images.append(img)
#                         batch_labels.append(angle)
#
#                         # image augmentation, we augment only the images that have an angle superior to 0.05
#                         batch_images.append(image_augmentation(img))
#                         batch_labels.append(-angle)
#                     else:
#                         break
#
#                 yield (np.array(batch_images), np.array(batch_labels))
#
#
#
# def validation_generator():
#
#     lines = []
#     with open('./data/validation_data/driving_log.csv') as csv_file:
#         reader = csv.reader(csv_file)
#         for line in reader:
#             lines.append(line)
#     number_of_images = len(lines)
#
#     while True:
#             for offset in range(0, number_of_images, BATCH_SIZE):
#                 batch_images = []
#                 batch_labels = []
#
#                 for i in range(BATCH_SIZE):
#                     if offset+i < number_of_images:
#                         img, angle = get_img_and_angle(lines[offset+i], VALIDATION_DIR)
#
#                         # image processing
#                         img = normalize(image_preprocessing(img))
#
#                         # Adding the original image
#                         batch_images.append(img)
#                         batch_labels.append(angle)
#
#                         # image augmentation
#                         batch_images.append(image_augmentation(img))
#                         batch_labels.append(-angle)
#                     else:
#                         break
#
#                 yield (np.array(batch_images), np.array(batch_labels))


def train_model(model, training_steps, validation_steps):
    """
    Trains the model
    """

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

    checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

    train_gen = batch_generator(TRAINING_DIR)
    valid_gen = batch_generator(VALIDATION_DIR)

    model.fit_generator(train_gen, samples_per_epoch=training_steps, nb_epoch=EPOCH,
                        validation_data=valid_gen, nb_val_samples=validation_steps,
                        verbose=1, callbacks=[checkpoint])

    model.save("model.h5")


# Building the NN model

if __name__ == '__main__':


    # Get the number of images for the fit_generator, since we augement all the images, we multiplythe number of
    # unique image by 2

    # For the validation dataset
    with open('./data/validation_data/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        validation_images_count = sum(1 for line in reader)
        validation_steps = 2 * validation_images_count

    # For the training dataset
    with open('./data/training_data/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        training_images_count = sum(1 for line in reader)
        training_steps = 2 * training_images_count

    model = Sequential()
    model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2), input_shape=(80, 320, 3)))
    model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    train_model(model, training_steps, validation_steps)

