#!/usr/bin/env python3

# for working with images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# randomness
import random

# matrices
import numpy as np


class EigenFaces(object):
    """
    A PCA object that performs PCA on a dataset
    """

    def __init__(self, data_dir):
        """Constructor
        data_dir - the directory that contains the required images
        """
        self.data_dir = data_dir
        self.neutral_imgs = []
        self.smiling_imgs = []
        self.mean_img = None

    def load_data(self):
        """Function to load the data"""
        for i in range(200):
            nimg = mpimg.imread(self.data_dir + str(i + 1) + "a.jpg")
            simg = mpimg.imread(self.data_dir + str(i + 1) + "b.jpg")
            self.neutral_imgs.append(nimg)
            self.smiling_imgs.append(simg)

    def show_image(self, ind, mode="neutral"):
        """Function to show a random image"""
        if mode == "neutral":
            plt.imshow(self.neutral_imgs[ind], cmap="gray")
            plt.show()

        else:
            plt.imshow(self.smiling_imgs[ind], cmap="gray")
            plt.show()

    def mean_image(self):
        """Calculate the mean image of a given neutral dataset"""
        pass

    def pca(self):
        pass


def main():
    dataset_dir = "../datasets/"
    efc = EigenFaces(dataset_dir)
    efc.load_data()
    for _ in range(5):
        j = random.randint(0, 199)
        efc.show_image(j)
        efc.show_image(j, "smiling")


if __name__ == "__main__":
    main()
