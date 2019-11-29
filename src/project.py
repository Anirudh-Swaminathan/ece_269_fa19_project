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
        self.dataset = []
        self.image_size = None

    def load_data(self):
        """Function to load the data"""
        for i in range(200):
            nimg = mpimg.imread(self.data_dir + str(i + 1) + "a.jpg")
            simg = mpimg.imread(self.data_dir + str(i + 1) + "b.jpg")
            self.neutral_imgs.append(nimg)
            self.smiling_imgs.append(simg)
        self.dataset = self.neutral_imgs[:190]
        self.dataset = np.array(self.dataset)
        self.image_size = (self.dataset.shape[1], self.dataset.shape[2])
        self.dataset = self.dataset.reshape(self.dataset.shape[0], self.dataset.shape[1] * self.dataset.shape[2])

    def show_image(self, ind, mode="neutral"):
        """Function to show a random image"""
        if mode == "neutral":
            plt.imshow(self.neutral_imgs[ind], cmap="gray")
            plt.show()

        else:
            plt.imshow(self.smiling_imgs[ind], cmap="gray")
            plt.show()

    def display_image(self, img):
        """Generic function to display an image"""
        plt.imshow(img, cmap="gray")
        plt.show()

    def mean_image(self):
        """Calculate the mean image of a given neutral dataset"""
        num_imgs = self.dataset.shape[0]
        # print(self.dataset.shape)
        self.mean_img = np.matmul(np.transpose(self.dataset), np.ones((num_imgs, 1))) / num_imgs
        self.mean_img = np.transpose(self.mean_img)
        print(self.mean_img.shape)

    def pca(self):
        self.mean_image()
        self.mean_offset = self.dataset - self.mean_img


def main():
    dataset_dir = "../datasets/"
    efc = EigenFaces(dataset_dir)
    efc.load_data()
    for _ in range(5):
        j = random.randint(0, 199)
        efc.show_image(j)
        efc.show_image(j, "smiling")
    efc.pca()
    plot_mean_img = efc.mean_img.reshape(efc.image_size[0], efc.image_size[1])
    efc.display_image(plot_mean_img)


if __name__ == "__main__":
    main()
