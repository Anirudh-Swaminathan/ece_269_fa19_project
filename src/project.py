#!/usr/bin/env python3

# for working with images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# randomness
import random

# matrices
import numpy as np

# eigen value computation
from scipy.linalg import eigh


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
        self.eigen_faces = None

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
        # 31266 * 190 - dataset shape
        self.dataset = np.transpose(self.dataset)

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
        num_imgs = self.dataset.shape[1]
        # print(self.dataset.shape)
        self.mean_img = np.matmul(self.dataset, np.ones((num_imgs, 1))) / num_imgs
        # 31266 * 1 - mean_img
        self.mean_img = self.mean_img
        print(self.mean_img.shape)

    def pca(self):
        # computes the mean image of the dataset
        self.mean_image()

        # computes image - mean for all images
        # 31266 * 180
        self.mean_offset = self.dataset - self.mean_img

        # 180*180
        mod_cov = np.matmul(np.transpose(self.mean_offset), self.mean_offset)
        print(mod_cov.shape)

        # check if this is a symmetric matrix
        print(np.allclose(mod_cov, np.transpose(mod_cov)))

        # compute the eigen values and the corresponding eigenvectors in ascending order
        # 190 eigen values
        # 190 * 190 eigen vectors
        # ith column - corresponding to the ith eigen vector
        eig_vals, mod_eig_vecs = eigh(mod_cov)
        print(eig_vals.shape, mod_eig_vecs.shape)

        # 31266 * 190 - eigen vectors of the original data
        eig_vecs = np.matmul(self.mean_offset, mod_eig_vecs)

        # normalize the eigen vectors now
        # 31266 * 190 normalized eigen vectors
        norm_cnst = np.sum(np.square(eig_vecs), 0)
        self.eigen_faces = np.divide(eig_vecs, norm_cnst)
        print(self.eigen_faces.shape)


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
    for j in range(5):
        img = efc.eigen_faces[:, 189-j]
        plot_face = img.reshape(efc.image_size[0], efc.image_size[1])
        efc.display_image(plot_face)


if __name__ == "__main__":
    main()
