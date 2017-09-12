# Image-Recognition-in-Python

# python  ObjectRecognization.py

# import the necessary packages
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import Tkinter, tkFileDialog
from PIL import Image
import os, glob


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

m=1e9
s=1e9
global num_file
imageC=2

def compare_images(imageA, imageB, title,fig_name):
    # compute the mean squared error and structural similarity
    # index for the images
    global m
    global s
    global imageC
    global ssim_fig_name
    temp_m = mse(imageA, imageB)
    temp_s= ssim(imageA, imageB)

    #put the less mse in m and as well as ssim in s
    if m>temp_m:
        m=temp_m
        s=temp_s
        imageC=imageB
        ssim_fig_name=fig_name

    rnd_s=format(s, '.2f')
    rnd_ss=format(1.0003,'.2f')
    #num_file : number of file
    if num_file==0:
        if rnd_s==rnd_ss:
            fig = plt.figure(title)
            plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

            # show first image
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(imageA, cmap=plt.cm.gray)
            plt.axis("off")

            # show the second image
            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(imageC, cmap=plt.cm.gray)
            plt.axis("off")

            # plot image name as text to right upper side
            plt.text(0, 1, "matching image is :" + ssim_fig_name, fontsize=20)

            # show the images
            plt.show()
        else:
            plt.imshow(imageA, cmap=plt.cm.gray)
            plt.axis("off")
            plt.text(0, 1, "this image is not found in dataset", fontsize=20)
            plt.show()




# choose file from directory
root = Tkinter.Tk()
file = tkFileDialog.askopenfilename(parent=root, title='choose a file')

# resize  and save the image
img = Image.open(file)
new_width = 250
new_height = 350
img = img.resize((new_width, new_height), Image.ANTIALIAS)
img.save(file)

# read the comapareable image
original = cv2.imread(file)

# convert the image to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# count the number of picutre in folder
num_file=len(glob.glob('DataSet/*'))


# comapare the images
for infile in glob.glob("DataSet/*.jpeg"):
    file1, ext = os.path.splitext(infile)
    dataimage=cv2.imread(file1+".jpeg")
    num_file=num_file-1

    image_name = file1[8:]

    dataimage=cv2.cvtColor(dataimage, cv2.COLOR_BGR2GRAY)
    compare_images(original,dataimage, "Orginal vs. DataImage",image_name)





