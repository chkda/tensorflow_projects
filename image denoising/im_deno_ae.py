"""
high level support for doing this and that.
"""
from PIL import Image
import numpy as np 
import tensorflow as tf
from sklearn.datasets import load_files
from tqdm import tqdm
def load_path(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    return files
def tensor(img):
    imag = Image.open(img)
    imag = tf.image.resize_images(imag, size=(24, 24))
    imag = tf.image.convert_image_dtype(imag, dtype='float32')
    imag = tf.image.per_image_standardization(imag)
    return imag
def file_tensor(img_fol):
    list_of_tensors = [tensor(im) for im in tqdm(img_fol)]
    return np.stack(list_of_tensors,axis=0)
fil = load_path('F:/my files/python files/neural networks/tensorflow_projects/image denoising/shapes')
tens = file_tensor(fil)
print (tens[0].shape)