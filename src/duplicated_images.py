# Data from http://vision.stanford.edu/aditya86/ImageNetDogs/
# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import cv2
import os
from src.utils import get_image_files


def dhash(image, hash_size=8):
	# convert the image to grayscale and resize the grayscale image,
	# adding a single column (width) so we can compute the horizontal
	# gradient
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (hash_size + 1, hash_size))
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash and return it
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def find_duplicates_dhash(path):
	# grab the paths to all images in our input dataset directory and
	# then initialize our hashes dictionary
	print("[INFO] computing image hashes...")
	image_paths = list(paths.list_images(path))
	hashes = {}
	# loop over our image paths
	for imagePath in image_paths:
		# load the input image and compute the hash
		image = cv2.imread(imagePath)
		h = dhash(image)
		# grab all image paths with that hash, add the current image
		# path to it, and store the list back in the hashes dictionary
		p = hashes.get(h, [])
		p.append(imagePath)
		hashes[h] = p

	return hashes
