from PIL import Image
import imagehash
import os
import numpy as np
from imutils import paths
from collections import defaultdict

from src.utils import get_image_files


class DuplicateDetector:
    def __init__(self, dirname, hash_size=8):
        self.dirname = dirname
        self.hash_size = hash_size

    def find_duplicates(self):
        """
        Find and Delete Duplicates
        """

        fnames = list(paths.list_images(self.dirname))
        hashes = defaultdict(list)
        print("Finding Duplicates Now!\n")
        for image in fnames:
            with Image.open(image) as img:
                temp_hash = imagehash.average_hash(img, self.hash_size)
                if temp_hash in hashes:
                    print("Duplicate {} \nfound for Image {}!\n".format(image, hashes[temp_hash]))
                    hashes[temp_hash].append(image)
                else:
                    hashes[temp_hash].append(image)

        return hashes

    def find_similar(self, location, similarity=80):
        fnames = list(paths.list_images(self.dirname))
        threshold = 1 - similarity / 100
        diff_limit = int(threshold * (self.hash_size ** 2))

        with Image.open(location) as img:
            hash1 = imagehash.average_hash(img, self.hash_size).hash

        print("Finding Similar Images to {} Now!\n".format(location))
        for image in fnames:
            with Image.open(os.path.join(image)) as img:
                hash2 = imagehash.average_hash(img, self.hash_size).hash

                if np.count_nonzero(hash1 != hash2) <= diff_limit:
                    print("{} image found {}% similar to {}".format(image, similarity, location))
