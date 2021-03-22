# Source https://github.com/philipbl/duplicate-images/blob/master/duplicate_finder.py

import os
from collections import defaultdict
from imutils import paths
from termcolor import cprint
from PIL import Image, ExifTags
import imagehash


def get_file_size(file_name):
    try:
        return os.path.getsize(file_name)
    except FileNotFoundError:
        return 0


def get_image_size(img):
    return "{} x {}".format(*img.size)


def get_capture_time(img):
    try:
        exif = {
            ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in ExifTags.TAGS
        }
        return exif["DateTimeOriginal"]
    except Exception as e:
        print(f"Problem detected. Reason {str(e)}")
        return "Time unknown"


def duplicates_finder(path):
    """
    Find and Delete Duplicates
    """

    fnames = list(paths.list_images(path))
    hashes = defaultdict(list)
    print("Finding Duplicates Now!\n")
    for image in fnames:
        file, hashes_file, file_size, image_size, capture_time = hash_file(image)
        if hashes_file in hashes:
            print("Duplicate {} \nfound for Image {}!\n".format(image, hashes[hashes_file]))
            hashes[hashes_file].append(image)
        else:
            hashes[hashes_file].append(image)

    return hashes


def hash_file(file):
    try:
        hashes = []
        img = Image.open(file)

        file_size = get_file_size(file)
        image_size = get_image_size(img)
        capture_time = get_capture_time(img)

        # hash the image 4 times and rotate it by 90 degrees each time
        for angle in [0, 90, 180, 270]:
            if angle > 0:
                turned_img = img.rotate(angle, expand=True)
            else:
                turned_img = img
            hashes.append(str(imagehash.phash(turned_img)))

        hashes = ''.join(sorted(hashes))

        cprint("\tHashed {}".format(file), "blue")
        return file, hashes, file_size, image_size, capture_time
    except OSError:
        cprint("\tUnable to open {}".format(file), "red")
        return None
