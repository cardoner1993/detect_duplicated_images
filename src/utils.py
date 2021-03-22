import uuid

import magic
import os
import random
from PIL import Image
import cv2
import shutil

from imutils import paths


def get_image_files(path):
    """
    Check path recursively for files. If any compatible file is found, it is
    yielded with its full path.
    :param path:
    :return: yield absolute path
    """
    def is_image(file_name):
        # List mime types fully supported by Pillow
        full_supported_formats = ['gif', 'jp2', 'jpeg', 'pcx', 'png', 'tiff', 'x-ms-bmp',
                                  'x-portable-pixmap', 'x-xbitmap']
        try:
            mime = magic.from_file(file_name, mime=True)
            return mime.rsplit('/', 1)[1] in full_supported_formats
        except IndexError:
            return False

    path = os.path.abspath(path)
    for root, dirs, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file)
            if is_image(file):
                yield file


def convert_image_to_greyscale(file):
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_file = f"{os.path.splitext(file)[0]}_greyscale{os.path.splitext(file)[1]}"
    im = Image.fromarray(gray)
    im.save(new_file)


def resize_image(file):
    image = cv2.imread(file)
    hash_size = random.randint(6, 14)
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    new_file = f"{os.path.splitext(file)[0]}_resized_{hash_size}{os.path.splitext(file)[1]}"
    im = Image.fromarray(resized)
    im.save(new_file)


def rotate_image(file):
    angle = random.randint(15, 360)
    image = Image.open(file)
    rotated = image.rotate(angle)
    # image.show()
    new_file = f"{os.path.splitext(file)[0]}_rotated_{angle}{os.path.splitext(file)[1]}"
    rotated.save(new_file)


def generate_new_equal_image(file):
    hash_name = uuid.uuid4()
    new_file = f"{os.path.splitext(file)[0]}_{hash_name}_{os.path.splitext(file)[1]}"
    shutil.copyfile(file, new_file)


def generate_similar_images(path):
    fnames = list(paths.list_images(path))
    for image in fnames:
        print(f"Preparing file {image}")
        convert_image_to_greyscale(image)
        rotate_image(image)
        generate_new_equal_image(image)
        resize_image(image)