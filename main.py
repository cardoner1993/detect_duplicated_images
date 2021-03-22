import os

from src.utils import generate_similar_images
from src.duplicate_finder import duplicates_finder
from src.duplicated_images import find_duplicates_dhash
from src.duplicate_remover import DuplicateDetector


dirname = 'data/test_images'

# generate_similar_images(dirname)


# Duplicates
dr = DuplicateDetector(dirname)
hashes = dr.find_duplicates()

# Find Similar Images
dr.find_similar(os.path.join(dirname, "n02085620_3423_resized_13.jpg"), 70)

# Duplicates finder
hashes_finder = duplicates_finder(dirname)

# Duplicates dhash
hashes_dhash = find_duplicates_dhash(dirname)

