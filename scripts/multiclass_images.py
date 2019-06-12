import os
import sys
import time
import numpy as np
from PIL import Image


def process_images(filename, img_array, target, directory):

# Open the image and convert to 8bit grayscale
    image = Image.open(imagefile).convert('L')

# Resize image with antialias filter
    image = image.resize((width, height), Image.ANTIALIAS)

# Flatten the matrix to an array
    flatimage = np.ravel(np.array(im1)).reshape((1,length))

# Stack starter image onto the array and update target
    img_array = np.vstack((img_array, flatimage))
    target.append(directory)

    if (len(img_array) % 10000 == 0):
        elapsed_time_img = time.time() - start_time
        print(f'{len(img_array)} images processed in {np.round(elapsed_time_img,2)}s')
            
    return img_array



# From the given directory, step though each file. If it's an image, process it
def get_images(directory, img_array, target):

# Create the image directory
    imgdir = './images/train/' + directory + '/'


# Step through the files, processing the .jpg\
    for file in os.listdir(imgdir):

        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            fullfile = imgdir + filename
            img_array = process_images(fullfile, img_array, target, directory)
        else:
            continue

    return img_array


# Constants

# List of directory names
directory_list = ['articulated_truck','background','bicycle','bus','car',
                  'motorcycle','non-motorized_vehicle','pedestrian',
                  'pickup_truck','single_unit_truck','work_van']

mapping_dict = {
        'articulated_truck' : 0,
        'background' : 1,
        'bicycle' : 2,
        'bus' : 3,
        'car' : 4,
        'motorcycle' : 5,
        'non-motorized_vehicle' : 6,
        'pedestrian' : 7,
        'pickup_truck' : 8,
        'single_unit_truck' : 9,
        'work_van' : 10
    }

width = 28
height = 28
length = width * height

# save file name. If you needed a comment to figure this one out, go lie down and put a wet towel on your head
img_save_file = ('./data/imagefile')
target_save_file = ('./data/targetfile')

# Get the first image and use it to create the array and target list
# I don't know why, but I can't figure out how to create an empty array, so I'll create
# an array with the first image

# open and convert to 8bit grayscale
imagefile = './images/train/articulated_truck/00000002.jpg'
im1 = Image.open(imagefile).convert('L')

# resize with antialias filter
im1 = im1.resize((width, height), Image.ANTIALIAS)

# reshape
im1 = np.ravel(np.array(im1)).reshape((1,length))

# create image array
image_array = im1

# Create the targetlist
image_target = ['articulated_truck']

# Step throught the directory list and get the images in each directory
start_time = time.time()

# print(f'Getting {1982 * 8 * 11} images.')

for imgdir in directory_list:

    image_array = get_images(imgdir, image_array, image_target)

elapsed_time = time.time() - start_time
print(f'Shape of image_array: {image_array.shape}')
print(f'Length of target: {len(image_target)}')
print(f'Size of image_array in MB {sys.getsizeof(image_array) / 1024**2}')
print(f'Total Time: {np.round(elapsed_time,2)}s')

# Write the array out
np.save(img_save_file, image_array)
print(f'File saved to: {img_save_file}')

# Write target array
np.save(target_save_file, image_target)
print(f'Target saved to: {target_save_file}')
