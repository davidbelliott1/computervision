# Technical Report

An indepth look at this project

## The problem

#### Can a computer recognize the objects around it from real-world images?
- More specifically, and I what I originally set out to answer: can a computer recognize a motorcycle from a car?
    
As an avid motorcyclist, I wanted to know if I could create a CNN that would pick out a motorcycle from an image. This allowed me to puruse something that I have a great deal of interest in, as well as pursuing neural networks, which also fascinate me. 


## Data Hunting

To be honest, finding my data source solidified what my Capstone would be. I didn't have to spend hours and hours hunting for data, as I found the motherlode.

[MIOvision Traffic Camera Data (MIO-TCD)](http://podoce.dinf.usherbrooke.ca/) has two enormouse datasets. The first, designed for the classification problem, and the one I will be using, contains 519,982 images in 11 different categories. This consumes just over 2GB.

First, a glimpse at the data:

| Category              | Images  | Example                                              |
|-----------------------|---------|------------------------------------------------------|
| Articulated Truck     | 10346   | ![](/images/train/articulated_truck/00000212.jpg)    |
| Background            | 160,000 | ![](/images/train/background/00591322.jpg)           |
| Bicycle               | 2284    | ![](/images/train/bicycle/00002069.jpg)              |
| Bus                   | 10316   | ![](/images/train/bus/00000433.jpg)                  |
| Car                   | 260,518 | ![](/images/train/car/00000042.jpg)                  |
| Motorcycle            | 1982    | ![](/images/train/motorcycle/00012408.jpg)           |
| Non-motorized Vehicle | 1751    | ![](/images/train/non-motorized_vehicle/00166678.jpg)|
| Pedestrian            | 6262    | ![](/images/train/pedestrian/00072293.jpg)           |
| Pick-up Truck         | 50906   | ![](/images/train/pickup_truck/00000001.jpg)         |
| Single Unit Truck     | 5120    | ![](/images/train/single_unit_truck/00000789.jpg)    |
| Work Van              | 9697    | ![](/images/train/work_van/00001067.jpg)             |


## Overview of the dataset

As you can see, this is heavily biased in favor of cars. This will create an enormously unbalanced set. This will be addressed in the modeling section. 

    - None of the images are of a set size
    - None of the images are of a set aspect ratio
    - Images are color (RGB)
    - Images are all relatively small from 50x50 pixel up to 150x150.
    
This will obviously require some preprocessing to be of any use.

### Preprocessing

    1. Binary
    2. Multiclass
    
#### Binary Preprocessing

I chose to look at two specific categories for this: motorcycles and cars. Motorcycles are limited to 1982 total images. Each image was opened as a grayscale image, scaled to the specified size, flattened into an array and then stacked into a matrix. In order to increase the amount of data available, each image was rotated 45$^{\circ}$ and the resulting image processed and stored.

After the inital model was fitted and run, with reasonable results, I decided to unbalance the classes and gathered almost twice the number of cars (4000 cars vs. 1982 motorcycles) to make things more interesting.

#### Multiclass Preprocessing.

1. Balanced classes
- I pulled the first 2000 images from each category and performed the same process as the binary preprocessing.
2. Full dataset
- I removed all limitations and used all the available data, scaled to 28x28 pixels. 


## Models. Models everywhere...
![](https://memecreator.org/static/images/memes/5047476.jpg)

    1. Binary
    2. Binary Unbalanced
    3. Multiclass
    4. Multiclass Full dataset
    5. ImageDataGenerator
    6. Different Image Sizes
    8. GPU
    9. AWS
    
All models used the same hidden layers. The input layers differ only in the `input_shape`. The output layers differ only in the type (multiclass or binary) and the activation (`softmax` or `sigmoid`).

The base model from the multiclass model:
```python
# Instantiate
cnn_model = Sequential()

#Input Layer
cnn_model.add(Conv2D(filters=112, kernel_size=(3,3), activation='relu',input_shape = (28, 28,1)))
cnn_model.add(MaxPooling2D(pool_size=2))
cnn_model.add(Dropout(0.5))

# Second Layer
cnn_model.add(Conv2D(56, kernel_size=3,activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=2))
cnn_model.add(Dropout(0.5))


# Third Layer
cnn_model.add(Conv2D(28, kernel_size=2,activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=2))
cnn_model.add(Dropout(0.3))

# Fourth Layer
cnn_model.add(Conv2D(14, kernel_size=2,activation='relu'))
cnn_model.add(Dropout(0.5))

# Flatten
cnn_model.add(Flatten())

# Fifth Layer
cnn_model.add(Dense(112, activation='relu'))

# Sixth Layer
cnn_model.add(Dense(56, activation='relu'))

# Seventh Layer
cnn_model.add(Dense(28, activation='relu'))

# Output Layer
cnn_model.add(Dense(11, activation='softmax'))
```

#### Binary
This model is the most straight forward. The initial attempt used a 28x28 image size and the classes were balanced. I chose 28x28 simply because that's what the MNIST Digits dataset as well as the MNIST Fashion dataset used. I did not record any of these results, as it was boring.

#### Binary Unbalanced
I used twice as many cars as motorcycles.

#### Multiclass
A balanced sample of all 11 categories was used. 

#### Multiclass Full Dataset
The entire dataset was used, in a 28x28 size.

#### ImageDataGenerator
Keras provides a library for [preprocessing images](https://keras.io/preprocessing/image/). While there are several ways to manipulate images, I kept mine fairly straightforward. Random image rotations, and flips, both horizontal and vertical. The `.flow_from_directory` method allows you point the generator object at a directory of unprocessed images. There were downsides to this method, which will be discussed later. [Patrick Cavens](https://www.linkedin.com/in/patrickcavins/) helped with code examples and answer to questions, since he got it running before I did.

#### Different Image Sizes
A conversation with the class regarding image sizes and the completely arbitrary nature of my choice made me re-evalute my decision. Since I had no data to go on, other than "Because!" I, being the good little data scientist that I am, experimented. I chose two other image sizes and re-ran the models with them. 28x28 always bugged me, as it's not a power of two. I therefore chose 64x64 and 128x128 as my two other image sizes.

I only ran these on the unbalanced binary classification model. This is due to compute time constraints. I believe there is enough data to justify a longer experiment, but I am, unfortunately, time limited. Something to pursue post-DSI.

#### GPU
GPUs drastically speed up the modeling process. I obtained a [CUDE](https://developer.nvidia.com/cuda-zone) one and installed it on my machine at home. I then reran all the unbalanced binary classification and full dataset multiclass models to get the results. They were, to put it mildly, surprising. I used [How to Install TensorFlow with GPU Support on Windows 10 (Without Installing CUDA) UPDATED!](https://www.pugetsystems.com/labs/hpc/How-to-Install-TensorFlow-with-GPU-Support-on-Windows-10-Without-Installing-CUDA-UPDATED-1419/) extensively to get my GPU up and running. This ended up being quite the ordeal.

#### AWS
I was able to run the gputest.py script on the free instance AWS allows us, but it was, unsurprisingly, very slow. I tried to spin up a larger, GPU-ready, deep-learning AMI, specifically a p3.2xlarge environment. At the time of this document, AWS has not gotten back to me with permission to open one.


### Results Overview
- If the column is empty, no data was collected.
- CPU = 2.7GHz Intel Core i7 (MacBookPro)
- GPU = 3.5GHz Intel Core i5-4690K, NVIDIA GeForce RTX 2080


|                 | CPU      | CPU   | GPU      | GPU  |
|-----------------|----------|-------|----------|------|
|                 | Accuracy | Time  | Accuracy | Time |
| Binary 28       | 78.25    | 59s   | 84.42    | 2.65 |
| Binary 64       | 74.06    | 380s  | 94.66    | 9s   |
| Binary 128      |          | 1548s | 97.56    | 37s  |
| ImageGen 28     | 42.14    | 310s  |          |      |
| ImageGen 64     | 41.39    | 410s  |          |      |
| Full Dataset 28 |          | 350s  | 66.24    | 19s  |


Takeaways:

1. The GPU makes a huge difference. I didn't even bother running the 128x128 on the CPU due to the time to complete one epoch.
2. The larger the size of the image, the better the results. 
3. The ImageGenerator model was actually slower on the GPU enabled system. I will discuss this more.
4. I did not run a 128x128 ImageGenerator model for the same reason: time.
    
    
## That's nice. Did it work?

Well, yes and no. Yes, the model compiled, results were obtained. Would I want to put my safety in its hands? No. Full stop.


## Issues Encountered

__1. Memory and computational efficency__
Processing such a large number of images, particularly when increasing the size of the image, caused an immense slow down as seen here:
![](/graphs/array-v-time.png)

Why?

[Veronica Gionnotta](https://www.linkedin.com/in/vgiannotta/) and I had a discussion which resulted in two things. 1, I closed every image after using it, thinking I was leaving the image around in memory. This did not speed things up much at all. 2, she sent me [Efficient Image Loading for Deep Learning](https://hjweide.github.io/efficient-image-loading). Basically, the method I was using to build my matrices is very memory intensive. The larger the size of the array, the more time was spend allocating memory and the necessary CPU cycles to do that. I did not have a chance to implement this in this code base.

__2. ImageDataGenerator__
There is a significant issue with CPU bottleneck with the `.flow_from_directory` and the subsequent `.fit_generator` methods. Basically, the CPU can't keep up with the GPU. This would explain why my MBP was actually faster: better CPU. In doing some late night research, I found this: [Improving CNN Training Times In Keras](https://medium.com/@joelognn/improving-cnn-training-times-in-keras-7405baa50e09). Too late to implement for this report, but something to try later.


## Lessons Learned

1. Decide on a naming convention early. And I should know better. Randomly naming datafiles is no way to go through life.
2. Record everything. I had to rerun a number of models because I didn't record some of the information from them.
3. Hit Google sooner. If I had asked questions sooner, I would have been able to implement code changes sooner, and possible have solved several of my issues.
4. Better understand what the hidden layers are doing and how to optimize them.
    
#### Things I actually did pretty well!

1. Commenting code as I go made the process of putting everything together quite easy. I was never looking at code and thinking "What on earth?!?!"
2. Trying a stepped approach to model optimization. I actually went about this in the right way. Changing one thing at a time and then comparing results.
    
## Moving Forward

1. Tensorboard
2. Keras functionality to look at the weights and determine the best ones.
3. Better memory and CPU management.
