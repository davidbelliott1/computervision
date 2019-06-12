# Vehicle Identification

- An exploration in computer vision with a convoluted neural network

### Top-Level Directory
```
.
├── Techical_Report.md
├── README.md
├── data
├── graphs
├── images
├── notebooks
└── scripts
```


In a world increasingly filled with semi-autonomous vehicles, and with the coming of completely autonomous vehicles, the ability to correctly identify the objects around the vehicle is increasingly important.

I am an avid motorcycle rider, track-rider, instructor, and instructor-trainer. It should therefore be obvious I have a vested interest in making these current and near-future vehicles see me. I set out to determine if I could create a Convoluted Neural Network (CNN) to correctly classify images.

## Project Overview

This project will read image data and train a CNN on the collected images. I was extremely lucky in my initial foray for data and came across the [MIOvision Traffic Camera Dataset (MIO-TCD)](http://podoce.dinf.usherbrooke.ca/). There is one significant downside to this dataset, however: I have no way of knowing the ground truth of the test set. Contained within the train dataset is 648,959 images divided into 11 categories:

    1.  Articulated truck
    2.  Bicycle
    3.  Bus
    4.  Car
    5.  Motorcycle
    6.  Non-motorized vehicle
    7.  Pedestrian
    8.  Pickup truck
    9.  Single unit truck
    10. Work van
    11. Background
    
I loaded and manipulated this data to standardize, and in some cases enhance it. The resulting arrays of data were then sent through a CNN. I first did a binary classification, to some success, so proceeded to make my life harder by trying multiclass classification. In addition to going to multiclass, I also added the use of a GPU into the project, as well as attempting to run this on AWS.


## Directory

I will create a `README.md` file for each sub-directory to outline their contents.

1. [Technical_Report.md](Technical_Report.md)
    - A more thorough discussion of the project, successes, limitations, failures, and lessons learned.
    
2. README.md
    - In an effort to be complete, I'm including the document you're currently reading.
    
3. [data](data)
    - I will be including a few small files to give you an idea of what I collected, but I will not be including the majority of the data I gathered as it amounts to several gigabytes.
    
4. [graphs](graphs)
    - All imagery used for the presentation or the report is stored here, with the exception of the example images
    
5. [images](images)
    - One sample image from each category is contained here. Should you choose to download the MIO-TCD dataset, it will unpack into a similar directory structure. My notebooks are constructed to read this structure.
    
6. [notebooks](notebooks)
    - All jupyter notebooks are stored here.
    
7. [scripts](scripts)
    - Python scripts are stored here.
