# 02_Modeling

These notebooks create the models.

```
.
├── Binary_Model.ipnyb
├── GPU_Test.ipynb
├── ImageGenerator_Model.ipynb
├── Multiclass_Model.ipynb
└── README.md
```


## Binary_Model.ipynb
> This notebook runs the model with `binary_crossentropy` as the loss function.

## GPU_Test
> A test notebook to check the performance of the GPU. Runs a small CNN on the MNIST digit data. This is a quick, 10-15s total test to compare speed between different platforms.

## ImageGenerator_Model.ipynb
> This notebook uses the ImageDataGenerate and `.flow_from_directory` method to do a multiclass model of the image data.

## Multiclass_Model
> Performs a GPU optimized, multi-class model using `categorical_crossentropy` as the loss function.

## README.md
> Recursively read?