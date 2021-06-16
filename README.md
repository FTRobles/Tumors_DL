# Tumors_DL
Breast Tumor Segmentation with Deeplearning

This projects helps implementing Concolutional Neural Networks for brest ultrasound lesion segmentation

Implemented Tools:

- Open 2D images
- Segments images using a CNN
    - VGG
    - UNET
    - Mask RCNN
- Interactive 3D Slicer for 2D visualization
- Ineractive tools for 3D rendering

## Installation

The develop of this tool was made on [Anaconda](https://www.anaconda.com), so is not necessary but the easiest way to get an environment with all the libraries with its own versions.

### Required extra librarys

|NAME|VERSION|
|:---:|:---:|
|vtk|9.0.1|
|pydicom|1.3.0|
|pillow|7.0.0|
|PyQt|5.12.3|

__NOTE: Use python 3.7 or below__

### Get ready all requirements

Once you have installed Anaconda you should follow the next steps.

- Create a new environment __(Python <= 3.7)__
- Add a new channel: Conda-Forge

Using pip from terminal:

~~~bash
conda create --name NEW_ENV python==3.7
pip install pydicom==1.3.0
pip install pillow==7.0.0
pip install vtk
~~~




