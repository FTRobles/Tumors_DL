# Tumors_DL
Breast Tumor Segmentation with Deeplearning

This projects helps implementing Concolutional Neural Networks for brest ultrasound lesion segmentation

Implemented Tools:

- Open 2D images
- Segments images using a CNN
    - LENET (Pathced based segmentation)
    - VGG
    - UNET
- Validation using accuracy, recall and sensitivity

## Installation

The develop of this tool was made on [Anaconda](https://www.anaconda.com), so is not necessary but the easiest way to get an environment with all the libraries with its own versions.

### Required extra librarys

|NAME|VERSION|
|:---:|:---:|
|open_cv|4.1.2|
|tensorflow|2.1.0|
|keras|2.3.1|
|SK-Learn|0.24.1|
|Matplotlib|3.1.3|

__NOTE: Use python 3.7 or below__

### Get ready all requirements

Once you have installed Anaconda you should follow the next steps.

- Create a new environment __(Python <= 3.7)__
- Add a new channel: Conda-Forge

Using pip from terminal:

~~~bash
conda create --name NEW_ENV python==3.7
conda install tensorflow==2.1.0
conda install keras==2.3.1
pip install opencv-python==4.1.2
pip isntall scikit-learn==0.24.1
pip install matplotlib==3.1.3
~~~




