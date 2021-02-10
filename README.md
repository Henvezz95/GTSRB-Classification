# GTSRB Classification with Multi-Scale Convolutional Neural Network
Training a  Multi-Scale Convolutional Neural Network to perform multi-class classification on the German Traffic Sign Recognition Benchmark.
The Architecture of the classifier takes inspiration from the 2011 [paper by Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).
It uses Data Augmentation (shear, rotation, translation, zoom, elastic transform, gaussian noise) to increase model accuracy.

The Best Accuracy achieved is 98.75% on the test set (```classifier.h5```) and is near human perfomance (recorded at 98.8% for this dataset). 
