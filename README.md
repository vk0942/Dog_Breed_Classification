# Dog_Breed_Clissification

Model files are here https://drive.google.com/drive/folders/13_m1TBZTBrgP4TAlaPJQ7kW0pVDCYN5S

                                                       Title
                                  Predicting dog breeds using image recognition.

  Abstract

The presented problem fits into the category of fine-grained image recognition.
Predicting dog breeds is a useful task as knowing a dog’s breed is important for knowing their individual breed’s conditions, health concerns, interaction behaviors and natural instincts.
There are over 180 dog breeds. In this project we will try to implement a deep learning model to classify images of dogs into their breeds. For this project we will use images of the dogs’ faces.

Various image processing techniques have been studied to identify the dog’s breeds.
Some of them are: Using coarse to fine classification[2], using coarse to fine classification with normalized cross correlation (NCC)[3], using histogram of oriented gradient (HOG)[4] and scale-invariant feature transform (SIFT)[5], in which an appearance model of their face parts was built with a 67% accuracy over 133 breeds[5], and a deep learning method by transfer learning on convolutional neural networks (CNN)[6,7] etc.
The conventional based approach using LBP (Local Binary pattern)  assigns labels to every pixel using the threshold of the neighboring pixels around the center pixel value and forms a binary number.  After each pixel is computed using the LBP algorithm, Histogram is generated and used to describe the pattern of the image.
CNN Architecture :- Transfer learning provides a performance boost by not requiring a full training from scratch. Instead, it uses pre-trained models which are taught general recurring features. The learning of these models represents fine-tuning the given dataset with the learned weights and biases.

To solve this problem we will be using a deep learning method utilizing transfer learning and data augmentation on CNN(Convolutional Neural Networks). We will first pre process the training data. Then we will use a pre-trained model(for implementing transfer learning) and adapt it to the dog breeds problem and use it to predict the dog breeds. For training data we will use data augmentation which is the  most common method to reduce overfitting on training data , to use different transformations before the feedforward pass during the training; this is called data augmentation.
