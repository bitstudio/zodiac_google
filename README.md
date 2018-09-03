# Zodiac

We aim to classify hand shadow.

## Introduction

De facto image classification standard utilizes convolutional neural network.
To generate labels from data, this has drawbacks. Changing examples involves retraining.
Instead we want the model to perform less-than-one-shot classification, compare the data to a given set of templates
to see which one is most similar.

### Less than one shot

Template > statistics > compare < statistics < data    

### Training paradigm

sample data from the same class > compare with the data > high confidence
 
sample data from different classes > compare with the data > low confidence

We can add some variation during the sample to get a better invarient behavior.

## Shadow projection data

### Scale and translation invariant

### Radial coordinate

### Rotation invariant

In classical computer vision has been studied this, and propose "moment" as the statistics.
Hu moment in particular is the based of many work that follow.
Moment is the sum of the data up to a number of exponent.

## Moment network

### Residue network

Based on our setting, though residue net supposes to help resolve exploding/vanishing gradient issues.
It's not good enough.

### Gated Residue network

Based from LSTM, could represent residue network, higher expressive power.


## Implementation

### Tensorflow
map_fn is slow. GPU utility drops there.

### TPU
Can't do map_fn on TPU at all. This is the current limitation. We have to resort to pre-generate a lot of sample beforehand.


## Results

A shadowplay system that we can change template on the fly. 
If the classification result is not good enough due to environmental change? You can add more example to refine the results.
Pretty convenient.