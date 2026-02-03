---
layout: default
title: "Module 7: Vision"
---

# Module 7: Vision
## "Machines that See"

*Research Document for DATA 201 Course Development*

---

# Table of Contents

1. [Introduction](#introduction)
2. [Part I: The Quest for Machine Vision](#part-i-the-quest-for-machine-vision)
3. [Part II: Classical Computer Vision](#part-ii-classical-computer-vision)
4. [Part III: The Deep Learning Revolution](#part-iii-the-deep-learning-revolution)
5. [Part IV: Modern Applications](#part-iv-modern-applications)
6. [DEEP DIVE: ImageNet and the 2012 Moment](#deep-dive-imagenet-and-the-2012-moment)
7. [Lecture Plan and Hands-On Exercise](#lecture-plan-and-hands-on-exercise)
8. [Recommended Resources](#recommended-resources)
9. [References](#references)

---

# Introduction

Vision seems effortless to humans—we recognize faces, read text, navigate spaces without conscious effort. But teaching computers to see has been one of AI's greatest challenges.

This module explores:
- Why computer vision is so hard
- How early approaches tried to engineer vision
- The deep learning breakthrough that changed everything

**Core Question:** What does it mean for a machine to "see"?

---

# Part I: The Quest for Machine Vision

## The Summer Vision Project (1966)

In 1966, MIT professor Marvin Minsky assigned a summer project to an undergraduate:

> "Connect a camera to a computer and get the computer to describe what it sees."

The project was expected to take one summer. It's now been 60 years and we're still working on it.

### Why Vision Is Hard

What seems "simple" to humans requires:
- Recognizing objects despite variations in lighting, angle, occlusion
- Understanding 3D structure from 2D images
- Distinguishing between millions of object categories
- Making sense of context

A 3-year-old effortlessly recognizes a cat. This took AI research 50+ years.

---

## David Marr's Computational Vision (1982)

David Marr, a neuroscientist at MIT, proposed a theory of how vision works in his book *Vision* (1982).

### Three Levels of Analysis

1. **Computational:** What problem is vision solving?
2. **Algorithmic:** What steps solve the problem?
3. **Implementation:** How does the brain/computer do it?

### Marr's Stages of Vision

1. **Primal sketch:** Edges, textures, basic shapes
2. **2.5D sketch:** Surfaces, depth, orientation
3. **3D model:** Full 3D representation of objects

### Influence

Marr died at 35, but his framework influenced computer vision for decades. His book remains required reading.

### Limitation

Marr's approach was top-down: figure out the theory, then implement it. Modern deep learning is bottom-up: learn from data.

---

## The AI Winter and Computer Vision

### Expert Systems Era (1980s)

Computer vision tried to encode human knowledge:
- Explicit rules for edge detection
- Hand-crafted feature detectors
- Knowledge bases of object properties

### The Problem

Real-world images vary endlessly:
- Lighting changes
- Viewpoint changes
- Occlusion
- Deformation

No amount of rules could handle this variability.

---

# Part II: Classical Computer Vision

## Edge Detection: Sobel, Canny

### Finding Edges

Edges are boundaries between regions—changes in intensity.

**Sobel Operator (1968):**
Detect horizontal and vertical gradients using 3x3 filters.

**Canny Edge Detector (1986):**
- Gaussian smoothing
- Gradient computation
- Non-maximum suppression
- Hysteresis thresholding

### Why Edges Matter

Edges were thought to be the fundamental building blocks:
- Object boundaries
- Surface discontinuities
- Essential structure

But edges alone don't tell you what you're looking at.

---

## Feature Detection: SIFT and HOG

### SIFT - Scale-Invariant Feature Transform (1999)

David Lowe at UBC developed SIFT to find "keypoints" that are:
- Invariant to scale
- Invariant to rotation
- Robust to illumination changes

### HOG - Histogram of Oriented Gradients (2005)

Dalal and Triggs developed HOG for pedestrian detection:
- Divide image into cells
- Compute gradient orientation histogram per cell
- Normalize across blocks

HOG was state-of-the-art for object detection until deep learning.

### The Approach

1. Extract hand-crafted features (SIFT, HOG)
2. Feed features to classifier (SVM)
3. Train on labeled data

This worked, but required engineering the right features for each problem.

---

## The MNIST Dataset (1998)

Yann LeCun and colleagues created MNIST: 70,000 handwritten digits.

### Why MNIST Matters

- **Benchmark:** Standard evaluation for classification
- **Simple:** 28x28 grayscale images
- **Non-trivial:** Still requires learning

### The First CNNs: LeNet

LeCun developed LeNet-5 to recognize MNIST digits using **Convolutional Neural Networks**:
- Convolutional layers detect local patterns
- Pooling layers provide translation invariance
- Fully connected layers for classification

LeNet achieved 99%+ accuracy on MNIST in 1998.

### "MNIST is solved"

Today, even simple models achieve >99.5% on MNIST. It's often called "the hello world of deep learning."

---

# Part III: The Deep Learning Revolution

## The Pieces Come Together

### GPUs for Computing (2006-2010)

NVIDIA GPUs, designed for video games, were repurposed for neural networks:
- Massively parallel computation
- 100x faster than CPUs for matrix operations
- Made deep networks trainable

### Large Datasets

- **ImageNet:** 14 million labeled images
- **COCO:** 330,000 images with detailed annotations
- **Web scraping:** Practically unlimited images

### Algorithmic Improvements

- **ReLU activation:** Faster training than sigmoid
- **Dropout:** Prevents overfitting
- **Batch normalization:** Stabilizes training
- **Better weight initialization**

---

## AlexNet: The 2012 Moment

In the 2012 ImageNet competition, Alex Krizhevsky's neural network (AlexNet) achieved:
- **Top-5 error:** 15.3%
- **Second place:** 26.2%

The gap was unprecedented. Deep learning had arrived.

### Architecture

- 8 layers (5 convolutional, 3 fully connected)
- 60 million parameters
- ReLU activations
- Dropout regularization
- Trained on two GPUs

### Impact

After AlexNet:
- Every ImageNet winner was a deep neural network
- Investment in deep learning exploded
- Computer vision transformed within years

---

## Deeper and Deeper: VGG, ResNet

### VGG (2014)

Oxford's VGG network showed: deeper is better.
- 16-19 layers
- Simple architecture: 3x3 convolutions throughout
- Top-5 error: 7.3%

### The Degradation Problem

But simply adding layers stopped working. Very deep networks trained poorly—not from overfitting, but from optimization difficulties.

### ResNet (2015)

Microsoft's ResNet introduced **skip connections**:

```
output = F(x) + x
```

If the layer can't learn something useful, it can learn F(x) = 0, preserving the input.

### The Result

- 152 layers
- Top-5 error: 3.6%
- **Superhuman performance** (human error ~5%)

---

# Part IV: Modern Applications

## Medical Imaging

### The Promise

AI radiologists could:
- Screen images faster than humans
- Catch things humans miss
- Work 24/7 without fatigue

### Diabetic Retinopathy Detection

Google developed a system to detect diabetic retinopathy from retinal scans:
- Trained on 128,000 images
- Performance comparable to ophthalmologists
- Deployed in India and Thailand

### Challenges

- Requires extensive validation
- Regulatory approval is slow
- Physicians concerned about liability
- Data privacy issues

---

## Self-Driving Cars

### DARPA Grand Challenge (2004-2007)

The US military sponsored competitions for autonomous vehicles:
- **2004:** No vehicle finished the 150-mile desert course
- **2005:** 5 vehicles finished (Stanley won)
- **2007:** Urban Challenge—traffic, intersections, parking

### Modern Autonomous Vehicles

Tesla, Waymo, and others use:
- Multiple cameras
- LiDAR (laser depth sensing)
- Radar
- Deep learning for perception

### The Trolley Problem Goes Real

How should autonomous vehicles handle impossible situations? These philosophical questions become engineering decisions.

---

## Face Recognition

### Eigenfaces (1991)

Turk and Pentland: represent faces as combinations of "eigenfaces" (PCA components).

### DeepFace (2014)

Facebook's DeepFace achieved near-human performance:
- 97.35% accuracy on LFW benchmark
- Used 3D face alignment
- 9-layer deep neural network

### Ethical Concerns

Face recognition raises serious issues:
- **Privacy:** Surveillance without consent
- **Bias:** Higher error rates on darker skin
- **Consent:** Used without subjects' knowledge
- **Misuse:** Authoritarian surveillance

Several cities have banned government use of facial recognition.

---

# DEEP DIVE: ImageNet and the 2012 Moment

## The Vision

In 2006, Fei-Fei Li, a young professor at Princeton (later Stanford), had an audacious idea: create a dataset with every object in the world.

### The Problem

Computer vision was stuck. Researchers used tiny datasets:
- Caltech 101: 9,000 images, 101 categories
- PASCAL VOC: ~10,000 images, 20 categories

Li realized: **data was the bottleneck.**

## Building ImageNet

### The WordNet Foundation

ImageNet organized images according to WordNet, a lexical database:
- 22,000 categories in ImageNet
- Based on English nouns
- Hierarchical structure

### The Scale

- **14 million images**
- **20,000+ categories**
- Labeled by humans

### Amazon Mechanical Turk

How do you label 14 million images? Li's insight: use the crowd.

**Amazon Mechanical Turk:** A platform where workers complete small tasks for small payments.

ImageNet workers:
- Verified whether images matched category labels
- $0.01-0.10 per task
- Quality control through redundancy

### Cost and Time

- Started in 2007
- Took 3 years
- Cost: approximately $50,000 in MTurk payments
- 49,000 workers from 167 countries

## The ImageNet Large Scale Visual Recognition Challenge (ILSVRC)

Starting in 2010, ImageNet hosted an annual competition:
- 1,000 categories
- ~1.2 million training images
- ~50,000 validation images
- ~100,000 test images

### The Metrics

**Top-5 error:** Did the correct label appear in the model's top 5 guesses?
**Top-1 error:** Was the top guess correct?

### Before 2012

Best systems used hand-crafted features (SIFT, HOG) plus classifiers (SVM):
- 2010 winner: 28% top-5 error
- 2011 winner: 26% top-5 error

## AlexNet: The Breakthrough

In 2012, Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton submitted a deep convolutional neural network.

### Results

- **Top-5 error: 15.3%**
- **Second place: 26.2%**
- Gap of **11 percentage points**

### Why It Won

1. **Deep architecture:** 8 layers, learned hierarchical features
2. **GPU training:** Used two NVIDIA GTX 580 GPUs
3. **ReLU activation:** Faster training than sigmoid/tanh
4. **Dropout:** Prevented overfitting
5. **Data augmentation:** Artificially increased training data

### The Paper

"ImageNet Classification with Deep Convolutional Neural Networks" became one of the most cited papers in history.

## The Aftermath

### 2013-2017: Deeper Networks Win

| Year | Winner | Error | Depth |
|------|--------|-------|-------|
| 2012 | AlexNet | 15.3% | 8 |
| 2013 | ZFNet | 11.2% | 8 |
| 2014 | VGG/GoogLeNet | 6.7% | 19/22 |
| 2015 | ResNet | 3.6% | 152 |
| 2017 | SENet | 2.3% | 154 |

### Superhuman Performance

By 2015, ResNet surpassed estimated human performance (~5% error).

### The Competition Ends

In 2017, ImageNet discontinued the classification challenge. The problem was "solved" (for this benchmark).

## The Controversies

### Dataset Bias

ImageNet's images are:
- Predominantly from the internet (Western bias)
- Object-centric (not scenes)
- Static images (not video)

Performance on ImageNet doesn't guarantee real-world performance.

### The Mechanical Turk Workers

The dataset was built on low-wage crowd labor:
- Workers paid cents per task
- No benefits or job security
- Performing repetitive labeling

### Problematic Categories

ImageNet included some troubling categories:
- Racial and ethnic slurs
- Derogatory terms
- Some categories removed in 2019

## The Legacy

### Positive

- Launched the deep learning revolution
- Established benchmarking culture
- Showed the importance of large datasets
- Enabled transfer learning

### Complicated

- Set expectations that more data always helps
- Led to data collection practices without consent
- Concentrated power in organizations that can collect data

### Fei-Fei Li's Reflection

Li has spoken about wanting AI development to be more inclusive and ethical. She later co-founded AI4ALL to diversify the field.

## The Data Journey

- **Collection:** 14 million images labeled by 49,000 workers worldwide
- **Understanding:** Benchmark revealed what's possible with deep learning
- **Prediction:** Pre-trained ImageNet models power countless applications

---

# Lecture Plan and Hands-On Exercise

## Lecture Plan: "Teaching Machines to See" (75-90 minutes)

### Part 1: Why Vision Is Hard (15 min)

**Opening:** Show an image and ask students what they see.
- They'll identify objects instantly
- Reveal: this takes billions of neurons and years of learning

**The 1966 Summer Project:** Minsky's optimism, 60 years later

### Part 2: From Pixels to Features (20 min)

**What is an image to a computer?**
- Grid of numbers (pixels)
- Demo: Load image, show array

**Edge detection:**
- Why edges matter
- Sobel/Canny operators
- Live demo on sample image

**The classical approach:**
1. Extract features (SIFT, HOG)
2. Train classifier (SVM)
3. Predict

### Part 3: The ImageNet Story (20 min)

- Fei-Fei Li's vision
- Building with Mechanical Turk
- The 2012 competition

**Show the graph:** Error rates dropping after 2012

### Part 4: How CNNs Work (15 min)

**Convolution intuition:**
- Filters detect local patterns
- Layer 1: edges
- Layer 2: textures
- Layer 3+: parts, objects

**Show visualizations** of what each layer "sees"

### Part 5: Applications and Ethics (10 min)

- Medical imaging
- Self-driving cars
- Face recognition
- Bias and surveillance concerns

---

## Hands-On Exercise: "Build a Cat vs. Dog Classifier"

### Objective

Train a convolutional neural network to classify images.

### Duration

2-3 hours

### Setup

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# Download cats vs dogs dataset
url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
path = keras.utils.get_file("cats_and_dogs.zip", origin=url, extract=True)
base_dir = path.replace('.zip', '')

train_dir = f"{base_dir}/train"
val_dir = f"{base_dir}/validation"
```

### Task 1: Explore the Data (20 min)

```python
import os
from PIL import Image

# Count images
train_cats = len(os.listdir(f"{train_dir}/cats"))
train_dogs = len(os.listdir(f"{train_dir}/dogs"))
print(f"Training: {train_cats} cats, {train_dogs} dogs")

# View some examples
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, animal in enumerate(['cats', 'dogs']):
    files = os.listdir(f"{train_dir}/{animal}")[:4]
    for j, f in enumerate(files):
        img = Image.open(f"{train_dir}/{animal}/{f}")
        axes[i, j].imshow(img)
        axes[i, j].axis('off')
        axes[i, j].set_title(animal)
plt.tight_layout()
plt.show()
```

**Questions:**
- How varied are the images?
- What challenges might the model face?

### Task 2: Build Data Pipeline (20 min)

```python
# Create data generators with augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
```

### Task 3: Build a Simple CNN (30 min)

```python
model = keras.Sequential([
    # First conv block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),

    # Second conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Third conv block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Dense layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### Task 4: Train the Model (30 min)

```python
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator
)

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss')
ax2.legend()

plt.show()
```

### Task 5: Transfer Learning (30 min)

```python
# Use pre-trained VGG16
base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

# Freeze base model
base_model.trainable = False

# Add our classifier
model_transfer = keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_transfer.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train only the top layers
history_transfer = model_transfer.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)
```

**Compare:** How does transfer learning compare to training from scratch?

### Task 6: Visualize What the Model Sees (20 min)

```python
# Get a sample image
img_path = f"{val_dir}/cats/cat.2000.jpg"
img = keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
x = keras.preprocessing.image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# Get first layer activations
layer_outputs = [layer.output for layer in model.layers[:6]]
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(x)

# Plot first layer filters
first_layer_activation = activations[0]
plt.figure(figsize=(15, 5))
for i in range(min(8, first_layer_activation.shape[-1])):
    plt.subplot(2, 4, i+1)
    plt.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.suptitle('First Conv Layer Activations')
plt.show()
```

---

# Recommended Resources

## Books

- **Goodfellow, Bengio, Courville.** *Deep Learning* (2016) - Chapter on CNNs
- **Chollet, F.** *Deep Learning with Python* (2021) - Practical guide
- **Marr, D.** *Vision* (1982) - The classic theoretical framework

## Online Courses

- **Stanford CS231n:** Convolutional Neural Networks for Visual Recognition
- **fast.ai:** Practical Deep Learning for Coders
- **Coursera:** Deep Learning Specialization (Andrew Ng)

## Tools

- **TensorFlow/Keras:** High-level neural network API
- **PyTorch:** Flexible deep learning framework
- **OpenCV:** Classical computer vision library
- **Torchvision:** Pre-trained models and datasets

## Videos

- **3Blue1Brown:** Neural networks series
- **Stanford CS231n lectures** on YouTube
- **Two Minute Papers:** Latest vision research

---

# References

## Historical
- Marr, D. (1982). *Vision*. MIT Press.
- LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.

## ImageNet and Deep Learning
- Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database. *CVPR*.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *NeurIPS*.
- He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.

## Applications
- Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*.
- Turk, M. & Pentland, A. (1991). Eigenfaces for recognition. *Journal of Cognitive Neuroscience*.

---

*Document compiled for SCDS DATA 201: Introduction to Data Science I*
*Module 7: Vision*
*"Machines that See"*
