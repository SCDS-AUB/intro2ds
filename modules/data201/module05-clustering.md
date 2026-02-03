---
layout: default
title: "Module 5: Unsupervised Learning - Clustering and Dimensionality Reduction"
---

# Module 5: Unsupervised Learning - Clustering and Dimensionality Reduction
## "Discovering Patterns"

*Research Document for DATA 201 Course Development*

---

# Table of Contents

1. [Introduction](#introduction)
2. [Part I: The Quest to Classify](#part-i-the-quest-to-classify)
3. [Part II: Clustering Algorithms](#part-ii-clustering-algorithms)
4. [Part III: Dimensionality Reduction](#part-iii-dimensionality-reduction)
5. [Part IV: Modern Applications](#part-iv-modern-applications)
6. [DEEP DIVE: The Iris Dataset - 85 Years of a Teaching Classic](#deep-dive-the-iris-dataset---85-years-of-a-teaching-classic)
7. [Lecture Plan and Hands-On Exercise](#lecture-plan-and-hands-on-exercise)
8. [Recommended Resources](#recommended-resources)
9. [References](#references)

---

# Introduction

Unsupervised learning finds hidden structure in data without labels. This module explores:

- **Clustering:** Grouping similar items together
- **Dimensionality reduction:** Finding low-dimensional representations
- **The fundamental tension:** What makes things "similar"?

**Core Question:** Can we discover meaningful patterns without being told what to look for?

---

# Part I: The Quest to Classify

## Linnaeus and the Birth of Taxonomy (1735)

Carl Linnaeus revolutionized biology with *Systema Naturae*, creating the hierarchical classification system still used today:

```
Kingdom → Phylum → Class → Order → Family → Genus → Species
```

### The First "Clustering" Algorithm

Linnaeus grouped organisms by observable features:
- Number of stamens (for plants)
- Body structure (for animals)
- Shared characteristics → shared category

### The Data Structure: Trees

Linnaean taxonomy is a **tree** (hierarchical clustering):
- Each node is a cluster
- Children are subclusters
- Leaves are individual species

This structure implies: organisms in the same group are more similar to each other than to organisms in other groups.

### Controversy

Linnaeus's system predated Darwin. Categories were assumed to reflect divine design, not evolutionary relationships. Modern phylogenetics uses genetic data to build trees that reflect evolutionary history—often disagreeing with morphological classification.

---

## Francis Galton's Composite Portraits (1878)

Francis Galton tried to find the "average" face of different groups:
- Criminals
- Tubercular patients
- Different ethnic groups

### The Method

Galton overlaid photographs, creating blurred "average" faces. He hoped to identify facial features associated with criminality or disease.

### The Failure

The composites were remarkably similar across groups. There was no distinct "criminal face."

### The Legacy

Despite the dubious goals, Galton's work pioneered:
- Averaging features across a population
- Early dimensionality reduction (finding what's common)
- The ancestor of eigenfaces

---

# Part II: Clustering Algorithms

## K-Means: Lloyd's Algorithm (1957)

Stuart Lloyd at Bell Labs developed k-means for pulse-code modulation (compressing signals).

### The Algorithm

1. Choose k initial cluster centers
2. Assign each point to the nearest center
3. Update centers to be the mean of assigned points
4. Repeat until convergence

### Why It Works

K-means minimizes within-cluster variance:
$$\sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

### Limitations

- Must choose k in advance
- Assumes spherical clusters
- Sensitive to initialization
- Finds local, not global, optima

### Solutions

- **k-means++**: Smart initialization
- **Elbow method**: Plot variance vs. k, look for "elbow"
- **Silhouette score**: Measure cluster quality

---

## Hierarchical Clustering

### Agglomerative (Bottom-Up)

1. Start with each point as its own cluster
2. Merge the two closest clusters
3. Repeat until one cluster remains

**Linkage methods:**
- Single: Distance between closest points
- Complete: Distance between farthest points
- Average: Average distance between points
- Ward's: Minimize variance increase

### Divisive (Top-Down)

Start with one cluster, recursively split.

### Dendrograms

Hierarchical clustering produces a **dendrogram**—a tree showing merge history. Cut at any height to get different numbers of clusters.

### Advantages

- No need to pre-specify k
- Produces interpretable hierarchy
- Works with any distance metric

---

## DBSCAN: Density-Based Clustering (1996)

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds clusters of varying shapes.

### The Idea

A cluster is a region of high density surrounded by low density.

**Parameters:**
- **ε (epsilon)**: Neighborhood radius
- **minPts**: Minimum points to form dense region

### Point Types

- **Core:** Has ≥ minPts neighbors within ε
- **Border:** Within ε of core point but not core itself
- **Noise:** Neither core nor border

### Advantages

- Discovers clusters of arbitrary shape
- Doesn't require specifying k
- Identifies outliers

### Limitation

Struggles with varying densities.

---

# Part III: Dimensionality Reduction

## Principal Component Analysis (1901-1933)

### Karl Pearson (1901)

Pearson defined PCA geometrically: find the line that minimizes perpendicular distances to points.

### Harold Hotelling (1933)

Hotelling developed the algebraic form: find orthogonal directions of maximum variance.

### The Math

PCA finds the eigenvectors of the covariance matrix:
$$\Sigma = \frac{1}{n} X^T X$$

The first principal component is the eigenvector with the largest eigenvalue.

### Why It Works

High-dimensional data often lies near a lower-dimensional subspace. PCA finds that subspace.

### Applications

- **Noise reduction:** Keep top components, discard rest
- **Visualization:** Project to 2D or 3D
- **Feature extraction:** Use components as new features
- **Compression:** Store only top components

---

## Eigenfaces: PCA for Face Recognition (1991)

Turk and Pentland applied PCA to face recognition.

### The Method

1. Collect many face images (flattened to vectors)
2. Compute mean face
3. Compute PCA on centered faces
4. The eigenvectors are "eigenfaces"

### Face as Linear Combination

Any face ≈ mean face + weighted sum of eigenfaces:
$$\text{face} = \bar{f} + w_1 e_1 + w_2 e_2 + ... + w_k e_k$$

### Recognition

To recognize a face:
1. Project onto eigenface space
2. Compare weights to known faces
3. Closest match is the identity

### Impact

Eigenfaces were state-of-the-art in the 1990s. They've since been superseded by deep learning, but remain pedagogically valuable.

### Sources
- Turk, M. & Pentland, A. (1991). Eigenfaces for Recognition. *Journal of Cognitive Neuroscience*.

---

## t-SNE: Visualization Breakthrough (2008)

t-distributed Stochastic Neighbor Embedding, developed by Laurens van der Maaten and Geoffrey Hinton.

### The Problem with Linear Methods

PCA preserves global structure but loses local neighborhoods. Points that are close in high dimensions may be far apart in 2D.

### The t-SNE Approach

1. Convert distances to probabilities (similar points → high probability)
2. Find 2D arrangement that preserves these probabilities
3. Use t-distribution in low-D space (handles crowding)

### What t-SNE Is Good At

- Visualizing clusters
- Revealing local structure
- Beautiful images for presentations

### What t-SNE Is NOT Good At

- **Not reproducible:** Different runs give different results
- **Not interpretable:** Distances aren't meaningful
- **Not for new data:** Can't project new points

### UMAP (2018)

Uniform Manifold Approximation and Projection offers:
- Better preservation of global structure
- Faster computation
- Support for new data projection

---

# Part IV: Modern Applications

## Gene Expression Clustering

### The Problem

Microarrays measure expression of thousands of genes. Which genes behave similarly? Which samples are similar?

### The Breakthrough: Cancer Subtypes

Clustering gene expression data revealed:
- Breast cancer subtypes invisible to pathologists
- Different subtypes respond differently to treatment
- Molecular classification more predictive than visual

### Famous Studies

- **Golub et al. (1999):** Clustered leukemia samples, distinguished AML from ALL
- **Perou et al. (2000):** Identified breast cancer molecular subtypes

### The Data Journey

- **Collection:** Expression levels for ~20,000 genes across hundreds of patients
- **Understanding:** Clusters reveal biological subtypes
- **Prediction:** Subtype determines treatment

---

## Customer Segmentation

### RFM Analysis

A classic segmentation approach:
- **Recency:** How recently did they purchase?
- **Frequency:** How often do they purchase?
- **Monetary:** How much do they spend?

### K-Means for Marketing

Cluster customers, then tailor:
- Promotions for each segment
- Communication frequency
- Product recommendations

### Example Segments

1. **Champions:** High RFM across all dimensions
2. **Loyal customers:** High frequency, varied recency
3. **At risk:** Once high value, declining activity
4. **Lost:** Haven't purchased in a long time

---

## Anomaly Detection

### What Is an Anomaly?

A point that doesn't belong to any cluster, or is far from cluster centers.

### Applications

- **Fraud detection:** Unusual transactions
- **Network intrusion:** Abnormal traffic patterns
- **Manufacturing:** Defect detection
- **Medical:** Disease identification

### Methods

- Distance to nearest cluster center
- Isolation Forest
- Local Outlier Factor
- One-class SVM

---

# DEEP DIVE: The Iris Dataset - 85 Years of a Teaching Classic

## The Story

The Iris dataset is the most famous dataset in machine learning history. Published by statistician Ronald Fisher in 1936, it contains measurements of 150 iris flowers from three species.

### The Measurements

For each flower:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)
- Species: setosa, versicolor, or virginica

### The Collector: Edgar Anderson

The data was actually collected by botanist Edgar Anderson in the Gaspé Peninsula, Quebec, Canada. Fisher acknowledged Anderson in his paper but the dataset is typically called "Fisher's Iris data."

### Fisher's Paper (1936)

"The Use of Multiple Measurements in Taxonomic Problems" introduced **Linear Discriminant Analysis (LDA)**—a method to find directions that best separate classes.

## Why This Dataset Is Everywhere

### Perfect for Teaching

- **Small:** 150 samples, 4 features—fits on a page
- **Real:** Actual measurements, not synthetic
- **Clean:** No missing values
- **Separable:** Clear cluster structure
- **Challenging:** Two species overlap

### Built Into Everything

```python
from sklearn.datasets import load_iris
iris = load_iris()
```

The Iris dataset is included in:
- R (built-in as `iris`)
- scikit-learn
- TensorFlow datasets
- Every statistics textbook

### The Pedagogical Journey

Iris perfectly illustrates:
1. **Exploratory analysis:** Scatter plots, pairwise comparisons
2. **Clustering:** K-means finds (nearly) the species
3. **Classification:** Train/test split, accuracy
4. **Dimensionality reduction:** PCA visualizes in 2D

## The Clustering Challenge

### What K-Means Finds

With k=3, k-means recovers species with about 89% accuracy:
- Setosa: Perfectly separated
- Versicolor and Virginica: Some overlap

### Why It's Not Perfect

Versicolor and virginica overlap in the feature space. No amount of clustering will perfectly separate them—they're genuinely similar.

This demonstrates: **Clustering reflects the data, not necessarily ground truth labels.**

## Ethical Considerations

### The Dark Connection

Fisher was a prominent eugenicist. His statistical methods (including LDA from the Iris paper) were developed partly to support eugenic research.

### Modern Discussion

Some argue for retiring the Iris dataset as a teaching example, given Fisher's troubling views. Others argue the dataset itself is innocuous (flower measurements) and separating the science from the scientist is possible.

### Alternatives

- Penguin dataset (Palmer Penguins)
- Wine dataset
- Digit datasets (MNIST)

## Hands-On with Iris

```python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Compare to true labels
from sklearn.metrics import adjusted_rand_score
print(f"Adjusted Rand Index: {adjusted_rand_score(y, clusters):.3f}")

# PCA visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis')
plt.title("True Species")

plt.subplot(1, 2, 2)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis')
plt.title("K-Means Clusters")

plt.tight_layout()
plt.show()
```

---

# Lecture Plan and Hands-On Exercise

## Lecture Plan: "Finding Hidden Structure" (75-90 minutes)

### Part 1: The Clustering Intuition (15 min)

**Opening Activity:** Show unlabeled scatter plot of Iris data (2D PCA).

"How many groups do you see? Where are the boundaries?"

Students will naturally identify clusters—this is what clustering algorithms formalize.

### Part 2: The Iris Story (10 min)

- Edgar Anderson's fieldwork
- Fisher's statistical contribution
- Why this dataset is everywhere
- The ethical complexities

### Part 3: K-Means Algorithm (25 min)

**Live Demo:** Step through k-means iteration by iteration:
1. Random initialization
2. Assignment step
3. Update step
4. Watch convergence

**Key Questions:**
- What if we initialize differently?
- How do we choose k?
- What shapes can k-means find?

### Part 4: Beyond K-Means (15 min)

- Hierarchical clustering (show dendrogram)
- DBSCAN (show arbitrary-shape clusters)
- When to use each method

### Part 5: Dimensionality Reduction (15 min)

- Why reduce dimensions?
- PCA: Finding directions of maximum variance
- t-SNE: Preserving local structure
- The visualization power of 2D

---

## Hands-On Exercise: "Cluster the World"

### Objective

Apply clustering to a real-world dataset and interpret the results.

### Duration

2-3 hours

### Dataset Options

1. **Mall Customer Segmentation:** CustomerID, Age, Income, Spending Score
2. **Country Development Indicators:** GDP, life expectancy, education, etc.
3. **Spotify Songs:** Danceability, energy, tempo, valence, etc.

### Task 1: Exploratory Analysis (30 min)

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('your_data.csv')

# Examine distributions
print(data.describe())

# Pairplot (if not too many features)
sns.pairplot(data)
plt.show()

# Correlation heatmap
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
```

**Questions to answer:**
- How many samples? Features?
- Are features on similar scales? (Need standardization?)
- Any obvious patterns or outliers?

### Task 2: K-Means Clustering (45 min)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Elbow method
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Choose k and fit final model
k = ???  # Your choice based on elbow
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add clusters to data
data['Cluster'] = clusters
```

**Questions:**
- Where is the "elbow"?
- Try silhouette score for different k values

### Task 3: Interpret the Clusters (30 min)

```python
# Profile each cluster
cluster_profiles = data.groupby('Cluster').mean()
print(cluster_profiles)

# Visualize profiles
cluster_profiles.T.plot(kind='bar', figsize=(12, 6))
plt.title('Cluster Profiles')
plt.ylabel('Mean Value (standardized)')
plt.show()

# Name your clusters!
# Based on the profiles, what would you call each cluster?
```

**Give meaningful names to your clusters** (e.g., "High Spenders," "Budget Conscious," etc.)

### Task 4: Visualize with PCA (30 min)

```python
from sklearn.decomposition import PCA

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot with cluster colors
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Clusters in PCA Space')
plt.show()

# What do the principal components represent?
print("PC1 loadings:", pca.components_[0])
print("PC2 loadings:", pca.components_[1])
```

### Task 5: Report (30 min)

Write a brief report (1-2 pages) including:
1. **Dataset description:** What are you clustering?
2. **Method:** How did you choose k? What did the elbow look like?
3. **Results:** Describe each cluster in plain language
4. **Visualization:** Include your PCA plot
5. **Business/Scientific Implications:** What would you do with these clusters?

---

# Recommended Resources

## Books

- **James, Witten, Hastie & Tibshirani.** *An Introduction to Statistical Learning* (ISLR) - Chapter 12 on Unsupervised Learning
- **Murphy, K.** *Machine Learning: A Probabilistic Perspective* (2012) - Rigorous treatment
- **Hastie, Tibshirani & Friedman.** *Elements of Statistical Learning* - Advanced

## Online Courses

- **Coursera:** Machine Learning (Andrew Ng) - Week 8 on clustering
- **StatQuest:** Hierarchical Clustering, K-means, PCA videos
- **3Blue1Brown:** PCA explanation

## Interactive Tools

- **Setosa.io:** Visual explanations of PCA, k-means
- **Explained Visually:** PCA visualization
- **K-Means Visualization:** https://www.naftaliharris.com/blog/visualizing-k-means-clustering/

## Datasets

- **UCI Machine Learning Repository:** Dozens of clustering datasets
- **Kaggle:** Customer segmentation datasets
- **Gapminder:** Country indicators for clustering

---

# References

## Historical
- Linnaeus, C. (1735). *Systema Naturae*.
- Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179-188.
- Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*, 2(11), 559-572.

## Algorithms
- Lloyd, S. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory* (paper from 1957).
- Ester, M., et al. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD*.
- van der Maaten, L. & Hinton, G. (2008). Visualizing data using t-SNE. *JMLR*.

## Applications
- Turk, M. & Pentland, A. (1991). Eigenfaces for recognition. *Journal of Cognitive Neuroscience*.
- Golub, T. R., et al. (1999). Molecular classification of cancer. *Science*.

---

*Document compiled for SCDS DATA 201: Introduction to Data Science I*
*Module 5: Unsupervised Learning*
*"Discovering Patterns"*
