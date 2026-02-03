---
layout: default
title: "Module 10: Regression and Classification - Predicting Outcomes"
---

# Module 10: Regression and Classification - Predicting Outcomes

## Introduction

At the heart of supervised machine learning lies a simple question: given what we know, can we predict what we don't? Will this email be spam? What price will this house sell for? Will this patient develop diabetes? Is this transaction fraudulent?

These questions divide into two types. **Regression** predicts continuous values—prices, temperatures, durations, quantities. **Classification** predicts categories—spam or not spam, malignant or benign, approved or denied. Together, they form the foundation of predictive modeling, turning historical data into actionable predictions about the future.

This module traces the intellectual history of these methods, from a Victorian scientist's study of heredity to the sophisticated ensemble methods that power modern AI systems. We'll learn not just how these algorithms work, but why they were invented and how they reflect evolving ideas about the relationship between inputs and outputs.

---

## Part 1: The Origins of Regression

### Francis Galton and the Birth of Regression (1886)

The word "regression" has a curious history. It does not mean what it sounds like—going backward—but rather refers to a specific statistical phenomenon first observed by **Francis Galton** (1822-1911), the Victorian polymath who gave us fingerprinting, weather maps, and the foundations of statistics.

Galton was obsessed with heredity and eugenics (a field he named and promoted, now rightfully discredited). In the 1880s, he conducted a famous experiment: he grew sweet pea seeds, carefully measured the sizes of parent seeds and their offspring, and noticed something unexpected.

Large parent seeds did indeed produce offspring that were larger than average. But they weren't *as* large as their parents. Small parent seeds produced offspring that were smaller than average—but not as small as their parents. Everything "regressed" toward the mean.

Galton then turned to humans. He collected data on the heights of parents and adult children. He found the same pattern: tall parents had children who were tall, but on average less tall than themselves. Short parents had children who were short, but on average taller than themselves.

This was **regression to the mean**—the tendency of extreme values to be followed by less extreme values. It's not that nature "corrects" extremes; it's that extreme values are partly due to chance, and chance doesn't repeat perfectly.

### Karl Pearson and the Correlation Coefficient

Galton's student **Karl Pearson** (1857-1936) took the next step. He developed the mathematical framework we still use today:

The **correlation coefficient** (r) measures the strength and direction of a linear relationship between two variables, ranging from -1 (perfect negative) to +1 (perfect positive).

The **regression line** is the best-fitting straight line through a scatter plot—the line that minimizes the sum of squared vertical distances from each point to the line (the method of least squares, invented earlier by Gauss and Legendre).

The equation:
$$\hat{y} = \beta_0 + \beta_1 x$$

where $\beta_1 = r \cdot \frac{s_y}{s_x}$ (slope) and $\beta_0 = \bar{y} - \beta_1 \bar{x}$ (intercept).

### Least Squares: From Gauss to Machine Learning

The method of least squares—finding parameters that minimize squared errors—predates Galton by decades. **Carl Friedrich Gauss** (1777-1855) and **Adrien-Marie Legendre** (1752-1833) developed it independently in the early 1800s for astronomical calculations.

Gauss used it to predict the orbit of the asteroid Ceres from just a few observations—his success made him famous throughout Europe. The method works because:
1. It has a closed-form solution (for linear problems)
2. It's computationally tractable
3. It has optimal properties if errors are normally distributed

Today, least squares remains the foundation—gradient descent for neural networks is just iterative least squares for complex, nonlinear functions.

---

## Part 2: From Linear to Logistic - Classification Emerges

### The Need for Classification

Regression predicts numbers, but many predictions are inherently categorical:
- Will this tumor be malignant or benign?
- Will this customer default on their loan?
- Is this email spam or legitimate?

You could try to use linear regression for classification (predict 0 or 1, threshold at 0.5), but this causes problems: predictions can go below 0 or above 1, the relationship between inputs and probability isn't linear, and extreme values distort the fit.

### The Logit Transform

The solution came from **logistic regression**, which predicts the *probability* of class membership and keeps predictions bounded between 0 and 1.

The key is the **logit function** (or log-odds):
$$\text{logit}(p) = \log\left(\frac{p}{1-p}\right)$$

And its inverse, the **sigmoid function**:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Instead of modeling the outcome directly, we model the log-odds as a linear function of predictors:
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ...$$

The sigmoid squashes any real number into the (0, 1) interval—perfect for probabilities.

### The History of Logistic Regression

The logistic function was introduced by **Pierre François Verhulst** in the 1830s-1840s to model population growth with resource constraints (the S-curve of growth that slows as carrying capacity is reached).

Its use for classification developed through the mid-20th century, with key contributions from:
- **Joseph Berkson** (1944): coined "logit" and advocated logistic models in biostatistics
- **David Cox** (1958): the theoretical foundations in regression context
- **Jerome Cornfield** (1962): applied to epidemiology and disease risk

### Maximum Likelihood Estimation

Unlike linear regression, logistic regression can't be solved in closed form. Instead, we use **maximum likelihood estimation (MLE)**—finding the parameters that make the observed data most probable.

For binary outcomes:
$$L(\beta) = \prod_{i=1}^{n} p_i^{y_i} (1-p_i)^{1-y_i}$$

We maximize this (or equivalently, minimize the negative log-likelihood) using iterative optimization—Newton's method or gradient descent.

---

## Part 3: Beyond the Line - Non-Linear Methods

### Polynomial Regression

The simplest extension of linear regression adds polynomial terms:
$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + ...$$

This is still "linear regression" because it's linear in the *parameters* (the βs), even though it's non-linear in x.

### k-Nearest Neighbors (k-NN)

Perhaps the simplest non-parametric method: to predict for a new point, find the k closest training points and:
- For regression: average their values
- For classification: take the majority vote

k-NN makes no assumptions about the form of the relationship. It can capture arbitrarily complex patterns. But it suffers from the **curse of dimensionality**: in high dimensions, "nearest neighbors" become meaningless because all points are approximately equidistant.

### Decision Trees

**Decision trees** partition the feature space through a series of yes/no questions:
- Is income > $50,000?
  - Yes: Is age > 35?
    - Yes: Class A
    - No: Class B
  - No: Class C

Invented in various forms over decades, decision trees were popularized by **Leo Breiman** and colleagues with their CART algorithm (Classification and Regression Trees) in 1984.

Trees are interpretable—you can follow the decision path—but they're unstable (small changes in data can produce very different trees) and prone to overfitting.

### Ensemble Methods: Random Forests and Boosting

The instability of trees became a feature, not a bug:

**Random Forests** (Leo Breiman, 2001): Build many trees, each on a random subset of data and features. Average their predictions. The randomness decorrelates the trees, reducing variance.

**Boosting**: Build trees sequentially, each one focusing on the errors of the previous ones. AdaBoost (Freund & Schapire, 1997) and Gradient Boosting (Friedman, 2001) became the dominant methods for structured data.

**XGBoost** (Tianqi Chen, 2014) optimized gradient boosting for speed and accuracy, becoming the winning algorithm in most Kaggle competitions. LightGBM and CatBoost followed with further improvements.

---

## Part 4: Support Vector Machines

### The Maximum Margin Classifier

In 1963, **Vladimir Vapnik** and **Alexey Chervonenkis** developed the foundations of statistical learning theory. Their work led to **Support Vector Machines (SVMs)** in the 1990s.

The key insight: for classification, find the hyperplane that maximizes the margin—the distance to the nearest training points (the "support vectors"). This provides theoretical guarantees about generalization.

### The Kernel Trick

SVMs became powerful through the **kernel trick**: instead of working in the original feature space, implicitly map the data to a higher-dimensional space where linear separation becomes possible.

The Gaussian (RBF) kernel effectively maps to an infinite-dimensional space:
$$K(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)$$

For a brief period in the late 1990s and early 2000s, SVMs were the state of the art for many classification tasks—until neural networks resurged.

---

## Part 5: Evaluation and Model Selection

### The Bias-Variance Tradeoff

Every prediction model balances two sources of error:

**Bias**: Error from oversimplifying—missing the true pattern. A linear model for a curved relationship has high bias.

**Variance**: Error from being too sensitive to training data. A model that fits noise as if it were signal has high variance.

Total Error = Bias² + Variance + Irreducible Noise

Simple models (high bias, low variance) underfit. Complex models (low bias, high variance) overfit. The sweet spot lies in between.

### Cross-Validation

How do we estimate how well a model will perform on new data?

**Holdout validation**: Split data into training and test sets. Simple but wastes data.

**k-Fold Cross-Validation**: Split data into k folds. Train on k-1 folds, test on the remaining fold. Repeat k times. Average the scores.

**Leave-One-Out**: k-fold with k = n (each observation is a test set once). Computationally expensive but uses all data.

### Regression Metrics

**Mean Squared Error (MSE)**: $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ - emphasizes large errors

**Root Mean Squared Error (RMSE)**: $\sqrt{MSE}$ - same units as outcome

**Mean Absolute Error (MAE)**: $\frac{1}{n}\sum|y_i - \hat{y}_i|$ - robust to outliers

**R² (Coefficient of Determination)**: Proportion of variance explained. $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$

### Classification Metrics

**Accuracy**: Proportion correct. Can be misleading with imbalanced classes.

**Confusion Matrix**: Table of true positives, false positives, true negatives, false negatives.

**Precision**: $\frac{TP}{TP + FP}$ - of predicted positives, how many are correct?

**Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ - of actual positives, how many did we find?

**F1 Score**: Harmonic mean of precision and recall.

**AUC-ROC**: Area under the Receiver Operating Characteristic curve. Measures discrimination ability across all thresholds.

---

## Part 6: Regularization - Controlling Complexity

### The Problem of Overfitting

With enough parameters, any model can fit the training data perfectly—even noise. But such a model fails on new data. This is **overfitting**.

### Ridge Regression (L2)

Add a penalty for large coefficients:
$$\text{minimize } \sum(y_i - \hat{y}_i)^2 + \lambda \sum \beta_j^2$$

This shrinks coefficients toward zero, reducing variance at the cost of some bias.

### Lasso Regression (L1)

Use the absolute value of coefficients as penalty:
$$\text{minimize } \sum(y_i - \hat{y}_i)^2 + \lambda \sum |\beta_j|$$

Lasso can shrink coefficients all the way to zero, performing automatic feature selection.

### Elastic Net

Combines L1 and L2 penalties, getting the best of both: sparsity from Lasso, stability from Ridge.

### The Regularization Parameter

The strength of regularization (λ) is a hyperparameter that must be tuned, typically using cross-validation. Too little regularization → overfitting. Too much → underfitting.

---

## Part 7: Modern Supervised Learning Practice

### Feature Engineering

Raw data rarely works well directly. **Feature engineering** transforms raw inputs into representations that help the model:
- **Scaling**: Standardize features to mean 0, std 1
- **Encoding**: Convert categories to numbers (one-hot, target encoding)
- **Interactions**: Create products of features
- **Transformations**: Log, square root, polynomial terms
- **Binning**: Convert continuous to categorical
- **Domain knowledge**: Create features that capture relevant information

Feature engineering often matters more than algorithm choice.

### Handling Imbalanced Data

When one class is rare (fraud detection, disease diagnosis):
- **Oversampling**: Create synthetic minority examples (SMOTE)
- **Undersampling**: Reduce majority class
- **Class weights**: Penalize errors on minority class more
- **Threshold adjustment**: Lower the classification threshold
- **Anomaly detection**: Treat minority as anomalies

### The ML Pipeline

A complete machine learning pipeline:
1. **Data collection and exploration**
2. **Data cleaning** (missing values, outliers)
3. **Feature engineering**
4. **Train/test split**
5. **Model selection and hyperparameter tuning** (with cross-validation)
6. **Evaluation on holdout test set**
7. **Interpretation and validation**
8. **Deployment and monitoring**

### AutoML

Automated machine learning systems (AutoML) automate steps 3-6:
- **Auto-sklearn**, **H2O AutoML**, **TPOT**: Search over algorithms and hyperparameters
- **AutoFeat**, **Featuretools**: Automated feature engineering
- **Google Cloud AutoML**, **AWS SageMaker Autopilot**: Cloud-based solutions

---

## DEEP DIVE: Francis Galton and the Discovery of Regression

### The Polymath's Obsession

Francis Galton was one of the most remarkable minds of the Victorian era—and one of the most troubling. A half-cousin of Charles Darwin, he made fundamental contributions to meteorology, psychology, genetics, and statistics. He also founded eugenics, believing that human society could be improved through selective breeding. His science was brilliant; his application of it was morally catastrophic.

But setting aside the darkness (which we must acknowledge and not minimize), Galton's statistical discoveries revolutionized how we understand the world. And it all started with peas.

### The Sweet Pea Experiment (1877)

In 1877, Galton distributed seeds from the same sweet pea plant to friends across Britain. He carefully measured each seed's weight before sending it and asked his friends to grow the peas, collect the offspring seeds, and return them for measurement.

The experiment was designed to understand heredity. Darwin's *Origin of Species* had been published in 1859, but the mechanisms of inheritance remained mysterious (Mendel's work was ignored until 1900). Galton wanted to quantify how traits passed from parents to offspring.

When the data came back, Galton plotted parent seed size against offspring seed size. He saw a linear relationship—larger parents produced larger offspring—but something odd: the slope was less than 1.

If a parent seed was 1 standard deviation above average, the offspring was only about 0.33 standard deviations above average. The offspring were *less extreme* than their parents.

Galton called this phenomenon **reversion**—later renamed **regression to the mean**.

### From Peas to People

Galton then turned to humans, studying family records of height across generations. In 1886, he published "Regression Towards Mediocrity in Hereditary Stature," presenting data on 928 adult children and their parents.

He invented several techniques for this analysis:

**The ellipse of equal frequency**: Galton drew contour lines on his scatter plot, showing that parent and child heights formed an elliptical distribution—the first visualization of bivariate data.

**The regression line**: He fitted a line through the data showing that if the mid-parent height was 1 inch above average, the child's expected height was only about 2/3 inch above average.

**The correlation coefficient**: He quantified the strength of association, though his student Karl Pearson would formalize this.

### Understanding Regression to the Mean

Galton initially thought regression was a biological phenomenon—nature "correcting" extremes. But he eventually realized it was mathematical. Consider why:

A tall person is tall partly because of genetics (which can be inherited) and partly because of random factors during development (which are not inherited). Their children inherit the genetic component but not the random component. So children of tall parents are, on average, less extremely tall.

This works both ways: short parents have children who are, on average, taller than themselves. The population isn't collapsing toward the mean—it's a statistical phenomenon about individual extremes, not about population change.

### The Regression Fallacy

Regression to the mean creates a cognitive trap. After an extreme event, things tend to become less extreme—but we often attribute this to our interventions:

- A student has a terrible test score; we tutor them; they improve. Was it the tutoring, or regression to the mean?
- An athlete has a career-best year; they appear on a magazine cover; their next year is worse. The "Sports Illustrated jinx" is just regression.
- A CEO takes over during a crisis; things improve; we credit their leadership. But things were likely to improve anyway.

Understanding regression to the mean is essential for evaluating interventions, treatments, and policies.

### Galton's Legacy

Galton invented:
- The correlation coefficient (in concept; Pearson formalized it)
- Regression analysis
- The standard deviation (he called it "probable error")
- The percentile
- Fingerprint analysis for identification
- The weather map (with isobars)
- Survey questionnaires
- Twin studies for nature vs. nurture

His 1889 book *Natural Inheritance* laid out the foundations of modern statistics.

Karl Pearson, his protégé, built on this work to create the mathematical framework we use today. Together with R.A. Fisher (who would develop ANOVA, maximum likelihood, and experimental design), they established statistics as a discipline.

### The Dark Side

We cannot discuss Galton without confronting his eugenics. He coined the term in 1883 and spent decades promoting the idea that human society could be improved by encouraging the "fit" to reproduce and discouraging the "unfit."

These ideas, dressed in the authority of science, had horrific consequences: forced sterilization laws in the United States (upheld by the Supreme Court in *Buck v. Bell*, 1927), the horrors of Nazi racial policies, and continuing discrimination.

The lesson is sobering: brilliant science can be put to terrible uses. Statistical tools are not neutral—they can be used to justify inequality, discrimination, and violence. Every data scientist must grapple with the ethical implications of their work.

### Why This Story Matters

Galton's story illustrates several critical themes:

1. **Discovery through data**: Galton's breakthroughs came from careful measurement and visualization. He let the data reveal patterns rather than imposing theories.

2. **The power of abstraction**: The regression line—a simple equation relating input to output—became the foundation for all of supervised learning.

3. **Statistical intuition**: Regression to the mean is subtle and counterintuitive. Understanding it prevents common reasoning errors.

4. **Responsibility**: Technical brilliance doesn't guarantee ethical wisdom. Data scientists must think carefully about how their work affects people.

---

## LECTURE PLAN: From Galton's Peas to Modern Prediction

### Learning Objectives
By the end of this lecture, students will be able to:
1. Explain regression to the mean and why it matters
2. Fit and interpret simple linear regression models
3. Understand logistic regression for classification
4. Evaluate models using appropriate metrics
5. Recognize the bias-variance tradeoff

### Lecture Structure (90 minutes)

#### Opening Hook (8 minutes)
**The Height Paradox**
- Ask students: "If tall parents have tall children, why doesn't everyone eventually become the same height?"
- Present Galton's puzzle
- Show the original scatter plot of parent and child heights
- Introduce regression to the mean through this historical mystery

#### Part 1: Linear Regression Foundations (20 minutes)

**From Data to Line (8 minutes)**
- Plot a simple scatter plot (two variables)
- Ask: "What's the best line to summarize this relationship?"
- Intuition: minimize the vertical distances (residuals)
- The formula: $\hat{y} = \beta_0 + \beta_1 x$
- Interactive: show how changing β₀ and β₁ changes the line

**Least Squares (6 minutes)**
- Why squared errors? (Differentiable, penalizes large errors)
- The closed-form solution: $\beta_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$
- Connection to correlation: $\beta_1 = r \cdot \frac{s_y}{s_x}$
- Demo: fit a line in Python, show coefficients

**Interpretation (6 minutes)**
- Slope: "For every 1-unit increase in X, Y changes by β₁"
- Intercept: "When X = 0, Y = β₀"
- R²: "What proportion of variance does X explain?"
- Important: correlation ≠ causation!

#### Part 2: Classification with Logistic Regression (20 minutes)

**Why Not Linear Regression for Categories? (5 minutes)**
- Demo: fit linear regression to binary outcome
- Problems: predictions outside [0, 1], wrong functional form
- The solution: predict probability, not outcome

**The Logistic Function (8 minutes)**
- The S-curve (sigmoid): $\sigma(z) = \frac{1}{1 + e^{-z}}$
- Squashes any input to (0, 1)—perfect for probability
- The model: $P(Y=1) = \sigma(\beta_0 + \beta_1 x)$
- Demo: show how sigmoid transforms linear combination

**Interpretation in Logistic Regression (7 minutes)**
- Log-odds (logit): $\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x$
- Coefficients as log-odds ratios
- Exponentiated coefficient = odds ratio
- Example: if exp(β₁) = 2, the odds double for each unit increase in X

#### Part 3: Evaluation and the Bias-Variance Tradeoff (18 minutes)

**Regression Metrics (5 minutes)**
- MSE, RMSE, MAE: when to use which
- R²: what it tells us, and its limitations
- Live demo: calculate metrics on a real dataset

**Classification Metrics (7 minutes)**
- Accuracy and its limitations (imbalanced classes)
- Confusion matrix: TP, FP, TN, FN
- Precision vs. recall: the tradeoff
- ROC curve and AUC: threshold-independent evaluation
- Demo: plot confusion matrix and ROC curve

**Bias-Variance Tradeoff (6 minutes)**
- Draw the classic U-shaped curves
- Underfitting: too simple, misses pattern (high bias)
- Overfitting: too complex, memorizes noise (high variance)
- The sweet spot: just complex enough
- Demo: polynomial regression with increasing degree

#### Part 4: Beyond Linear Models (15 minutes)

**Decision Trees (5 minutes)**
- The intuition: a flowchart of questions
- Demo: visualize a simple decision tree
- Pros: interpretable, handles non-linear relationships
- Cons: unstable, prone to overfitting

**Ensemble Methods (5 minutes)**
- Random Forest: many trees, random subsets, vote/average
- Gradient Boosting: sequential, each tree fixes previous errors
- Why ensembles work: wisdom of crowds

**Regularization (5 minutes)**
- The problem: too many features, overfitting
- Ridge (L2): shrink coefficients
- Lasso (L1): drive coefficients to zero (feature selection)
- Cross-validation to choose λ

#### Part 5: Practical Considerations (5 minutes)

**The ML Pipeline**
- Data cleaning → Feature engineering → Train/test split → Model selection → Evaluation
- The importance of holdout test sets
- Cross-validation for hyperparameter tuning

**Regression to the Mean in Practice**
- Return to opening question: Galton's insight applies everywhere
- Examples: sports performance, medical interventions, business cycles
- Key lesson: expect extremes to become less extreme

#### Wrap-Up (4 minutes)
- Recap: linear regression → logistic regression → evaluation → beyond
- Galton's legacy: both technical and cautionary
- Preview the hands-on exercise
- Key message: "The best model is the simplest one that works"

### Materials Needed
- Scatter plots from Galton's original data
- Interactive visualization of regression lines
- Confusion matrix examples
- Python notebooks with live demonstrations

### Discussion Questions
1. Why did Galton call it "regression" if it's about prediction?
2. When would you choose logistic regression over a decision tree?
3. How would you explain regression to the mean to someone who thinks tutoring always works?
4. What features would you engineer to predict house prices?

---

## HANDS-ON EXERCISE: Predicting Survival on the Titanic

### Overview
In this exercise, students will:
1. Explore and prepare the Titanic dataset
2. Build regression and classification models
3. Evaluate model performance with appropriate metrics
4. Compare different algorithms and interpret results

### Prerequisites
- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

### Setup

```python
# Install required packages
# pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, mean_squared_error, r2_score)

import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
```

### Part 1: Data Loading and Exploration (15 minutes)

```python
# Load the Titanic dataset
# You can download from: https://www.kaggle.com/c/titanic/data
# Or use seaborn's built-in version:
titanic = sns.load_dataset('titanic')

print("Dataset shape:", titanic.shape)
print("\nColumn names:", titanic.columns.tolist())
print("\nFirst few rows:")
titanic.head()
```

**Task 1.1**: Explore the data structure

```python
# Data types and missing values
print("Data types:")
print(titanic.dtypes)
print("\nMissing values:")
print(titanic.isnull().sum())

# Basic statistics
print("\nSummary statistics:")
titanic.describe()
```

**Task 1.2**: Visualize survival by different features

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Survival rate by class
sns.countplot(data=titanic, x='pclass', hue='survived', ax=axes[0, 0])
axes[0, 0].set_title('Survival by Class')

# Survival rate by sex
sns.countplot(data=titanic, x='sex', hue='survived', ax=axes[0, 1])
axes[0, 1].set_title('Survival by Sex')

# Age distribution by survival
sns.histplot(data=titanic, x='age', hue='survived', bins=30, ax=axes[0, 2])
axes[0, 2].set_title('Age Distribution by Survival')

# Survival rate by embarkation port
sns.countplot(data=titanic, x='embarked', hue='survived', ax=axes[1, 0])
axes[1, 0].set_title('Survival by Embarkation')

# Fare distribution by survival
sns.boxplot(data=titanic, x='survived', y='fare', ax=axes[1, 1])
axes[1, 1].set_title('Fare by Survival')

# Survival by siblings/spouse aboard
sns.countplot(data=titanic, x='sibsp', hue='survived', ax=axes[1, 2])
axes[1, 2].set_title('Survival by Siblings/Spouse')

plt.tight_layout()
plt.show()
```

### Part 2: Data Preparation (20 minutes)

```python
# Create a copy for processing
df = titanic.copy()

# Handle missing values
# Age: fill with median by class and sex
df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(
    lambda x: x.fillna(x.median())
)

# Embarked: fill with mode
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Drop columns we won't use
df = df.drop(['deck', 'alive', 'embark_town', 'class', 'who', 'adult_male'], axis=1)

print("Missing values after cleaning:")
print(df.isnull().sum())
```

**Task 2.1**: Feature engineering

```python
# Create new features
# Family size
df['family_size'] = df['sibsp'] + df['parch'] + 1

# Is alone?
df['is_alone'] = (df['family_size'] == 1).astype(int)

# Age categories
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100],
                          labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

# Fare per person
df['fare_per_person'] = df['fare'] / df['family_size']

print("\nNew features:")
print(df[['family_size', 'is_alone', 'age_group', 'fare_per_person']].head(10))
```

**Task 2.2**: Encode categorical variables

```python
# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['sex', 'embarked', 'age_group'], drop_first=True)

# Print the resulting features
print("Features after encoding:")
print(df_encoded.columns.tolist())
```

### Part 3: Building Classification Models (30 minutes)

```python
# Prepare features and target
X = df_encoded.drop(['survived'], axis=1)
y = df_encoded['survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Survival rate in training set: {y_train.mean():.2%}")
print(f"Survival rate in test set: {y_test.mean():.2%}")

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Task 3.1**: Logistic Regression

```python
# Fit logistic regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Predictions
y_pred_log = log_reg.predict(X_test_scaled)
y_prob_log = log_reg.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("Logistic Regression Results:")
print(classification_report(y_test, y_pred_log))

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': log_reg.coef_[0],
    'odds_ratio': np.exp(log_reg.coef_[0])
}).sort_values('coefficient', ascending=False)

print("\nFeature Coefficients (top 10):")
print(feature_importance.head(10))
```

**Task 3.2**: Decision Tree

```python
# Fit decision tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# Predictions
y_pred_tree = tree.predict(X_test)

# Evaluate
print("Decision Tree Results:")
print(classification_report(y_test, y_pred_tree))

# Visualize the tree (top levels)
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, class_names=['Died', 'Survived'],
          filled=True, rounded=True, max_depth=3)
plt.title('Decision Tree (First 3 Levels)')
plt.tight_layout()
plt.show()
```

**Task 3.3**: Random Forest

```python
# Fit random forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Evaluate
print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf))

# Feature importance
feature_importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(feature_importance_rf.head(10))

# Visualize
plt.figure(figsize=(12, 6))
top_features = feature_importance_rf.head(10)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### Part 4: Model Comparison and Evaluation (20 minutes)

```python
def evaluate_model(name, y_true, y_pred, y_prob=None):
    """Compute evaluation metrics."""
    results = {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        results['AUC'] = auc(fpr, tpr)
    return results

# Compare models
models_to_compare = [
    ('Logistic Regression', y_pred_log, y_prob_log),
    ('Decision Tree', y_pred_tree, None),
    ('Random Forest', y_pred_rf, y_prob_rf),
]

# Add Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
y_prob_gb = gb.predict_proba(X_test)[:, 1]
models_to_compare.append(('Gradient Boosting', y_pred_gb, y_prob_gb))

# Compute metrics for all models
results_list = []
for name, y_pred, y_prob in models_to_compare:
    results_list.append(evaluate_model(name, y_test, y_pred, y_prob))

comparison_df = pd.DataFrame(results_list)
print("\nModel Comparison:")
print(comparison_df.to_string(index=False))
```

**Task 4.1**: Plot ROC curves

```python
plt.figure(figsize=(10, 8))

for name, _, y_prob in models_to_compare:
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
```

**Task 4.2**: Analyze confusion matrices

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, y_pred, _) in zip(axes.flat, models_to_compare):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Died', 'Survived'],
                yticklabels=['Died', 'Survived'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{name}\nConfusion Matrix')

plt.tight_layout()
plt.show()
```

### Part 5: Cross-Validation and Regularization (15 minutes)

```python
# Cross-validation comparison
from sklearn.model_selection import cross_val_score

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3)
}

cv_results = []
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled if 'Logistic' in name else X_train,
                             y_train, cv=5, scoring='accuracy')
    cv_results.append({
        'Model': name,
        'Mean CV Accuracy': scores.mean(),
        'Std CV Accuracy': scores.std()
    })

cv_df = pd.DataFrame(cv_results)
print("Cross-Validation Results:")
print(cv_df.to_string(index=False))
```

**Task 5.1**: Regularization in Logistic Regression

```python
# Test different regularization strengths
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
cv_scores = []

for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append({
        'C': C,
        'Mean Accuracy': scores.mean(),
        'Std Accuracy': scores.std()
    })

reg_df = pd.DataFrame(cv_scores)
print("\nRegularization Effect (C = 1/λ):")
print(reg_df.to_string(index=False))

# Plot
plt.figure(figsize=(10, 5))
plt.errorbar(np.log10(reg_df['C']), reg_df['Mean Accuracy'],
             yerr=reg_df['Std Accuracy'], marker='o', capsize=5)
plt.xlabel('log10(C) (higher = less regularization)')
plt.ylabel('CV Accuracy')
plt.title('Effect of Regularization on Cross-Validation Accuracy')
plt.tight_layout()
plt.show()
```

### Challenge Questions

1. **Regression to the Mean**: If a model performs exceptionally well on one random train/test split, what should you expect on different splits? How does this relate to Galton's observations?

2. **Feature Engineering**: What additional features might you create to improve predictions? (Think about: titles in names, cabin locations, ticket classes)

3. **Threshold Selection**: The default threshold for classification is 0.5. How would you choose a different threshold if you wanted to maximize recall (finding all survivors)?

4. **Imbalanced Classes**: What if only 10% of passengers had survived? How would you modify your approach?

5. **Interpretability vs. Accuracy**: Logistic regression is more interpretable than random forest. When would you choose interpretability over higher accuracy?

### Expected Outputs

Students should submit:
1. Exploratory data analysis with visualizations and insights
2. Feature engineering decisions with justification
3. At least three trained models with evaluation metrics
4. ROC curves and confusion matrices comparison
5. Cross-validation results showing model reliability
6. Written analysis of which model they would deploy and why

### Evaluation Rubric

| Criteria | Points |
|----------|--------|
| Data exploration and visualization | 15 |
| Feature engineering quality | 15 |
| Correct model implementation | 20 |
| Proper evaluation methodology | 20 |
| Model comparison and selection | 15 |
| Code quality and interpretation | 15 |
| **Total** | **100** |

---

## Recommended Resources

### Books

**Technical**
- *An Introduction to Statistical Learning* (ISL) by James, Witten, Hastie, Tibshirani - Free online, the essential introduction
- *The Elements of Statistical Learning* (ESL) by Hastie, Tibshirani, Friedman - Free online, more mathematical
- *Pattern Recognition and Machine Learning* by Christopher Bishop - Deep, comprehensive
- *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron - Practical Python focus

**Historical and Popular**
- *The Lady Tasting Tea* by David Salsburg - Stories of statistical pioneers including Galton and Pearson
- *The Theory That Would Not Die* by Sharon McGrayne - History of Bayesian statistics
- *Moneyball* by Michael Lewis - Regression and prediction in baseball
- *Naked Statistics* by Charles Wheelan - Accessible introduction

### Academic Papers

- **Galton, F. (1886)**. "Regression Towards Mediocrity in Hereditary Stature" - The original regression paper
- **Breiman, L. (2001)**. "Random Forests" - Machine Learning, 45(1), 5-32
- **Friedman, J.H. (2001)**. "Greedy Function Approximation: A Gradient Boosting Machine"
- **Chen, T., & Guestrin, C. (2016)**. "XGBoost: A Scalable Tree Boosting System"
- **Hastie, T., et al. (2020)**. "Best Subset Selection is Hard" - On model selection complexity

### Video Lectures

- **StatQuest with Josh Starmer**: Clear explanations of regression, logistic regression, decision trees
- **Stanford CS229 Machine Learning**: Andrew Ng's classic course
- **3Blue1Brown: Gradient Descent** - Beautiful visualization
- **MIT OpenCourseWare: 18.S096 Topics in Mathematics** - Applied statistics

### Online Courses

- **Coursera: Machine Learning by Andrew Ng** - The classic introduction
- **Fast.ai: Practical Machine Learning** - Practical, code-first approach
- **DataCamp: Supervised Learning with scikit-learn** - Hands-on Python
- **Kaggle Learn: Intro to Machine Learning** - Short, practical tutorials

### Tools and Libraries

- **scikit-learn** (https://scikit-learn.org/) - Python machine learning
- **XGBoost** (https://xgboost.readthedocs.io/) - Gradient boosting
- **LightGBM** (https://lightgbm.readthedocs.io/) - Fast gradient boosting
- **CatBoost** (https://catboost.ai/) - Handles categorical features
- **SHAP** (https://shap.readthedocs.io/) - Model interpretability
- **Yellowbrick** (https://www.scikit-yb.org/) - ML visualization

### Datasets for Practice

- **Kaggle Titanic** - Classic binary classification
- **Boston Housing** - Regression (now deprecated due to ethical concerns)
- **California Housing** - Better housing price regression dataset
- **Pima Indians Diabetes** - Medical classification
- **Credit Card Fraud** - Imbalanced classification
- **UCI Machine Learning Repository** - Hundreds of datasets

---

## References

1. Galton, F. (1886). "Regression Towards Mediocrity in Hereditary Stature." *Journal of the Anthropological Institute*, 15, 246-263.

2. Galton, F. (1889). *Natural Inheritance*. London: Macmillan.

3. Pearson, K. (1896). "Mathematical Contributions to the Theory of Evolution. III. Regression, Heredity, and Panmixia." *Philosophical Transactions of the Royal Society of London*, 187, 253-318.

4. Cox, D.R. (1958). "The Regression Analysis of Binary Sequences." *Journal of the Royal Statistical Society: Series B*, 20(2), 215-242.

5. Breiman, L., Friedman, J.H., Olshen, R.A., & Stone, C.J. (1984). *Classification and Regression Trees*. Wadsworth.

6. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.

7. Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189-1232.

8. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD*.

9. Vapnik, V.N. (1995). *The Nature of Statistical Learning Theory*. Springer.

10. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

11. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.

12. Stigler, S.M. (1986). *The History of Statistics: The Measurement of Uncertainty before 1900*. Harvard University Press.

---

*Module 10 explores the fundamental methods of supervised learning—regression and classification—tracing their origins from Galton's pioneering statistical work to the ensemble methods that dominate modern machine learning. Through the story of how "regression" got its name, we learn not just the techniques but the deeper insights about prediction, variability, and the statistical nature of the world.*
