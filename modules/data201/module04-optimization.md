---
layout: default
title: "Module 4: Optimization and Introduction to Machine Learning"
---

# Module 4: Optimization and Introduction to Machine Learning
## "Optimization and Model Fitting"

*Research Document for DATA 201 Course Development*

---

# Table of Contents

1. [Introduction](#introduction)
2. [Part I: The History of Optimization](#part-i-the-history-of-optimization)
3. [Part II: Operations Research - Math Goes to War](#part-ii-operations-research---math-goes-to-war)
4. [Part III: The Dawn of Machine Learning](#part-iii-the-dawn-of-machine-learning)
5. [Part IV: Modern Optimization and Deep Learning](#part-iv-modern-optimization-and-deep-learning)
6. [DEEP DIVE: George Dantzig and the Homework Problems](#deep-dive-george-dantzig-and-the-homework-problems)
7. [Lecture Plan and Hands-On Exercise](#lecture-plan-and-hands-on-exercise)
8. [Recommended Resources](#recommended-resources)
9. [References](#references)

---

# Introduction

Optimization is the mathematical backbone of machine learning: finding the best parameters, the best decisions, the best predictions. This module traces:

- How mathematicians developed tools to find optimal solutions
- How World War II transformed optimization into a practical science
- How fitting models to data became "machine learning"

**Core Question:** Given constraints and objectives, how do we find the best solution?

---

# Part I: The History of Optimization

## Fermat and the Principle of Least Time (1662)

Pierre de Fermat proposed that light travels along the path that minimizes travel time. This "principle of least time" was among the first optimization principles in physics.

**Example:** Light bending at water's surface follows exactly the path that minimizes total travel time—even though the light "doesn't know" it's optimizing.

This principle foreshadowed variational calculus and optimization throughout physics.

---

## Newton's Method (1669)

Isaac Newton developed an iterative method for finding roots of equations:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

To find minima, apply this to the derivative f'(x) = 0.

### The Idea
Start somewhere, use local information (derivative) to improve, repeat until convergence.

### Why It Matters
Newton's method anticipates gradient-based optimization that dominates modern machine learning.

---

## Lagrange Multipliers (1788)

Joseph-Louis Lagrange developed a method for optimizing functions subject to constraints.

**Problem:** Maximize f(x,y) subject to g(x,y) = 0

**Solution:** Find where ∇f = λ∇g (gradients are parallel)

### Example
Maximize area of rectangle with fixed perimeter:
- Objective: A = xy
- Constraint: 2x + 2y = P

The answer: a square. Lagrange's method proves it elegantly.

### Applications
- Economics: Maximize utility subject to budget
- Physics: Mechanics with constraints
- Machine learning: Regularization, support vector machines

---

## Cauchy and Gradient Descent (1847)

Augustin-Louis Cauchy proposed the method of steepest descent:

$$x_{n+1} = x_n - \alpha \nabla f(x_n)$$

Move in the direction opposite to the gradient (steepest downhill direction).

### The Algorithm
1. Start at some point
2. Compute the gradient (direction of steepest increase)
3. Take a step in the opposite direction
4. Repeat until converged

### Why This Is Everything

Gradient descent (and its variants) powers virtually all modern machine learning:
- Neural network training
- Logistic regression
- Recommender systems
- Large language models

The 1847 idea scales to billions of parameters.

---

# Part II: Operations Research - Math Goes to War

## The Birth of Operations Research (1937-1945)

World War II created urgent optimization problems:
- How to allocate scarce resources?
- How to route convoys to avoid submarines?
- How to schedule bombing raids?

### Radar and the Bawdsey Manor Scientists

In 1937, British scientists at Bawdsey Manor studied how to optimally deploy the new radar technology. They called their work "operational research"—researching operations.

### Key WWII Problems Solved

**Convoy Routing:** How large should convoys be? Analysis showed larger convoys were safer—contradicting intuition that big convoys were bigger targets.

**Bombing Accuracy:** What altitude and formation maximized effectiveness?

**Inventory Management:** How much spare parts to stock?

**Anti-submarine Warfare:** Where to search for U-boats? The operations research team found optimal patrol patterns.

---

## Linear Programming and the Simplex Algorithm

### Leonid Kantorovich (1939)

Soviet mathematician Leonid Kantorovich developed linear programming to optimize plywood production. His work was suppressed for ideological reasons—optimization implied central planning could work.

He received the Nobel Prize in Economics in 1975.

### George Dantzig and the Simplex Method (1947)

George Dantzig, working for the US Air Force, independently developed linear programming and invented the **simplex algorithm** for solving it.

**Linear Programming:**
- Maximize a linear objective: c^T x
- Subject to linear constraints: Ax ≤ b, x ≥ 0

**The Simplex Algorithm:**
Move along edges of the feasible region, improving at each step, until reaching the optimum.

### Why Linear Programming Matters

Applications:
- Airline scheduling
- Supply chain optimization
- Portfolio allocation
- Manufacturing planning

The simplex algorithm is one of the most practically important algorithms ever invented.

### Sources
- Dantzig, G. B. (1963). *Linear Programming and Extensions*
- [Wikipedia - Simplex Algorithm](https://en.wikipedia.org/wiki/Simplex_algorithm)

---

# Part III: The Dawn of Machine Learning

## The Perceptron: Frank Rosenblatt (1958)

Frank Rosenblatt, a psychologist at Cornell, created the **Perceptron**—the first machine that could learn from data.

### The Machine

The perceptron was physical hardware with:
- 400 photocells (20x20 grid) as input
- Potentiometers (variable resistors) as weights
- Motor-driven weight updates during learning

### The Algorithm

$$\hat{y} = \text{sign}(w \cdot x + b)$$

**Learning rule:** If prediction is wrong, adjust weights toward the correct answer.

### The Hype

The *New York Times* (1958) reported:
> "The Navy revealed the embryo of an electronic computer today that it expects will be able to walk, talk, see, write, reproduce itself and be conscious of its existence."

### The Crash: Minsky and Papert (1969)

Marvin Minsky and Seymour Papert published *Perceptrons*, proving that single-layer perceptrons couldn't learn certain functions (like XOR).

The book triggered the first "AI Winter"—a collapse in neural network research funding that lasted until the 1980s.

### The Lesson

The perceptron *was* limited, but multi-layer networks (which Minsky and Papert acknowledged could solve XOR) were dismissed as untrainable. This proved wrong—but it took 20 years to demonstrate.

### Sources
- Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain. *Psychological Review*.
- Minsky, M. & Papert, S. (1969). *Perceptrons*.

---

## Backpropagation: The Algorithm That Changed Everything

### The Key Problem

Multi-layer neural networks could theoretically learn anything, but how do you train them? How do you know which weights to adjust when the error occurs at the output?

### The Solution: Credit Assignment via Calculus

**Backpropagation** uses the chain rule to efficiently compute how each weight contributes to the final error:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### History

The algorithm was discovered multiple times:
- **1960s:** Henry J. Kelley, Arthur E. Bryson (control theory)
- **1970:** Seppo Linnainmaa (automatic differentiation)
- **1974:** Paul Werbos (PhD thesis, largely ignored)
- **1986:** Rumelhart, Hinton, and Williams - Nature paper that sparked the neural network renaissance

### The 1986 Paper

"Learning representations by back-propagating errors" by Rumelhart, Hinton, and Williams demonstrated that backpropagation could train multi-layer networks to learn useful representations.

This reignited neural network research and eventually led to deep learning.

### Sources
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.

---

# Part IV: Modern Optimization and Deep Learning

## The Netflix Prize (2006-2009)

In 2006, Netflix offered $1 million to anyone who could improve their recommendation algorithm by 10%.

### The Challenge

Predict how users would rate movies they hadn't seen, based on their previous ratings and the ratings of similar users.

**Data:** 100 million ratings from 480,000 users on 17,770 movies.

### The Competition

Over 40,000 teams from 186 countries competed. The competition:
- Popularized matrix factorization methods
- Demonstrated ensemble methods (combining models)
- Showed that marginal improvements require exponentially more effort

### The Winner

The winning team "BellKor's Pragmatic Chaos" combined 107 different models. They achieved 10.06% improvement—just barely qualifying for the prize.

**Twist:** A second team submitted an identical score just 20 minutes later. Netflix awarded based on submission time.

### Why Netflix Never Used It

Netflix never fully implemented the winning solution. By 2009:
- Streaming had become more important than DVD ratings
- The complex ensemble was too expensive to maintain
- The 10% improvement wasn't worth the engineering cost

### Lessons
- Competitions drive innovation but may not solve business problems
- Ensemble methods are powerful but complex
- The gap between research and production is real

### Sources
- [Netflix Prize on Wikipedia](https://en.wikipedia.org/wiki/Netflix_Prize)
- Bennett, J., & Lanning, S. (2007). The Netflix Prize. *KDD Cup*.

---

## AlphaGo: Reinforcement Learning Triumphs (2016)

In March 2016, DeepMind's AlphaGo defeated world champion Lee Sedol at Go, 4-1.

### Why Go Was Hard

- **Complexity:** More possible Go positions than atoms in the universe
- **Intuition:** Top players couldn't explain their moves—"it just feels right"
- **History:** Chess was solved in 1997 (Deep Blue); Go resisted for 19 more years

### The Technology

AlphaGo combined:
1. **Deep neural networks** to evaluate positions
2. **Monte Carlo tree search** to look ahead
3. **Reinforcement learning** to improve by playing itself

### Move 37

In Game 2, AlphaGo played Move 37—a move no human would make, initially dismissed as a mistake. It turned out to be brilliant, demonstrating that the AI had discovered strategies beyond human knowledge.

### The Data Journey
- **Collection:** 30 million positions from human games
- **Understanding:** Neural networks learned position evaluation
- **Prediction:** Self-play generated superhuman strategies

### Legacy

AlphaGo inspired:
- AlphaFold (protein structure prediction)
- AlphaCode (programming)
- Broader application of reinforcement learning

### Sources
- Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*.
- DeepMind. "AlphaGo" documentary (2017).

---

# DEEP DIVE: George Dantzig and the Homework Problems
**The Legend That Launched a Thousand Inspirational Emails**

## The Story

In 1939, George Dantzig arrived late to a statistics class at UC Berkeley taught by Jerzy Neyman. On the blackboard were two problems. Assuming they were homework, Dantzig copied them down.

A few days later, he apologized to Neyman for how long the "homework" had taken and handed in his solutions.

Six weeks later, Neyman knocked on Dantzig's door, ecstatic. The "homework problems" weren't homework at all—they were two famous **unsolved problems in statistics**.

Dantzig had solved them simply because he didn't know they were supposed to be hard.

## The True Story

This actually happened. Dantzig confirmed it in interviews and wrote about it in his 1986 book.

### The Problems

1. **Proof of a theorem** about the limiting distribution of the ratio of two random variables
2. **Construction of an optimal statistical test** (related to the Neyman-Pearson lemma)

Both were open questions that Neyman had mentioned in passing. Dantzig, arriving late, missed that context.

### Dantzig's Own Words

> "A year later, when I began to worry about a thesis topic, Neyman just shrugged and told me to wrap the two problems in a binder and he would accept them as my thesis."

## The Myth Machine

This true story became the basis for:

### Urban Legends
The story spread, mutated, and was often attributed to other mathematicians (Einstein, unnamed students). Details changed—sometimes it was one problem, sometimes calculus homework.

### Inspirational Books
The story appeared in:
- *Chicken Soup for the Soul*
- Motivational speeches
- Self-help books about the power of not knowing your limitations

### The Movie *Good Will Hunting* (1997)
The film's premise—a janitor solves impossible math problems—was inspired by the Dantzig legend (and a similar story about Ramanujan).

## The Real Lesson

The story is usually told as: "If you don't know something is impossible, you might achieve it."

But Dantzig himself had a more nuanced take:

### 1. Preparation Matters
Dantzig wasn't a random student. He was exceptionally well-prepared, having already studied probability theory extensively.

### 2. The Problems Were Solvable
These weren't problems that had defeated generations of mathematicians. They were open questions that someone of Dantzig's training could potentially solve.

### 3. Mindset Is Real (But Not Magic)
Not knowing a problem is "hard" does help—you don't give up. But you still need the skills to solve it.

## Dantzig's Greater Contribution

The homework story is charming, but Dantzig's real legacy is **linear programming and the simplex algorithm**.

### Impact
The simplex algorithm is used billions of times daily:
- Airline flight scheduling
- Military logistics
- Manufacturing optimization
- Financial portfolio optimization
- Shipping and logistics

### Recognition
- WWII: Optimized Air Force logistics
- 1975: National Medal of Science
- One of the most important algorithms of the 20th century

## The Data Journey

**The Homework Legend:**
- **Collection:** Copying problems off a blackboard (incomplete information)
- **Understanding:** Treating unsolved problems as routine exercises
- **Prediction:** Solutions that became a PhD thesis

**Linear Programming:**
- **Collection:** Resource allocation constraints
- **Understanding:** Formulating optimization mathematically
- **Prediction:** Optimal decisions for complex systems

---

# Lecture Plan and Hands-On Exercise

## Lecture Plan: "Finding the Best Answer" (75-90 minutes)

### Part 1: The Optimization Framework (15 min)

**Opening Question:** "You're running a lemonade stand. How do you maximize profit?"

Define:
- **Objective function:** What we want to optimize
- **Constraints:** What limits our choices
- **Decision variables:** What we can control

Show how everyday decisions are optimization problems.

### Part 2: The Dantzig Story (15 min)

Tell the homework legend, then reveal:
- The true story
- His greater contribution: linear programming
- The simplex algorithm's impact

**Transition:** "What if our objective isn't linear? What if we have millions of variables?"

### Part 3: Gradient Descent (25 min)

**Intuition:** You're blindfolded on a hill. How do you find the lowest point?

**The Algorithm:**
1. Feel which way is downhill (gradient)
2. Take a step that direction
3. Repeat

**Live Demo:** Visualize gradient descent on a 2D surface.

```python
import numpy as np
import matplotlib.pyplot as plt

# Loss surface
def loss(x, y):
    return x**2 + 3*y**2

# Gradient
def grad(x, y):
    return np.array([2*x, 6*y])

# Gradient descent
path = [(3, 3)]
lr = 0.1
for _ in range(20):
    x, y = path[-1]
    g = grad(x, y)
    new_point = (x - lr*g[0], y - lr*g[1])
    path.append(new_point)

# Visualize
```

### Part 4: From Optimization to Learning (15 min)

**Key Insight:** Machine learning = finding parameters that minimize prediction error

- **Linear regression:** Minimize squared error
- **Neural networks:** Minimize loss via backpropagation
- **Everything else:** Gradient descent variations

### Part 5: Modern Examples (10 min)

- Netflix Prize: What they optimized
- AlphaGo: Reinforcement learning as optimization
- Your phone's autocomplete: Optimized predictions

---

## Hands-On Exercise: "Building a Gradient Descent Optimizer"

### Objective
Implement gradient descent from scratch and watch it learn.

### Duration
2 hours

### Task 1: Manual Gradient Descent (30 min)

Minimize f(x) = x² + 10sin(x) (has multiple local minima!)

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 10*np.sin(x)

def df(x):
    return 2*x + 10*np.cos(x)

def gradient_descent(start, lr, n_steps):
    """
    Implement gradient descent.

    Args:
        start: Initial x value
        lr: Learning rate
        n_steps: Number of iterations

    Returns:
        List of x values visited
    """
    path = [start]
    x = start

    for _ in range(n_steps):
        # YOUR CODE: Update x using gradient descent
        x = x - lr * df(x)
        path.append(x)

    return path

# Experiment with different starting points and learning rates
starts = [-5, 0, 3, 7]
lrs = [0.01, 0.1, 0.5]

# Visualize: Where does each run converge?
x_range = np.linspace(-10, 10, 1000)
plt.figure(figsize=(12, 4))
plt.plot(x_range, f(x_range), 'b-', label='f(x)')

for start in starts:
    path = gradient_descent(start, lr=0.1, n_steps=50)
    plt.plot(path, [f(x) for x in path], 'o-', markersize=3, label=f'start={start}')

plt.legend()
plt.title("Gradient Descent with Different Starting Points")
plt.show()
```

**Questions:**
1. How does the starting point affect the final answer?
2. What happens with a learning rate that's too large? Too small?
3. Did you always find the global minimum?

### Task 2: Linear Regression from Scratch (45 min)

```python
# Generate data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 3 * X + 2 + 0.5 * np.random.randn(100, 1)

def predict(X, w, b):
    return X @ w + b

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def gradient(X, y, w, b):
    """
    Compute gradients of MSE loss with respect to w and b.

    Returns:
        dw: Gradient with respect to w
        db: Gradient with respect to b
    """
    n = len(y)
    y_pred = predict(X, w, b)
    error = y_pred - y

    dw = (2/n) * X.T @ error
    db = (2/n) * np.sum(error)

    return dw, db

# Initialize parameters
w = np.zeros((1, 1))
b = 0.0
lr = 0.1

# Training loop
losses = []
for epoch in range(100):
    # Compute loss
    y_pred = predict(X, w, b)
    loss = mse_loss(y, y_pred)
    losses.append(loss)

    # Compute gradients
    dw, db = gradient(X, y, w, b)

    # Update parameters
    w = w - lr * dw
    b = b - lr * db

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w[0,0]:.4f}, b = {b:.4f}")

print(f"\nFinal: w = {w[0,0]:.4f} (true: 3), b = {b:.4f} (true: 2)")
```

**Questions:**
1. How close did you get to the true parameters?
2. Plot the loss over epochs. What shape is it?
3. Compare with `sklearn.linear_model.LinearRegression`. Same answer?

### Task 3: Visualizing the Loss Landscape (30 min)

```python
# Create a 2D loss surface for linear regression
w_range = np.linspace(0, 6, 100)
b_range = np.linspace(-1, 5, 100)
W, B = np.meshgrid(w_range, b_range)

# Compute loss for each (w, b) combination
Loss = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        y_pred = X * W[i,j] + B[i,j]
        Loss[i,j] = np.mean((y - y_pred)**2)

# Contour plot
plt.figure(figsize=(10, 8))
plt.contour(W, B, Loss, levels=50)
plt.colorbar(label='MSE Loss')
plt.xlabel('w')
plt.ylabel('b')
plt.title('Loss Landscape for Linear Regression')

# Add gradient descent path (from Task 2)
# ... plot the path w, b took during training

plt.show()
```

### Extension: Stochastic Gradient Descent

Modify your code to use mini-batches instead of the full dataset:

```python
batch_size = 16

for epoch in range(100):
    # Shuffle data
    indices = np.random.permutation(len(X))

    for i in range(0, len(X), batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]

        # Compute gradient on mini-batch only
        dw, db = gradient(X_batch, y_batch, w, b)

        # Update
        w = w - lr * dw
        b = b - lr * db
```

**Questions:**
1. How does the loss curve look different with SGD?
2. Why might SGD actually find better solutions sometimes?

---

# Recommended Resources

## Books

### Optimization
- **Boyd, S. & Vandenberghe, L.** *Convex Optimization* (2004) - The standard textbook (free online)
- **Nocedal, J. & Wright, S.** *Numerical Optimization* (2006) - Comprehensive algorithms

### Machine Learning
- **Bishop, C.** *Pattern Recognition and Machine Learning* (2006) - Classic ML textbook
- **Goodfellow, I., Bengio, Y., & Courville, A.** *Deep Learning* (2016) - Deep learning bible (free online)

### History
- **Dantzig, G.** *Linear Programming and Extensions* (1963)
- **Sutton, R. & Barto, A.** *Reinforcement Learning: An Introduction* (2018, free online)

## Online Courses

- **Stanford CS229:** Machine Learning (Andrew Ng)
- **Stanford CS231n:** Convolutional Neural Networks
- **Fast.ai:** Practical Deep Learning

## Videos

- **3Blue1Brown:** Neural networks and backpropagation series
- **Two Minute Papers:** AlphaGo and other AI breakthroughs
- **AlphaGo Documentary** (2017) - Available on YouTube

## Interactive Tools

- **TensorFlow Playground:** https://playground.tensorflow.org/
- **ConvNetJS:** https://cs.stanford.edu/people/karpathy/convnetjs/
- **Distill.pub:** Interactive ML explanations

---

# References

## Historical
- Cauchy, A. (1847). Méthode générale pour la résolution des systèmes d'équations simultanées.
- Dantzig, G. B. (1963). *Linear Programming and Extensions*. Princeton University Press.

## Neural Networks
- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386.
- Minsky, M., & Papert, S. (1969). *Perceptrons*. MIT Press.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.

## Modern ML
- Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484-489.
- Bennett, J., & Lanning, S. (2007). The Netflix Prize. *Proceedings of KDD Cup*.

---

*Document compiled for SCDS DATA 201: Introduction to Data Science I*
*Module 4: Optimization and Introduction to Machine Learning*
*"Optimization and Model Fitting"*
