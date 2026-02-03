---
layout: default
title: "Module 3: Statistical Thinking"
---

# Module 3: Statistical Thinking
## "Probabilistic Thinking"

*Research Document for DATA 201 Course Development*

---

# Table of Contents

1. [Introduction](#introduction)
2. [Part I: The Birth of Probability](#part-i-the-birth-of-probability)
3. [Part II: The Great Statistical Ideas](#part-ii-the-great-statistical-ideas)
4. [Part III: Statistical Pitfalls and Paradoxes](#part-iii-statistical-pitfalls-and-paradoxes)
5. [Part IV: The Bayesian Revolution](#part-iv-the-bayesian-revolution)
6. [Part V: Modern Statistical Practice](#part-v-modern-statistical-practice)
7. [DEEP DIVE: The Literary Digest Disaster of 1936](#deep-dive-the-literary-digest-disaster-of-1936)
8. [Lecture Plan and Hands-On Exercise](#lecture-plan-and-hands-on-exercise)
9. [Recommended Resources](#recommended-resources)
10. [References](#references)

---

# Introduction

Statistical thinking is the art of reasoning under uncertainty. This module traces the journey from gambling problems to modern data science, exploring:

- How probability emerged from games of chance
- The central ideas that revolutionized inference
- Famous failures that teach us what NOT to do
- The ongoing debate between Bayesian and frequentist approaches

**Core Question:** How do we make decisions when we can't be certain?

---

# Part I: The Birth of Probability

## The Problem of Points: Pascal and Fermat (1654)

In 1654, a French gambler named Antoine Gombaud (known as the Chevalier de Méré) posed a problem to mathematician Blaise Pascal: If a fair game of chance is interrupted, how should the stakes be divided based on the current score?

### The Correspondence

Pascal wrote to Pierre de Fermat, and their exchange of letters founded probability theory. They solved the "problem of points" by reasoning about all possible future outcomes—the first systematic use of expected value.

**Example Problem:**
Two players need 3 wins each to win the pot. Player A has 2 wins, Player B has 1 win. The game is interrupted. How should they split the pot fairly?

**The Solution:**
Consider all possible ways the game could end:
- A wins next game: A wins (probability 1/2)
- B wins next, then A wins: A wins (probability 1/4)
- B wins two in a row: B wins (probability 1/4)

A should get 3/4 of the pot, B should get 1/4.

### The Data Journey
- **Collection:** Enumeration of all possible outcomes
- **Understanding:** Concept of expected value and fair division
- **Prediction:** Rational decision-making under uncertainty

### Sources
- [Wikipedia - Problem of Points](https://en.wikipedia.org/wiki/Problem_of_points)
- Devlin, K. (2008). *The Unfinished Game: Pascal, Fermat, and the Seventeenth-Century Letter that Made the Modern World*

---

## The Bernoulli Family: A Dynasty of Probabilists

The Bernoulli family of Basel, Switzerland produced at least eight prominent mathematicians across three generations, many contributing to probability and statistics.

### Jacob Bernoulli (1655-1705)
- Published *Ars Conjectandi* (1713) posthumously
- Proved the **Law of Large Numbers**: As sample size increases, the sample mean approaches the population mean
- First rigorous foundation for statistical inference

### Daniel Bernoulli (1700-1782)
- Introduced the **St. Petersburg Paradox**
- Developed **expected utility theory**—people don't just maximize expected value; they weight outcomes by personal utility

### The St. Petersburg Paradox

A casino offers a game: Flip a coin until it lands heads. If heads appears on flip n, you win $2^n.

Expected value: $1 + $1 + $1 + ... = ∞

Would you pay $1,000,000 to play? Most wouldn't—revealing that humans don't simply maximize expected value.

---

## Pierre-Simon Laplace: The Probability of Causes (1749-1827)

Laplace systematized probability in his monumental *Théorie analytique des probabilités* (1812). He developed:

- **Bayes' theorem** in its modern form
- The **Central Limit Theorem** (refined version)
- **Laplace's rule of succession**: If an event has occurred n times consecutively, the probability it occurs again is (n+1)/(n+2)

### Laplace's Demon

Laplace imagined an intellect that knew the position and momentum of every particle in the universe—such a being could predict all future states. This "demon" represents the deterministic worldview that probability theory would eventually challenge.

> "Probability is nothing but common sense reduced to calculation." — Laplace

---

# Part II: The Great Statistical Ideas

## The Central Limit Theorem: Order from Chaos

The CLT states that the sum (or average) of many independent random variables approaches a normal distribution, regardless of the original distribution.

### History
- **Abraham de Moivre (1733)**: First special case (binomial)
- **Laplace (1812)**: More general formulation
- **Lyapunov (1901)**: Rigorous proof under general conditions

### Why It Matters

The CLT explains why the normal distribution appears everywhere:
- Heights of people (sum of genetic and environmental factors)
- Measurement errors (sum of many small errors)
- Test scores (sum of many question performances)

### The Data Journey
- **Collection:** Any measurements from complex processes
- **Understanding:** Sums of random effects → normal distribution
- **Prediction:** Enables confidence intervals, hypothesis testing

### Hands-On Demo
```python
import numpy as np
import matplotlib.pyplot as plt

# Sample means from ANY distribution approach normal
# Try with: exponential, uniform, Bernoulli, etc.

np.random.seed(42)
sample_means = []

for _ in range(10000):
    # Sample from exponential distribution
    sample = np.random.exponential(scale=1, size=30)
    sample_means.append(np.mean(sample))

plt.hist(sample_means, bins=50, density=True, alpha=0.7)
plt.title("Distribution of Sample Means (n=30)")
plt.xlabel("Sample Mean")
plt.show()
```

---

## Correlation Is Not Causation

### The Classic Example: Ice Cream and Drowning

Ice cream sales and drowning deaths are strongly correlated. Does ice cream cause drowning?

No—both are caused by a **confounding variable**: hot weather. More people buy ice cream AND more people swim when it's hot.

### Spurious Correlations

Tyler Vigen's website [tylervigen.com/spurious-correlations](https://tylervigen.com/spurious-correlations) documents absurd correlations:
- US spending on science correlates with suicides by hanging
- Nicolas Cage films correlate with pool drownings
- Divorce rate in Maine correlates with margarine consumption

### When Correlation DOES Suggest Causation

Bradford Hill's criteria (1965) for inferring causation:
1. **Strength**: Strong associations more likely causal
2. **Consistency**: Reproducible across studies
3. **Specificity**: Specific exposure → specific outcome
4. **Temporality**: Cause precedes effect
5. **Biological gradient**: Dose-response relationship
6. **Plausibility**: Mechanism makes sense
7. **Coherence**: Consistent with known biology
8. **Experiment**: Manipulation produces effect
9. **Analogy**: Similar causes produce similar effects

### Sources
- [Tyler Vigen - Spurious Correlations](https://tylervigen.com/spurious-correlations)
- Hill, A. B. (1965). The environment and disease: association or causation? *Proceedings of the Royal Society of Medicine*

---

## Simpson's Paradox: When Aggregation Misleads

### The Berkeley Admissions Case (1973)

UC Berkeley's graduate admissions appeared to discriminate against women:
- Men: 44% admitted
- Women: 35% admitted

But examining individual departments told a different story—in most departments, women had HIGHER admission rates than men!

### The Explanation

Women applied disproportionately to more competitive departments (humanities, arts). Men applied more to less competitive departments (engineering, sciences). The aggregated data hid this pattern.

### The Kidney Stone Treatment Paradox

Two treatments for kidney stones:
- Treatment A: 78% success overall
- Treatment B: 83% success overall

Treatment B seems better! But for small stones, A is better. For large stones, A is also better. How?

Treatment A was used more often on large (harder to treat) stones, pulling down its overall average.

### Lesson

**Always examine subgroups.** Aggregate statistics can completely reverse when disaggregated.

### Sources
- Bickel, P. J., Hammel, E. A., & O'Connell, J. W. (1975). Sex Bias in Graduate Admissions. *Science*, 187(4175), 398-404.
- [Wikipedia - Simpson's Paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox)

---

# Part III: Statistical Pitfalls and Paradoxes

## The Challenger Disaster (1986): A Statistical Tragedy

On January 28, 1986, the Space Shuttle Challenger exploded 73 seconds after launch, killing all seven crew members. The cause: O-ring failure in cold weather.

### The Night Before

Engineers at Morton Thiokol warned that the O-rings might fail at the predicted launch temperature of 36°F. They had data showing O-ring damage at lower temperatures.

### The Statistical Failure

When presenting data, engineers showed only flights WITH O-ring damage, not the full dataset. The critical scatter plot—showing temperature vs. O-ring incidents for ALL flights—was never created.

### What the Full Data Showed

When ALL data points are plotted (including flights without damage), the relationship is clear: colder temperatures strongly predict O-ring problems. The coldest previous launch (53°F) had significant damage. Launching at 36°F was far outside the safe range.

### The Lesson

**Show ALL the data.** Selection bias—even unintentional—can lead to fatal decisions.

### Sources
- Tufte, E. R. (1997). *Visual Explanations*. Chapter on Challenger.
- Presidential Commission on the Space Shuttle Challenger Accident (1986). Rogers Commission Report.

---

## The Hot Hand Fallacy (Or Is It?)

### The Original Study (1985)

Gilovich, Vallone, and Tversky analyzed basketball shooting data and concluded that the "hot hand" (a player being more likely to make a shot after making previous shots) was a cognitive illusion. Fans and players believed in streaks that didn't exist statistically.

### The Reversal (2014-2018)

Researchers Miller and Sanjurjo discovered a subtle but critical flaw: the original analysis had a **selection bias**. When you condition on having just made a shot, you're more likely to be in a sequence that looks "cold" by chance.

Correcting for this bias, there IS evidence of a hot hand—just smaller than intuition suggests.

### The Lesson

Even experts can miss subtle statistical traps. Always question the sampling procedure.

### Sources
- Gilovich, T., Vallone, R., & Tversky, A. (1985). The hot hand in basketball: On the misperception of random sequences. *Cognitive Psychology*.
- Miller, J. B., & Sanjurjo, A. (2018). Surprised by the hot hand fallacy? A truth in the law of small numbers. *Econometrica*.

---

# Part IV: The Bayesian Revolution

## Thomas Bayes and the Essay (1763)

Thomas Bayes was an English Presbyterian minister whose *Essay towards solving a Problem in the Doctrine of Chances* was published posthumously in 1763 by his friend Richard Price.

### Bayes' Theorem

$$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$$

- **P(H)**: Prior probability of hypothesis
- **P(E|H)**: Likelihood of evidence given hypothesis
- **P(H|E)**: Posterior probability after seeing evidence

### The Bayesian vs. Frequentist Debate

**Frequentist view:**
- Probability is long-run frequency
- Parameters are fixed (unknown) constants
- Inference uses sampling distributions, p-values, confidence intervals

**Bayesian view:**
- Probability is degree of belief
- Parameters have probability distributions
- Inference updates prior beliefs with data

### The Revival of Bayesian Methods

For most of the 20th century, frequentist methods dominated. The Bayesian revival came with:
- **Computational advances**: MCMC methods (1990s) made complex Bayesian models tractable
- **Dennis Lindley** and **Bruno de Finetti**: Philosophical foundations
- **Practical success**: Spam filtering, medical diagnosis, machine learning

### Sources
- McGrayne, S. B. (2011). *The Theory That Would Not Die: How Bayes' Rule Cracked the Enigma Code, Hunted Down Russian Submarines, and Emerged Triumphant from Two Centuries of Controversy*

---

## James Lind and the Scurvy Trial (1747): The First Clinical Trial

### The Problem

On long sea voyages, sailors developed scurvy—bleeding gums, weakness, and death. On some voyages, more than half the crew died.

### Lind's Experiment

Ship's surgeon James Lind conducted what's considered the first controlled clinical trial. He took 12 sailors with scurvy and divided them into six groups of two, each receiving a different treatment:
1. Cider
2. Elixir vitriol (sulfuric acid)
3. Vinegar
4. Seawater
5. Oranges and lemons
6. Nutmeg and barley water

### The Result

The two sailors given citrus fruit recovered almost immediately. The others showed no improvement.

### Why It Took 50 Years

Despite clear evidence, the British Navy didn't mandate citrus until 1795. Reasons:
- No understanding of WHY it worked (vitamin C unknown until 1932)
- Cost and logistics of supplying citrus
- Competing theories about scurvy's cause

### The Data Journey
- **Collection:** Controlled experiment with comparison groups
- **Understanding:** Citrus cures scurvy (mechanism unknown)
- **Prediction:** Prevented countless deaths once implemented

### Sources
- [James Lind Library](https://www.jameslindlibrary.org/)
- Bartholomew, M. (2002). James Lind's Treatise of the Scurvy (1753). *Postgraduate Medical Journal*.

---

# Part V: Modern Statistical Practice

## A/B Testing: Statistics at Scale

### The Netflix Optimization Machine

Netflix runs hundreds of A/B tests simultaneously, optimizing everything from thumbnail images to recommendation algorithms. Each test involves millions of users.

**Famous example:** Netflix discovered that images of faces showing complex emotions drive 30% more clicks than neutral expressions.

### A/B Testing Best Practices

1. **Randomization**: Users randomly assigned to variants
2. **Sample size calculation**: Ensure sufficient power
3. **Pre-registration**: Specify analysis before seeing data
4. **Multiple testing correction**: Adjust for running many tests
5. **Practical significance**: Statistical significance ≠ meaningful effect

### The Replication Crisis

In 2015, the Open Science Collaboration attempted to replicate 100 psychology studies. Only 36% replicated successfully.

Causes:
- Publication bias (only significant results published)
- P-hacking (trying analyses until p < 0.05)
- Underpowered studies
- Flexible analysis decisions

### Solutions
- Pre-registration of studies
- Larger sample sizes
- Reporting all results (not just significant ones)
- Bayesian methods that quantify evidence for null hypothesis

### Sources
- Open Science Collaboration. (2015). Estimating the reproducibility of psychological science. *Science*.
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*

---

# DEEP DIVE: The Literary Digest Disaster of 1936
**The Most Famous Polling Failure in History**

## The Story

In 1936, *The Literary Digest*—a prestigious American magazine—conducted the largest poll in history to predict the presidential election between Franklin D. Roosevelt (Democrat) and Alf Landon (Republican).

### The Method

The Digest mailed **10 million questionnaires** to Americans, using lists from:
- Telephone directories
- Automobile registrations
- Magazine subscription lists (including their own)

They received over **2.3 million responses**—an unprecedented sample size.

### The Prediction

Based on these massive returns, the Digest confidently predicted:
- **Landon: 57%**
- **Roosevelt: 43%**

### The Actual Result

Roosevelt won in one of the largest landslides in American history:
- **Roosevelt: 61%**
- **Landon: 37%**

The Digest was off by **19 percentage points**. The magazine, which had correctly predicted the previous five elections, never recovered from the humiliation and ceased publication within two years.

## What Went Wrong

### 1. Selection Bias: The Sample Wasn't Representative

In 1936, during the Great Depression:
- Only wealthy Americans owned telephones
- Only wealthy Americans owned automobiles
- Magazine subscribers skewed affluent

The Digest's sample systematically overrepresented wealthy Republicans and excluded poorer Democratic voters.

### 2. Non-Response Bias

Of 10 million surveys mailed, only 2.3 million responded (23%). Those who felt strongly about the election—disproportionately anti-Roosevelt—were more likely to return surveys.

### 3. The Fallacy of Large Numbers

**A biased sample doesn't become unbiased by making it larger.**

The Digest's massive sample size gave false confidence. In the words of statistician George Gallup:

> "A sampling procedure that has built-in biases will not improve its accuracy no matter how big it gets."

## The Counterexample: George Gallup's Success

Meanwhile, a young statistician named George Gallup used **scientific sampling** with only **50,000 respondents**:

- Random selection to ensure representativeness
- Quota sampling to match demographic proportions
- Face-to-face interviews instead of mail surveys

Gallup predicted Roosevelt's victory—and, brilliantly, also predicted the Literary Digest's error!

### Gallup's Method

1. Define the target population
2. Use demographic quotas (age, gender, region, income)
3. Random selection within quotas
4. Weight responses to match population

## Why This Matters Today

### Modern Echoes

**2016 US Presidential Election:** Most polls predicted Hillary Clinton would win. Donald Trump won. What went wrong?

- Education polarization: Less-educated white voters (Trump supporters) underrepresented
- Differential turnout: Polls measured preferences, not actual votes
- Late deciders: Many undecided voters broke for Trump

**2020 Polling Errors:** Polls again underestimated Trump support, suggesting systematic issues with reaching certain voters.

### The Fundamental Lessons

1. **Sample quality > sample size**: A small random sample beats a large biased one
2. **Non-response matters**: Who doesn't answer tells you something
3. **The population changes**: Who the "likely voters" are shifts
4. **Beware of confidence**: Large numbers create false certainty

## The Data Journey

- **Collection:** Mail surveys to 10 million Americans (from biased lists)
- **Understanding:** Failed to recognize sampling bias
- **Prediction:** Catastrophically wrong—not because of analysis, but because of data collection

---

# Lecture Plan and Hands-On Exercise

## Lecture Plan: "Why Sampling Matters" (75-90 minutes)

### Part 1: The Setup (15 min)

**Opening Question:** "If you want to know what Americans think, how many do you need to ask?"

Show students:
- Literary Digest's confidence (2.3 million responses!)
- The prediction: Landon wins easily
- The actual result: Roosevelt landslide

"What went wrong?"

### Part 2: The Story (25 min)

**Historical Context:**
- 1936: Great Depression, New Deal controversy
- The Literary Digest's previous success
- Why telephone/automobile owners skewed Republican

**The Two Errors:**
1. Selection bias (demonstrate with classroom example)
2. Non-response bias

**George Gallup's Alternative:**
- Scientific sampling principles
- His bold prediction about the Digest's error

### Part 3: The Statistics (20 min)

**Sampling Theory:**
- Population vs. sample
- Random sampling
- Sampling distribution
- Margin of error

**Key Formula:**
$$\text{Margin of Error} \approx \frac{1}{\sqrt{n}}$$

But this assumes random sampling! With bias:
$$\text{True Error} = \text{Sampling Error} + \text{Bias}$$

Bias doesn't shrink with sample size.

### Part 4: Modern Applications (15 min)

**Discussion:**
- 2016/2020 polling errors
- Online survey challenges
- How Netflix/Google/Facebook deal with sampling

**The Replication Crisis:**
- Why do published findings often fail to replicate?
- Connection to sampling, selection bias, and incentives

---

## Hands-On Exercise: "The Biased Sample"

### Objective
Experience how bias corrupts inference, even with large samples.

### Duration
1.5-2 hours

### Setup

**Dataset:** Simulated population of 10,000 "voters" with known preferences
- Each voter has: Age, Income, Education, Vote preference (A or B)
- True population split: 55% A, 45% B

### Task 1: Random Sampling (30 min)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create population
np.random.seed(42)
n_pop = 10000

population = pd.DataFrame({
    'age': np.random.normal(45, 15, n_pop).clip(18, 90),
    'income': np.random.lognormal(10.5, 0.8, n_pop),
    'education_years': np.random.normal(14, 3, n_pop).clip(8, 22)
})

# True preference: affected by demographics
# Younger, lower income → prefer A
prob_A = 0.4 + 0.2 * (population['age'] < 45) - 0.15 * (population['income'] > 60000)
prob_A = prob_A.clip(0.1, 0.9)
population['vote'] = np.random.binomial(1, prob_A)

# True population proportion
true_prop_A = population['vote'].mean()
print(f"True proportion voting A: {true_prop_A:.3f}")

# Random sample of size 1000
random_sample = population.sample(1000)
est_random = random_sample['vote'].mean()
print(f"Random sample estimate: {est_random:.3f}")
```

**Questions:**
1. What's your estimate from the random sample?
2. Run it 100 times. What's the distribution of estimates?
3. What's the standard error?

### Task 2: Biased Sampling (30 min)

```python
# "Literary Digest" style sampling: oversample wealthy
# Only people with income > $50,000 can receive survey
wealthy_only = population[population['income'] > 50000]
wealthy_sample = wealthy_only.sample(min(2000, len(wealthy_only)))
est_wealthy = wealthy_sample['vote'].mean()
print(f"Wealthy-only sample estimate: {est_wealthy:.3f}")

# Add non-response bias: only 25% respond,
# and strong A supporters more likely to respond
response_prob = 0.25 + 0.15 * (wealthy_sample['vote'] == 1)
responders = wealthy_sample[np.random.random(len(wealthy_sample)) < response_prob]
est_biased = responders['vote'].mean()
print(f"Wealthy + non-response bias estimate: {est_biased:.3f}")
```

**Questions:**
1. How does the biased estimate compare to truth?
2. Is a larger biased sample more accurate?
3. How would you correct for this bias if you knew the population demographics?

### Task 3: Post-Stratification (30 min)

**Technique:** If your sample is biased but you know population demographics, you can reweight.

```python
# Post-stratification: weight sample to match population
# Assume we know population proportion in each income bracket

pop_income_cats = pd.cut(population['income'],
                          bins=[0, 30000, 60000, 100000, np.inf],
                          labels=['low', 'medium', 'high', 'very_high'])
pop_weights = pop_income_cats.value_counts(normalize=True)

sample_income_cats = pd.cut(wealthy_sample['income'],
                             bins=[0, 30000, 60000, 100000, np.inf],
                             labels=['low', 'medium', 'high', 'very_high'])
sample_weights = sample_income_cats.value_counts(normalize=True)

# Calculate weights for each observation
# ... (implement post-stratification weighting)

# Compare weighted vs unweighted estimates
```

### Task 4: Reflection (20 min)

Write responses to:
1. Why didn't the Literary Digest's huge sample help?
2. In what modern contexts might we face similar biases?
3. What's the minimum information you need to correct for sampling bias?

### Evaluation Rubric

| Criterion | Excellent | Good | Needs Work |
|-----------|-----------|------|------------|
| **Coding** | Correct implementation, clear code | Minor errors | Significant errors |
| **Analysis** | Quantifies bias accurately | Identifies bias | Misunderstands bias |
| **Reflection** | Deep insight about real-world applications | Good connections | Surface-level |

---

# Recommended Resources

## Books

### Accessible
- **Salsburg, D.** *The Lady Tasting Tea* (2001) - Stories of statistical pioneers
- **Wheelan, C.** *Naked Statistics* (2013) - Fun introduction
- **Ellenberg, J.** *How Not to Be Wrong* (2014) - Mathematical thinking
- **McGrayne, S. B.** *The Theory That Would Not Die* (2011) - Bayesian history

### Technical
- **Freedman, D., Pisani, R., & Purves, R.** *Statistics* (4th ed.) - Classic textbook
- **Gelman, A. & Hill, J.** *Data Analysis Using Regression and Multilevel/Hierarchical Models* - Modern applied stats
- **McElreath, R.** *Statistical Rethinking* - Bayesian approach

### History
- **Stigler, S. M.** *The History of Statistics* (1986) - Definitive academic history
- **Hacking, I.** *The Emergence of Probability* (1975) - Philosophical history

## Online Resources

### Courses
- **Coursera: Statistics with R** (Duke) - R-based intro
- **edX: Statistics and R** (Harvard) - Part of Data Science program
- **Khan Academy: Statistics** - Free fundamentals

### Websites
- **Seeing Theory:** https://seeing-theory.brown.edu/ - Visual probability
- **STAT 110 (Harvard):** https://projects.iq.harvard.edu/stat110 - Joe Blitzstein's probability course
- **Spurious Correlations:** https://tylervigen.com/spurious-correlations

### Videos
- **StatQuest (YouTube):** Josh Starmer's excellent explanations
- **3Blue1Brown:** Bayesian probability visualization
- **Numberphile:** Various statistics topics

## Datasets

- **Gallup Poll Historical Data:** https://www.gallup.com/
- **GSS (General Social Survey):** https://gss.norc.org/
- **NHANES (Health survey):** https://www.cdc.gov/nchs/nhanes/

---

# References

## Historical
- Bernoulli, J. (1713). *Ars Conjectandi*.
- Laplace, P. S. (1812). *Théorie analytique des probabilités*.
- Bayes, T. (1763). An Essay towards solving a Problem in the Doctrine of Chances.

## Classical Papers
- Bickel, P. J., Hammel, E. A., & O'Connell, J. W. (1975). Sex Bias in Graduate Admissions: Data from Berkeley. *Science*, 187(4175), 398-404.
- Hill, A. B. (1965). The Environment and Disease: Association or Causation? *Proceedings of the Royal Society of Medicine*, 58(5), 295-300.

## Modern
- Open Science Collaboration. (2015). Estimating the reproducibility of psychological science. *Science*, 349(6251), aac4716.
- Miller, J. B., & Sanjurjo, A. (2018). Surprised by the hot hand fallacy? *Econometrica*, 86(6), 2019-2047.

## Books
- Salsburg, D. (2001). *The Lady Tasting Tea*. Holt Paperbacks.
- McGrayne, S. B. (2011). *The Theory That Would Not Die*. Yale University Press.
- Tufte, E. R. (1997). *Visual Explanations*. Graphics Press.

---

*Document compiled for SCDS DATA 201: Introduction to Data Science I*
*Module 3: Statistical Thinking*
*"Probabilistic Thinking"*
