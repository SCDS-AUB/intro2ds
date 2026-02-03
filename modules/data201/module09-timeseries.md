---
layout: default
title: "Module 9: Time Series Data and Forecasting"
---

# Module 9: Time Series Data and Forecasting

## Introduction

Time is the universal dimension of data. Every stock price, every heartbeat, every weather observation, every website click exists at a particular moment, part of an endless stream flowing from past to future. Time series data—measurements recorded sequentially over time—represents one of the oldest and most consequential forms of data analysis.

From ancient Babylonian astronomers tracking planetary movements across decades to predict celestial events, to modern hedge funds analyzing millisecond-resolution market data, humans have always sought to find patterns in time and glimpse what comes next. This module explores the mathematics, methods, and human stories behind time series analysis—the quest to read the rhythms of the world.

---

## Part 1: The Quest to Predict - From Stars to Stocks

### Ancient Timekeepers

The earliest systematic time series analysis began with astronomy. Babylonian priests in the first millennium BCE kept meticulous records of planetary positions, lunar phases, and celestial events. Their clay tablets, recovered from ancient Mesopotamia, contain centuries of observations and increasingly sophisticated methods for predicting eclipses and planetary conjunctions.

The Greek astronomer Hipparchus (c. 190-120 BCE) discovered the precession of the equinoxes by comparing his observations with records made 150 years earlier—an early example of finding patterns in long time series. His successors, culminating in Ptolemy, built mathematical models to predict planetary positions centuries into the future.

### The Birth of Economic Forecasting

The application of time series methods to human affairs came much later. The first recorded attempt at systematic economic forecasting is often attributed to William Stanley Jevons (1835-1882), who in 1878 proposed that business cycles were linked to sunspot cycles. His hypothesis was wrong (though the search for such connections continues), but his approach—seeking repeating patterns in economic time series—was pioneering.

The great economists of the early 20th century developed the tools we still use:

**Ragnar Frisch** (1895-1973), who coined the terms "econometrics" and "macroeconomics," developed methods for analyzing cyclical fluctuations. He shared the first Nobel Prize in Economics in 1969.

**Jan Tinbergen** (1903-1994), Frisch's Nobel co-laureate, built the first macroeconometric model—a system of equations describing how economic variables evolve and interact over time.

### Box and Jenkins: The ARIMA Revolution

The modern era of time series analysis began with a collaboration between two statisticians:

**George E.P. Box** (1919-2013) was a British statistician who had worked on chemical processes and quality control. He is also famous for the aphorism: "All models are wrong, but some are useful."

**Gwilym Jenkins** (1932-1982) was a Welsh statistician specializing in control systems and time series.

Their 1970 book, *Time Series Analysis: Forecasting and Control*, introduced what became known as the **Box-Jenkins methodology** for building ARIMA (AutoRegressive Integrated Moving Average) models. The approach was revolutionary in its systematic, iterative process:

1. **Identification**: Examine the time series to determine the appropriate model structure
2. **Estimation**: Fit the model parameters using the data
3. **Diagnostic Checking**: Validate the model using residual analysis
4. **Forecasting**: Generate predictions with confidence intervals

Their work became the foundation for statistical time series analysis and remains influential more than 50 years later.

---

## Part 2: Understanding Time Series Structure

### Components of Time Series

Most time series can be decomposed into fundamental components:

**Trend**: Long-term increase or decrease in the data. Global average temperature shows an upward trend; the market share of landline phones shows a downward trend.

**Seasonality**: Regular patterns that repeat at fixed intervals. Retail sales peak before Christmas; electricity demand is higher in summer (for cooling) and winter (for heating); ice cream sales are seasonal.

**Cycles**: Fluctuations that are not of fixed period. Business cycles typically last 5-10 years but vary in length and amplitude.

**Noise**: Random variation that cannot be explained by the other components.

The **classical decomposition** separates these components:

$$Y_t = T_t + S_t + C_t + \epsilon_t \quad \text{(additive)}$$

or

$$Y_t = T_t \times S_t \times C_t \times \epsilon_t \quad \text{(multiplicative)}$$

### Stationarity: The Crucial Assumption

A time series is **stationary** if its statistical properties—mean, variance, autocorrelation—do not change over time. Most time series methods assume stationarity or require transforming non-stationary series to stationary ones.

A random walk (like stock prices on short timescales) is the classic non-stationary process:

$$Y_t = Y_{t-1} + \epsilon_t$$

Each step depends on the previous step, and the variance grows without bound over time. Predicting the next step is easy (it's likely close to the current value), but predicting far into the future is nearly impossible.

**Differencing** is the standard technique for making series stationary:

$$Y'_t = Y_t - Y_{t-1}$$

This is the "I" (Integrated) in ARIMA—we model the differenced series, then integrate to get forecasts for the original series.

### Autocorrelation: The Memory of Time

The **autocorrelation function (ACF)** measures how the current value correlates with past values. A time series with strong autocorrelation has "memory"—knowing the past helps predict the future.

The **partial autocorrelation function (PACF)** measures the correlation between observations separated by k time periods, after removing the effects of intermediate observations.

Together, ACF and PACF plots are the primary diagnostic tools for time series:
- In an AR(p) process, PACF cuts off after lag p; ACF decays gradually
- In an MA(q) process, ACF cuts off after lag q; PACF decays gradually
- In an ARMA process, both decay gradually

---

## Part 3: Models for Time Series

### Autoregressive Models (AR)

An AR(p) model predicts the current value as a linear combination of p past values:

$$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p Y_{t-p} + \epsilon_t$$

Think of it as the time series "remembering" its recent past. An AR(1) model with $\phi_1 = 0.9$ means today's value is strongly influenced by yesterday's, with some random noise.

### Moving Average Models (MA)

An MA(q) model predicts based on past forecast errors:

$$Y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q}$$

This captures the idea that shocks to the system persist for some time before dissipating.

### ARIMA: The Workhorse Model

ARIMA(p, d, q) combines:
- AR(p): Autoregressive component with p lags
- I(d): Integration (differencing) d times for stationarity
- MA(q): Moving average component with q lags

Box and Jenkins showed how to systematically identify appropriate values for p, d, and q by examining the data's properties.

### Seasonal Models: SARIMA

For data with strong seasonal patterns, SARIMA extends ARIMA with seasonal components:

$$\text{SARIMA}(p,d,q)(P,D,Q)_m$$

where m is the seasonal period (12 for monthly data with annual seasonality, 7 for daily data with weekly patterns).

### Exponential Smoothing

Developed independently from Box-Jenkins, exponential smoothing methods form another major family of forecasting techniques:

**Simple Exponential Smoothing**: For data without trend or seasonality
$$\hat{Y}_{t+1} = \alpha Y_t + (1-\alpha) \hat{Y}_t$$

**Holt's Linear Method**: Adds a trend component

**Holt-Winters Method**: Adds seasonal component

The parameter α (0 < α < 1) controls how quickly the forecast responds to recent observations. High α gives more weight to recent data; low α gives more weight to historical patterns.

### State Space Models

Modern approaches reformulate time series as **state space models**:
- The system has an unobserved "state" that evolves over time
- We observe noisy measurements of the state

The **Kalman Filter** (Rudolf Kálmán, 1960) provides the optimal way to estimate the hidden state from noisy observations—essential for everything from GPS navigation to economic nowcasting.

---

## Part 4: Machine Learning for Time Series

### The Neural Network Revolution

Traditional time series methods assume linear relationships and specific error distributions. Machine learning approaches can capture complex nonlinear patterns.

**Recurrent Neural Networks (RNNs)**: Designed specifically for sequential data, RNNs maintain a hidden state that updates with each input. But standard RNNs struggle with long-term dependencies—they "forget" distant past.

**Long Short-Term Memory (LSTM)**: Invented by Sepp Hochreiter and Jürgen Schmidhuber in 1997, LSTMs add "gates" that control what information to remember and forget. They became the standard for sequence modeling.

**Transformers**: The attention mechanism, introduced in the 2017 paper "Attention Is All You Need," revolutionized sequence modeling. Transformers can directly attend to any part of the input sequence, regardless of distance.

### Deep Learning Forecasting Models

**DeepAR** (Amazon, 2017): Uses autoregressive RNNs to produce probabilistic forecasts, learning patterns across many related time series.

**Prophet** (Facebook, 2017): Designed for business forecasting, Prophet handles seasonality, holidays, and missing data with minimal tuning.

**Temporal Fusion Transformers** (Google, 2021): Combines interpretability with state-of-the-art accuracy, using attention to highlight which inputs matter most.

**TimeGPT** and **Lag-Llama** (2023): Foundation models for time series, trained on millions of series and applicable zero-shot to new forecasting problems.

### The Challenge of Benchmarking

An embarrassing secret of ML forecasting: for many problems, simple methods beat complex ones. The M-Competitions (organized by Spyros Makridakis) have repeatedly shown that exponential smoothing and ARIMA often outperform elaborate neural networks, especially on short series.

**M4 Competition (2018)**: Combined statistical and ML methods (hybrids) won, with pure ML methods performing poorly.

**M5 Competition (2020)**: LightGBM and other gradient boosting methods, engineered with careful features, dominated.

The lesson: good features and appropriate problem framing often matter more than model sophistication.

---

## Part 5: Forecasting in the Real World

### The Limitations of Prediction

Not all time series are predictable. **Nassim Nicholas Taleb** has famously criticized overconfident forecasting, arguing that rare "Black Swan" events are fundamentally unpredictable yet have outsized impact.

**Philip Tetlock's** research on expert political judgment found that most expert predictions were barely better than chance—but that specific cognitive practices ("superforecasting") could improve accuracy.

The key insight: forecast uncertainty matters as much as the point forecast. A 50% confidence interval that captures 50% of outcomes is useful; one that captures 10% is dangerously misleading.

### Evaluating Forecasts

**Point Forecast Metrics**:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE) / Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Mean Absolute Scaled Error (MASE)

**Probabilistic Forecast Metrics**:
- Continuous Ranked Probability Score (CRPS)
- Coverage probability of prediction intervals

**Critical Principles**:
- Always use out-of-sample evaluation (test on data not used for training)
- Consider multiple metrics (MAPE can be misleading with zeros or small values)
- Compare against naive baselines (persistence forecast, seasonal naive)

### Forecasting at Scale

Modern tech companies forecast millions of time series:

**Amazon**: Predicts demand for millions of products at thousands of fulfillment centers

**Uber**: Forecasts demand by location and time to position drivers and set prices

**Netflix**: Predicts server capacity needs based on viewing patterns

These applications require:
- Automated model selection (no human can tune millions of models)
- Hierarchical forecasting (category totals should equal sum of item forecasts)
- Probabilistic forecasts (for inventory optimization and decision-making)

---

## Part 6: Applications and Case Studies

### Weather Forecasting

Weather prediction is one of the oldest and most successful applications of time series analysis and numerical modeling. The 3-day forecast today is as accurate as the 1-day forecast was 30 years ago—a remarkable achievement of applied mathematics, physics, and computing.

Key developments:
- **Numerical Weather Prediction (NWP)**: Solving the equations of atmospheric motion on a grid
- **Ensemble Forecasting**: Running multiple simulations with slightly different initial conditions to estimate uncertainty
- **Data Assimilation**: Combining observations with model predictions using Kalman filtering

### Financial Markets

Stock prices on short timescales approximate random walks—past prices provide minimal information about future prices. This is the **Efficient Market Hypothesis**: if patterns existed, traders would exploit them until they disappeared.

Yet patterns do exist in:
- Volatility (large moves predict more large moves—"volatility clustering")
- Cross-asset correlations
- Earnings seasonality and calendar effects

**Quantitative Finance** uses time series for:
- Risk modeling (Value at Risk, Expected Shortfall)
- Options pricing (stochastic volatility models)
- High-frequency trading (autocorrelation on microsecond timescales)

### Epidemiology

Disease surveillance relies heavily on time series analysis:
- Detecting outbreaks (sudden deviations from expected patterns)
- Estimating reproductive number R_t
- Forecasting healthcare capacity needs

The COVID-19 pandemic highlighted both the importance and limitations of epidemiological forecasting.

### Energy Demand

Electricity grids must continuously balance supply and demand. Time series forecasting is essential for:
- Day-ahead scheduling of power plants
- Integration of renewable energy (solar and wind are weather-dependent)
- Demand response programs
- Grid stability

---

## DEEP DIVE: Lewis Fry Richardson and the Dream of Numerical Weather Prediction

### The Impossible Calculation

It is 1916, the height of World War I. In a converted barn in the French countryside, a British ambulance driver sits with a stack of papers, calculating. Outside, the Western Front stretches in both directions—trenches, mud, and the constant thunder of artillery. Inside, Lewis Fry Richardson (1881-1953) is attempting something no one has ever done: to predict the weather using mathematics alone.

Richardson was an unusual man. A Quaker pacifist, he refused to bear arms but volunteered to drive ambulances in one of history's bloodiest conflicts. Trained as a physicist, he had worked at the Meteorological Office and become convinced that weather, governed by the laws of fluid dynamics and thermodynamics, should in principle be predictable by solving the governing equations.

The equations were known. They had been written down by Claude-Louis Navier and George Gabriel Stokes decades earlier. But solving them—actually computing what the atmosphere would do over the next six hours—required calculations of staggering magnitude.

Richardson decided to try anyway.

### The First Numerical Weather Forecast

Richardson's approach was revolutionary in its conception if premature in its execution:

**Step 1: Grid the Atmosphere**
He divided a column of atmosphere over Central Europe into a three-dimensional grid—layers at different heights, cells at different latitudes and longitudes. About 25 cells in all, a crude approximation of reality.

**Step 2: Specify Initial Conditions**
Using weather observations from May 20, 1910, at 7 AM, he specified temperature, pressure, wind speed, and humidity at each grid point.

**Step 3: Apply the Equations**
The Navier-Stokes equations, the laws of thermodynamics, the gas laws—these describe how each quantity changes based on its neighbors and the forces acting on it.

**Step 4: Step Forward in Time**
Compute the change in each variable over a short time step. Update all values. Repeat.

The calculation for a single 6-hour forecast, covering a portion of Europe with his coarse grid, took Richardson about six weeks. He performed the arithmetic by hand, using logarithm tables and mechanical calculators, in the spaces between driving wounded soldiers to aid stations.

### The Result: Complete Failure

When Richardson completed his calculation, the predicted change in surface pressure was 145 millibars over 6 hours. The actual change was less than 1 millibar. His forecast was wrong by a factor of more than 100.

The error was devastating—but instructive. Richardson identified two major problems:

1. **Initial Conditions**: The observations were sparse and contained errors. Small errors in initial conditions led to large errors in the forecast.

2. **Numerical Instability**: The time step was too large. The equations demanded a shorter step to remain stable—but that would multiply the computational burden many times over.

### The Vision: A Weather Factory

Despite the failure, Richardson's 1922 book *Weather Prediction by Numerical Process* contained a remarkable passage—perhaps the first description of what we would now call massively parallel computing:

> "After so much hard work, I find it difficult to believe that all this computation can lead to nothing. Perhaps some day in the dim future it will be possible to advance the computations faster than the weather advances and at a cost less than the saving to mankind due to the information gained. But that is a dream."

He then described his "forecast factory":

> "Imagine a large hall like a theatre... The walls of this chamber are painted to form a map of the globe... A myriad of computers are at work upon the weather of the part of the map where each sits... From the floor of the pit a tall pillar rises to half the height of the hall. It carries a large pulpit on its top. In this sits the man in charge of the whole theatre... His staff are equipped with coloured signal lights... If a computer is ahead of the rest, a green light is shown; if behind, a red light. The man in charge of the whole theatre needs only to glance at the lights to see how the work is progressing."

Richardson estimated that 64,000 human computers, working in coordinated shifts, could just barely keep pace with the weather—producing forecasts as fast as the weather itself evolved.

### From Dream to Reality

Richardson's dream had to wait for electronic computers. The first successful numerical weather forecast came in 1950, when a team led by Jule Charney at the Institute for Advanced Study in Princeton used the ENIAC computer to produce 24-hour forecasts.

The ENIAC calculation took 24 hours to produce a 24-hour forecast—barely keeping pace with the weather, just as Richardson had imagined. But computers improved exponentially while the weather stayed the same.

By the 1970s, numerical weather prediction had surpassed all other methods. By the 2000s, a 5-day forecast was as reliable as a 2-day forecast had been in the 1980s. Today, ensemble forecasting—running the equations many times with slightly different initial conditions—provides probabilistic predictions with calibrated uncertainty.

### Richardson's Other Legacy: Chaos and Limits

Richardson's failed forecast hinted at a fundamental truth that would take decades to articulate fully. In 1963, Edward Lorenz discovered that atmospheric equations exhibit **sensitive dependence on initial conditions**—the butterfly effect. Small errors in initial conditions grow exponentially, making long-term prediction fundamentally impossible regardless of computational power.

The practical limit of weather prediction is about 10-14 days. Beyond that, chaos overwhelms signal, and forecasts become no better than climatology (historical averages for that date).

This is not a failure of meteorology but a property of the atmosphere itself. Richardson's work helped launch a century of understanding about both what we can predict and what we cannot.

### The Man Behind the Mathematics

Richardson's story extends beyond weather. After the war, he became a pioneer of mathematical psychology, attempting to model the arms race spiral that led to World War I. He developed **Richardson's Arms Race Model**, a system of differential equations describing how fear and rivalry drive military spending—one of the first applications of dynamical systems to social science.

He spent his final years on the mathematics of the length of geographic features, discovering that coastlines have no definite length—their measured length increases as you use finer rulers. This observation would later inspire Benoit Mandelbrot's work on fractals.

Richardson was nominated for the Nobel Peace Prize for his quantitative research on the causes of war. He died in 1953, just as electronic computers were beginning to realize his vision of numerical weather prediction.

### Why This Story Matters for Data Science

Richardson's story embodies several timeless lessons:

1. **The Audacity of Abstraction**: Richardson believed the atmosphere, in all its complexity, could be reduced to equations and computed. This faith in mathematical modeling underlies all of data science.

2. **The Value of Failure**: His forecast was spectacularly wrong, but the errors were instructive. He diagnosed the problems—data quality and numerical stability—that future scientists would solve.

3. **Computational Thinking**: Richardson imagined the forecast factory decades before computers existed. The ability to think algorithmically, to see computation as a solution, is the essence of computational thinking.

4. **The Limits of Prediction**: Richardson's work led, eventually, to chaos theory and a proper understanding of predictability. Not all futures can be forecast, and knowing the limits is as important as pushing them.

5. **Persistence and Vision**: Working by hand in a war zone, Richardson pursued a calculation that everyone thought impossible. It was impossible—for him, at that time. But his vision was correct, and within a generation, it was realized.

---

## LECTURE PLAN: Forecasting - From Richardson's Dream to Modern Prediction

### Learning Objectives
By the end of this lecture, students will be able to:
1. Explain the components of time series (trend, seasonality, noise)
2. Understand stationarity and why it matters
3. Build and interpret simple forecasting models (AR, MA, ARIMA)
4. Apply time series decomposition techniques
5. Appreciate the limits of predictability and the role of uncertainty

### Lecture Structure (90 minutes)

#### Opening Hook (8 minutes)
**The Weather Calculation**
- Present Richardson's 1916 challenge: forecast the weather using mathematics alone
- Show images of early weather maps, the calculation sheets
- Ask: "How long did this calculation take? How accurate was it?"
- Reveal: 6 weeks of hand calculation for a 6-hour forecast—that was 100x wrong
- Pose: "What went wrong? And how do we do it today?"

#### Part 1: The Nature of Time Series (15 minutes)

**What Makes Time Data Special? (5 minutes)**
- Sequential dependence: past affects future
- Demo: shuffle the order of a time series—lose all structure
- Show examples: stock prices, temperature, retail sales
- The fundamental question: What comes next?

**Components of Time Series (10 minutes)**
- Interactive decomposition: show a time series, ask students to identify:
  - Trend (long-term direction)
  - Seasonality (repeating patterns)
  - Noise (random fluctuations)
- Live demo: decompose real data (airline passengers, retail sales)
- Additive vs. multiplicative decomposition
- Python: `from statsmodels.tsa.seasonal import seasonal_decompose`

#### Part 2: Stationarity and Preparation (12 minutes)

**Why Stationarity Matters (5 minutes)**
- Definition: statistical properties don't change over time
- The random walk: stock prices, drunk person walking
- Why non-stationarity breaks forecasting: spurious correlations
- Demo: regress two independent random walks—high R² every time

**Making Series Stationary (7 minutes)**
- Differencing: remove trend by subtracting previous value
- Log transformation: stabilize variance
- Seasonal differencing: remove seasonality
- Live demo: transform a trending series to stationary
- The Dickey-Fuller test for stationarity

#### Part 3: Building Forecasting Models (25 minutes)

**The Autocorrelation Function (7 minutes)**
- Question: "How much does today's value depend on yesterday's?"
- ACF: correlation between series and its lags
- PACF: correlation after removing intermediate effects
- Interactive: calculate ACF for simple example on board
- Show: ACF/PACF plots for different types of series

**AR and MA Models (8 minutes)**
- AR(1): tomorrow = α × today + noise
- Demo: simulate AR(1) with different α values
- MA(1): effect of past shocks persisting
- When to use which: look at ACF/PACF signatures

**ARIMA: Putting It Together (10 minutes)**
- ARIMA(p, d, q): AR + differencing + MA
- The Box-Jenkins methodology:
  1. Plot and identify transformations needed
  2. Examine ACF/PACF to determine p, q
  3. Fit model, check residuals
  4. Forecast
- Live demo: build ARIMA model for airline passengers
- Show prediction intervals, not just point forecasts

#### Part 4: The Limits of Prediction (15 minutes)

**Back to Richardson: Chaos and Uncertainty (7 minutes)**
- Richardson's forecast failed because of chaos
- Edward Lorenz and the butterfly effect (1963)
- Sensitive dependence: small errors grow exponentially
- The 14-day weather barrier
- Video clip or animation of Lorenz attractor

**Predictability Across Domains (8 minutes)**
- Weather: ~10-14 days predictable
- Earthquakes: essentially unpredictable
- Stock prices: short-term random walk, long-term trends
- Epidemics: depends on R and interventions
- Key insight: know what you can and can't predict

#### Part 5: Modern Methods and Machine Learning (10 minutes)

**Beyond ARIMA (5 minutes)**
- Exponential smoothing and its simplicity
- Prophet: handling holidays and changepoints
- LSTMs and Transformers for sequence prediction
- Foundation models: TimeGPT, Lag-Llama

**Forecasting at Scale (5 minutes)**
- Amazon: millions of products, automated model selection
- Hierarchical forecasting: parts must sum to whole
- Probabilistic forecasts for decision-making

#### Wrap-Up and Preview (5 minutes)
- Recap: decomposition → stationarity → model building → limits
- Richardson's legacy: the dream that became reality
- Preview the hands-on exercise
- Key message: "Forecast uncertainty is as important as the forecast itself"

### Materials Needed
- Time series visualization software (Python/Jupyter)
- Historical time series data (airline passengers, daily temperatures)
- Interactive ACF/PACF demonstration
- Video clips of chaos/butterfly effect

### Discussion Questions
1. Why did Richardson's first forecast fail so badly?
2. If stock prices are random walks, why do people think they can predict them?
3. How would you decide if a time series is predictable or not?
4. What's the difference between trend and drift?

---

## HANDS-ON EXERCISE: Time Series Analysis and Forecasting with Python

### Overview
In this exercise, students will:
1. Explore and decompose time series data
2. Test for stationarity and apply transformations
3. Build and evaluate ARIMA models
4. Compare different forecasting approaches

### Prerequisites
- Python 3.8+
- Libraries: pandas, numpy, matplotlib, statsmodels, scikit-learn
- Dataset: Airline passengers, or other provided time series

### Setup

```python
# Install required packages
# pip install pandas numpy matplotlib statsmodels scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
```

### Part 1: Loading and Exploring Time Series Data (15 minutes)

```python
# Load classic airline passengers dataset
from statsmodels.datasets import get_rdataset

# Get the AirPassengers dataset
air = get_rdataset('AirPassengers').data
air.columns = ['date', 'passengers']
air['date'] = pd.date_range(start='1949-01-01', periods=len(air), freq='MS')
air.set_index('date', inplace=True)

print("Dataset shape:", air.shape)
print("\nFirst few rows:")
print(air.head())
print("\nBasic statistics:")
print(air.describe())
```

**Task 1.1**: Plot the time series and identify visually:
- Is there a trend?
- Is there seasonality?
- Does the variance appear constant?

```python
# Plot the time series
plt.figure(figsize=(14, 6))
plt.plot(air.index, air['passengers'], 'b-', linewidth=1.5)
plt.title('Monthly Airline Passengers (1949-1960)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Passengers (thousands)')
plt.tight_layout()
plt.show()

# What do you observe about:
# 1. The overall trend?
# 2. The seasonal pattern?
# 3. The variance over time?
```

### Part 2: Time Series Decomposition (20 minutes)

```python
# Perform seasonal decomposition
# Use multiplicative model since variance increases with level
decomposition = seasonal_decompose(air['passengers'], model='multiplicative', period=12)

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

axes[0].plot(air.index, air['passengers'], 'b-')
axes[0].set_title('Original Series')
axes[0].set_ylabel('Passengers')

axes[1].plot(air.index, decomposition.trend, 'g-')
axes[1].set_title('Trend Component')
axes[1].set_ylabel('Trend')

axes[2].plot(air.index, decomposition.seasonal, 'r-')
axes[2].set_title('Seasonal Component')
axes[2].set_ylabel('Seasonal')

axes[3].plot(air.index, decomposition.resid, 'purple')
axes[3].set_title('Residual Component')
axes[3].set_ylabel('Residual')

plt.tight_layout()
plt.show()
```

**Task 2.1**: Analyze the decomposition:
- In which months is the seasonal factor highest/lowest?
- How much does the trend grow over the period?
- Are the residuals random or is there remaining pattern?

```python
# Examine the seasonal pattern
seasonal_means = decomposition.seasonal[:12]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(10, 5))
plt.bar(months, seasonal_means.values)
plt.title('Seasonal Factors by Month')
plt.ylabel('Multiplicative Factor')
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Which month has the highest travel?
# Which has the lowest?
```

### Part 3: Stationarity Analysis (20 minutes)

```python
def test_stationarity(series, title="Time Series"):
    """
    Perform Augmented Dickey-Fuller test and visualize.
    """
    # Rolling statistics
    rolling_mean = series.rolling(window=12).mean()
    rolling_std = series.rolling(window=12).std()

    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(series.index, series.values, label='Original', color='blue')
    plt.plot(rolling_mean.index, rolling_mean.values, label='12-month Rolling Mean', color='red')
    plt.plot(rolling_std.index, rolling_std.values, label='12-month Rolling Std', color='green')
    plt.legend()
    plt.title(f'{title}: Rolling Mean & Std')
    plt.tight_layout()
    plt.show()

    # Dickey-Fuller test
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')
    print()
    if result[1] < 0.05:
        print("Result: Series IS stationary (reject null hypothesis)")
    else:
        print("Result: Series is NOT stationary (fail to reject null hypothesis)")


# Test original series
test_stationarity(air['passengers'], "Airline Passengers")
```

**Task 3.1**: The original series is not stationary. Apply transformations to make it stationary.

```python
# Apply log transformation to stabilize variance
air['log_passengers'] = np.log(air['passengers'])

# Apply differencing to remove trend
air['log_diff'] = air['log_passengers'].diff()

# Apply seasonal differencing to remove seasonality
air['log_diff_seasonal'] = air['log_passengers'].diff(12)

# Test transformed series
test_stationarity(air['log_diff'].dropna(), "Log-Differenced Series")
test_stationarity(air['log_diff_seasonal'].dropna(), "Log-Seasonal-Differenced Series")
```

### Part 4: ACF and PACF Analysis (15 minutes)

```python
# Plot ACF and PACF for the stationary series
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Use the log-differenced series
series_for_analysis = air['log_diff'].dropna()

plot_acf(series_for_analysis, ax=axes[0], lags=36)
axes[0].set_title('Autocorrelation Function (ACF)')

plot_pacf(series_for_analysis, ax=axes[1], lags=36, method='ywm')
axes[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()
```

**Task 4.1**: Interpret the ACF and PACF plots:
- What lags show significant autocorrelation?
- What do the seasonal spikes at lags 12, 24, 36 indicate?
- Based on PACF, what AR order might you choose?
- Based on ACF, what MA order might you choose?

### Part 5: Building ARIMA Models (25 minutes)

```python
# Split data into train and test
train = air['passengers'][:'1958-12-31']
test = air['passengers']['1959-01-01':]

print(f"Training set: {len(train)} observations")
print(f"Test set: {len(test)} observations")

# Fit ARIMA model
# ARIMA(p, d, q) x (P, D, Q, s) for seasonal
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Start with a simple model: ARIMA(1,1,1) x (1,1,1,12)
model = SARIMAX(train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)

print(results.summary())
```

**Task 5.1**: Check the residuals of your model:

```python
# Residual diagnostics
results.plot_diagnostics(figsize=(14, 10))
plt.tight_layout()
plt.show()

# The ideal residuals should:
# 1. Show no autocorrelation (ACF within bounds)
# 2. Be approximately normally distributed
# 3. Show no patterns in the residual time plot
```

**Task 5.2**: Generate forecasts and evaluate:

```python
# Forecast the test period
forecast = results.get_forecast(steps=len(test))
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot forecast vs actual
plt.figure(figsize=(14, 6))
plt.plot(train.index, train.values, label='Training Data', color='blue')
plt.plot(test.index, test.values, label='Actual Test Data', color='green')
plt.plot(test.index, forecast_mean, label='Forecast', color='red')
plt.fill_between(test.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 color='red', alpha=0.2, label='95% CI')
plt.legend()
plt.title('SARIMA Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.tight_layout()
plt.show()

# Calculate error metrics
mae = mean_absolute_error(test, forecast_mean)
rmse = np.sqrt(mean_squared_error(test, forecast_mean))
mape = np.mean(np.abs((test - forecast_mean) / test)) * 100

print(f"\nForecast Accuracy Metrics:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
```

### Part 6: Comparing Methods (15 minutes)

```python
def forecast_and_evaluate(name, forecasts):
    """Evaluate forecast against test set."""
    mae = mean_absolute_error(test, forecasts)
    rmse = np.sqrt(mean_squared_error(test, forecasts))
    mape = np.mean(np.abs((test - forecasts) / test)) * 100
    return {'Method': name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

results_list = []

# 1. Naive forecast (last observed value)
naive_forecast = pd.Series([train.iloc[-1]] * len(test), index=test.index)
results_list.append(forecast_and_evaluate('Naive', naive_forecast))

# 2. Seasonal naive (same month last year)
seasonal_naive = train.iloc[-12:].values
results_list.append(forecast_and_evaluate('Seasonal Naive', seasonal_naive))

# 3. Simple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ses_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
ses_results = ses_model.fit()
ses_forecast = ses_results.forecast(len(test))
results_list.append(forecast_and_evaluate('Holt-Winters', ses_forecast))

# 4. Our SARIMA model (already computed above)
results_list.append(forecast_and_evaluate('SARIMA', forecast_mean))

# Compare methods
comparison = pd.DataFrame(results_list)
print("\nMethod Comparison:")
print(comparison.to_string(index=False))

# Visualize comparison
plt.figure(figsize=(14, 6))
plt.plot(test.index, test.values, 'ko-', label='Actual', markersize=4)
plt.plot(test.index, naive_forecast, 'r--', label='Naive', alpha=0.7)
plt.plot(test.index, seasonal_naive, 'g--', label='Seasonal Naive', alpha=0.7)
plt.plot(test.index, ses_forecast, 'b--', label='Holt-Winters', alpha=0.7)
plt.plot(test.index, forecast_mean, 'm-', label='SARIMA', linewidth=2)
plt.legend()
plt.title('Forecast Comparison')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.tight_layout()
plt.show()
```

### Challenge Questions

1. **Model Selection**: Try different ARIMA orders. Does ARIMA(2,1,2) x (1,1,1,12) perform better or worse? What about ARIMA(0,1,1) x (0,1,1,12)?

2. **Forecast Horizon**: How does accuracy degrade as you forecast further ahead? Plot MAPE vs. forecast horizon.

3. **Alternative Data**: Apply these methods to a different time series (temperature, stock prices, web traffic). What patterns do you find? What models work best?

4. **Uncertainty**: The 95% confidence interval gets wider as you forecast further ahead. Why? What does this mean for long-term planning?

5. **Richardson's Challenge**: If you had to forecast 6 hours of weather by hand, what approach would you take? How is it similar to/different from time series methods?

### Expected Outputs

Students should submit:
1. Decomposition analysis of the time series with interpretations
2. Stationarity tests and appropriate transformations
3. ACF/PACF plots with interpretation of model orders
4. At least two different ARIMA models with comparison
5. Forecast accuracy evaluation against a baseline
6. Written reflection on forecast uncertainty and limits

### Evaluation Rubric

| Criteria | Points |
|----------|--------|
| Correct decomposition and interpretation | 15 |
| Stationarity testing and transformation | 15 |
| ACF/PACF analysis and model identification | 15 |
| ARIMA model fitting and diagnostics | 20 |
| Forecast evaluation and comparison | 20 |
| Code quality and documentation | 15 |
| **Total** | **100** |

---

## Recommended Resources

### Books

**Technical**
- *Time Series Analysis: Forecasting and Control* by Box, Jenkins, Reinsel, and Ljung - The classic reference
- *Forecasting: Principles and Practice* by Hyndman and Athanasopoulos - Free online, modern, practical
- *Time Series Analysis and Its Applications* by Shumway and Stoffer - Graduate-level with R examples
- *Introduction to Time Series and Forecasting* by Brockwell and Davis - Rigorous but accessible

**Historical and Popular**
- *The Signal and the Noise* by Nate Silver - Forecasting in politics, sports, weather, and more
- *Superforecasting* by Philip Tetlock - The science of prediction
- *Weather Prediction by Numerical Process* by L.F. Richardson - The original 1922 book (available free online)
- *Chaos* by James Gleick - Popular account of chaos theory and Lorenz

### Academic Papers

- **Box, G.E.P. & Jenkins, G.M. (1970)**. "Time Series Analysis: Forecasting and Control" - The foundational work
- **Hyndman, R.J., et al. (2006)**. "Another Look at Measures of Forecast Accuracy" - On MASE metric
- **Lorenz, E.N. (1963)**. "Deterministic Nonperiodic Flow" - The chaos theory paper
- **Makridakis, S., et al. (2020)**. "The M4 Competition" - State-of-the-art forecasting comparison
- **Salinas, D., et al. (2020)**. "DeepAR: Probabilistic Forecasting with Autoregressive RNNs"
- **Lim, B., et al. (2021)**. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"

### Video Lectures

- **StatQuest: Time Series Analysis** - Clear, visual explanations
- **MIT 18.S096: Mathematical Concepts and Methods for Finance** - Includes time series
- **Forecasting: Principles and Practice (YouTube)** - Rob Hyndman's lectures
- **3Blue1Brown: Fourier Transform** - Essential for understanding spectral methods

### Online Courses

- **Coursera: Practical Time Series Analysis** - State University of New York
- **Udacity: Time Series Forecasting** - Practical applications
- **DataCamp: Time Series with Python** - Hands-on coding
- **Fast.ai: Practical Deep Learning** - Includes sequence models

### Tools and Libraries

- **statsmodels** (https://www.statsmodels.org/) - Statistical modeling in Python
- **Prophet** (https://facebook.github.io/prophet/) - Facebook's forecasting tool
- **sktime** (https://www.sktime.net/) - Scikit-learn for time series
- **GluonTS** (https://ts.gluon.ai/) - Deep learning for time series
- **Darts** (https://unit8co.github.io/darts/) - Easy-to-use forecasting library
- **tsfresh** - Automatic feature extraction for time series

### Datasets

- **M-Competitions** - Thousands of time series for benchmarking
- **UCI Time Series Repository** - Diverse time series datasets
- **Kaggle Competitions** - Store sales, web traffic, energy demand
- **FRED** (Federal Reserve Economic Data) - Economic time series
- **Climate Data Store** - Weather and climate data
- **Yahoo Finance / Alpha Vantage** - Financial time series

---

## References

1. Box, G.E.P., & Jenkins, G.M. (1970). *Time Series Analysis: Forecasting and Control*. San Francisco: Holden-Day.

2. Richardson, L.F. (1922). *Weather Prediction by Numerical Process*. Cambridge University Press.

3. Lorenz, E.N. (1963). "Deterministic Nonperiodic Flow." *Journal of the Atmospheric Sciences*, 20(2), 130-141.

4. Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

5. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). "The M4 Competition: 100,000 Time Series and 61 Forecasting Methods." *International Journal of Forecasting*, 36(1), 54-74.

6. Taylor, S.J., & Letham, B. (2018). "Forecasting at Scale." *The American Statistician*, 72(1), 37-45.

7. Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks." *International Journal of Forecasting*, 36(3), 1181-1191.

8. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.

9. Tetlock, P.E., & Gardner, D. (2015). *Superforecasting: The Art and Science of Prediction*. Crown Publishers.

10. Lynch, P. (2006). *The Emergence of Numerical Weather Prediction*. Cambridge University Press.

11. Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction Problems." *Journal of Basic Engineering*, 82(1), 35-45.

12. Charney, J.G., Fjørtoft, R., & von Neumann, J. (1950). "Numerical Integration of the Barotropic Vorticity Equation." *Tellus*, 2(4), 237-254.

---

*Module 9 explores the quest to predict the future through the analysis of time series data. From Lewis Fry Richardson's visionary but premature attempt to forecast weather by hand to modern deep learning systems, we trace the evolution of methods that seek patterns in time—while learning to respect the fundamental limits that chaos imposes on predictability.*
