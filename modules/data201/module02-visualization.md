---
layout: default
title: "Module 2: Data Structures and Data Visualization"
---

# Module 2: Data Structures and Data Visualization
## "Telling Stories with Data"

*Research Document for DATA 201 Course Development*

---

# Table of Contents

1. [Introduction](#introduction)
2. [Part I: The Evolution of Data Representation](#part-i-the-evolution-of-data-representation)
3. [Part II: Pioneers of Information Design](#part-ii-pioneers-of-information-design)
4. [Part III: The Science of Visual Perception](#part-iii-the-science-of-visual-perception)
5. [Part IV: Modern Data Visualization](#part-iv-modern-data-visualization)
6. [Part V: Data Structures - From Biology to Networks](#part-v-data-structures---from-biology-to-networks)
7. [DEEP DIVE: W.E.B. Du Bois's Data Portraits](#deep-dive-web-du-boiss-data-portraits)
8. [Lecture Plan and Hands-On Exercise](#lecture-plan-and-hands-on-exercise)
9. [Recommended Resources](#recommended-resources)
10. [References](#references)

---

# Introduction

This module explores how humans have developed ways to represent and visualize data across millennia—from ancient maps to modern interactive dashboards. We examine:

- **Data structures**: How we organize information (lists, matrices, trees, graphs, images)
- **Visual encoding**: How we map data to visual properties (position, color, size, shape)
- **Storytelling**: How visualizations communicate insights and drive action

The central question: **How do we transform raw data into understanding?**

---

# Part I: The Evolution of Data Representation

## From Cave Paintings to Coordinate Systems

### The Oldest "Data Visualizations"

Human beings have been visualizing information for at least 40,000 years. Cave paintings at Lascaux (France) and Altamira (Spain) include tally marks that may represent lunar cycles or hunting counts—proto-data visualizations carved in stone.

### Ancient Maps: The First Spatial Data

**The Babylonian World Map (c. 600 BCE)**
- Oldest known world map
- Babylon at center, surrounded by a circular ocean
- Represents both geographic knowledge and cosmological beliefs

**Ptolemy's *Geographia* (2nd century CE)**
- Introduced latitude and longitude coordinates
- Catalogued over 8,000 locations
- Systematic approach to spatial data representation

### The Coordinate Revolution: René Descartes (1637)

René Descartes's invention of the Cartesian coordinate system in *La Géométrie* transformed how we represent relationships between variables. For the first time, abstract mathematical relationships could be visualized as geometric shapes.

**The Data Journey:**
- **Collection:** Mathematical relationships expressed as equations
- **Understanding:** Mapping algebraic expressions to geometric forms
- **Prediction:** Using visual intuition to understand mathematical behavior

---

## The Mercator Projection (1569): When Maps Lie

Gerardus Mercator, a Flemish cartographer, created his famous projection in 1569 specifically for navigation. His innovation: representing rhumb lines (constant compass bearings) as straight lines.

### Why Mercator Worked for Navigation

Sailors could plot a straight course on the map and follow it with a constant compass bearing—revolutionary for ocean navigation.

### The Hidden Distortion

The Mercator projection preserves angles but distorts areas. Landmasses near the poles appear vastly larger than they are:
- Greenland appears larger than South America (actual ratio: 1:8)
- Africa appears smaller than it is (actual area: larger than US, China, India, and Europe combined)

### The Politics of Projection

In 1974, historian Arno Peters promoted an "equal area" projection, arguing that Mercator's distortions reinforced colonial perceptions—making European countries appear larger relative to Africa and South America.

**Modern Alternatives:**
- Robinson projection (used by National Geographic, 1988-1998)
- Winkel Tripel (National Geographic, 1998-present)
- Equal Earth projection (supported by African Union, 2018)

**Key Insight:** Every map projection involves trade-offs. There is no "neutral" way to flatten a sphere—all choices embed values and priorities.

### Sources
- [Wikipedia - Mercator projection](https://en.wikipedia.org/wiki/Mercator_projection)
- [Britannica - Mercator projection](https://www.britannica.com/science/Mercator-projection)
- [Atlas.co - History of Cartography](https://atlas.co/blog/history-of-cartography/)

---

# Part II: Pioneers of Information Design

## Otto Neurath and ISOTYPE (1920s-1930s)
**The Visual Language That Became Modern Infographics**

Austrian sociologist Otto Neurath believed that visual communication could democratize knowledge. In 1920s Vienna, he developed ISOTYPE (International System of Typographic Picture Education)—a standardized pictogram language for showing social and economic data.

### The Vienna Method

At the Social and Economic Museum of Vienna (1925-1934), Neurath and his collaborators created a system of:
- **Standardized symbols** designed by Gerd Arntz (over 4,000 pictograms)
- **Consistent rules** for combining symbols
- **Serial repetition** where each symbol represents a fixed quantity

### Philosophy: "Words Divide, Pictures Unite"

Neurath believed visual statistics could communicate across language barriers and education levels. ISOTYPE was designed for "visual education"—always accompanied by text, but making complex data accessible to ordinary citizens.

### The "Transformer"

Marie Reidemeister (later Marie Neurath) served as the "transformer"—the crucial role of translating data and ideas from subject experts into visual form. This anticipated the modern role of data visualization designers.

### Legacy

ISOTYPE's influence appears everywhere today:
- Airport signage and international symbols
- Olympic pictograms
- Modern infographics
- Data dashboards

### Sources
- [The Marginalian - Otto Neurath's ISOTYPE](https://www.themarginalian.org/2018/12/10/exact-thinking-in-demented-times-otto-neurath-isotype/)
- [Wikipedia - ISOTYPE](https://en.wikipedia.org/wiki/Isotype_(picture_language))
- [Stanford Encyclopedia - Otto Neurath Visual Education](https://plato.stanford.edu/entries/neurath/visual-education.html)

---

## Jacques Bertin's *Semiology of Graphics* (1967)
**The First Theory of Data Visualization**

French cartographer Jacques Bertin published *Sémiologie graphique* in 1967—the first systematic theoretical foundation for information graphics.

### The Seven Visual Variables

Bertin identified seven fundamental visual properties that can encode data:

| Variable | Best For | Example |
|----------|----------|---------|
| **Position** | Quantitative data | X-Y coordinates on scatter plot |
| **Size** | Quantitative data | Bubble size in bubble chart |
| **Value** (lightness) | Ordered data | Light to dark shading |
| **Texture** | Categorical data | Patterns in maps |
| **Color** (hue) | Categorical data | Different colors for categories |
| **Orientation** | Categorical data | Angle of lines |
| **Shape** | Categorical data | Circles vs. squares |

### Planar vs. Retinal Variables

Bertin distinguished between:
- **Planar variables**: Position (x, y) in the visualization space
- **Retinal variables**: Properties perceived by the eye (size, color, texture, etc.)

### Influence

Bertin's framework influenced:
- Computer graphics research
- GIS and cartography
- Modern visualization tools (D3.js, ggplot2)
- Leland Wilkinson's *Grammar of Graphics* → ggplot2

### Sources
- [History of Information - Bertin's Sémiologie graphique](https://www.historyofinformation.com/detail.php?id=3361)
- [Wikipedia - Visual variable](https://en.wikipedia.org/wiki/Visual_variable)

---

## Edward Tufte: The Leonardo da Vinci of Data
**Chartjunk, Data-Ink, and Graphical Excellence**

Edward Tufte, professor emeritus at Yale, published *The Visual Display of Quantitative Information* in 1983—probably the most influential book on data visualization ever written.

### Core Principles

**1. The Data-Ink Ratio**

$$\text{Data-Ink Ratio} = \frac{\text{Ink used to display data}}{\text{Total ink used in graphic}}$$

Maximize this ratio. Remove everything that doesn't contribute to understanding.

**2. Chartjunk**

Tufte coined "chartjunk" to describe unnecessary decorative elements:
- 3D effects that distort perception
- Excessive gridlines
- Decorative illustrations
- Redundant labels

**3. The Lie Factor**

$$\text{Lie Factor} = \frac{\text{Size of effect shown in graphic}}{\text{Size of effect in data}}$$

A Lie Factor of 1 is truthful. Tufte found examples with Lie Factors of 14.8—grossly misrepresenting the underlying data.

**4. Small Multiples**

Display multiple small graphics to reveal patterns across different conditions or time periods.

**5. Sparklines**

Tufte invented "sparklines"—word-sized graphics that can be embedded in text, maximizing data-ink ratio.

### The Self-Published Books

Tufte self-published his books after being dissatisfied with traditional publishers' treatment of visual content. They became design classics:
- *The Visual Display of Quantitative Information* (1983)
- *Envisioning Information* (1990)
- *Visual Explanations* (1997)
- *Beautiful Evidence* (2006)

### Modern Critique

Recent research suggests Tufte's minimalism isn't always optimal:
- Some "chartjunk" may increase memorability
- Decorative elements can increase engagement
- Context matters—scientific papers vs. public communication

### Sources
- [InfoVis Wiki - Data-Ink Ratio](https://infovis-wiki.net/wiki/Data-Ink_Ratio)
- [EU Data Visualization Guide - Chart junk and data ink](https://data.europa.eu/apps/data-visualisation-guide/chart-junk-and-data-ink-origins)
- [GeeksforGeeks - Tufte's Principles](https://www.geeksforgeeks.org/data-visualization/mastering-tuftes-data-visualization-principles/)

---

# Part III: The Science of Visual Perception

## Anscombe's Quartet (1973)
**Why We Must Always Visualize Data**

In 1973, statistician Francis Anscombe constructed four datasets that revolutionized how we think about data analysis.

### The Four Datasets

All four datasets have nearly identical statistical properties:
- Same mean of X: 9
- Same mean of Y: 7.50
- Same variance of X: 11
- Same variance of Y: 4.12
- Same correlation: 0.816
- Same linear regression line: y = 3.00 + 0.500x

### But When Plotted...

The four datasets reveal completely different patterns:
1. **Dataset I**: Normal linear relationship with scatter
2. **Dataset II**: Perfect curved (quadratic) relationship—not linear at all!
3. **Dataset III**: Perfect linear relationship with one outlier
4. **Dataset IV**: No relationship except for one extreme outlier

### The Lesson

> "Numerical calculations are exact, but graphs are rough" — a misconception Anscombe sought to counter.

Summary statistics can completely obscure the nature of your data. **Always visualize.**

### The Datasaurus Dozen (2017)

Researchers at Autodesk extended Anscombe's idea to create the "Datasaurus Dozen"—twelve datasets with identical statistics that form:
- A dinosaur
- Stars
- Circles
- Lines at various angles

"Never trust summary statistics alone; always visualize your data."

### Sources
- [Wikipedia - Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet)
- [Autodesk Research - Same Stats, Different Graphs](https://www.research.autodesk.com/publications/same-stats-different-graphs/)

---

## Pre-Attentive Processing
**What the Eye Sees Before the Brain Thinks**

Certain visual features are processed by the brain in less than 250 milliseconds—before conscious attention engages. These "pre-attentive" features enable instant pattern recognition.

### Pre-Attentive Visual Features

**Highly Pre-Attentive:**
- Color (hue)
- Position
- Size
- Orientation
- Motion

**Moderately Pre-Attentive:**
- Shape
- Enclosure
- Density

### Design Implications

1. Use pre-attentive features to highlight the most important information
2. Don't use too many pre-attentive channels simultaneously (visual overload)
3. Color is powerful but should be used consistently

### Gestalt Principles

The Gestalt psychologists identified how we perceive visual groupings:
- **Proximity**: Items close together are seen as a group
- **Similarity**: Similar items are grouped together
- **Continuity**: We follow continuous lines and curves
- **Closure**: We complete incomplete shapes
- **Enclosure**: Items in a boundary are grouped

These principles inform effective chart design.

---

# Part IV: Modern Data Visualization

## Hans Rosling and Gapminder
**The Best Stats You've Ever Seen**

Hans Rosling (1948-2017), a Swedish physician and professor, transformed how the world sees global development data.

### The 2006 TED Talk

Rosling's TED talk "The Best Stats You've Ever Seen" is one of the most viewed TED videos ever. In 19 minutes, he:
- Debunked myths about "developing" vs. "developed" countries
- Showed 200 years of progress in 4 minutes
- Made statistics feel like sports commentary

> "I produce a road-less sound."

### The Gapminder Bubble Chart

The visualization showed:
- **X-axis**: Income per person (log scale)
- **Y-axis**: Life expectancy
- **Bubble size**: Population
- **Color**: Region/continent
- **Animation**: Changes over time (1800-present)

### The Tool: Trendalyzer

Rosling's son Ola built Trendalyzer, later acquired by Google (2007) and released as Google Motion Charts and Public Data Explorer.

### Rosling's Teaching Philosophy

Rosling didn't just show data—he told stories:
- "The world is not divided into 'us' and 'them'"
- Progress is real but uneven
- Data literacy is democratic empowerment

### Legacy

- Gapminder Foundation continues his work
- *Factfulness* (2018), co-authored with family, became a bestseller
- Free teaching materials at gapminder.org

### Sources
- [TED - Hans Rosling Speaker Page](https://www.ted.com/speakers/hans_rosling)
- [Wikipedia - Hans Rosling](https://en.wikipedia.org/wiki/Hans_Rosling)
- [Gapminder - Teaching Materials](https://www.gapminder.org/teaching/materials/)

---

## The Grammar of Graphics
**From Bertin to ggplot2**

### Leland Wilkinson's *Grammar of Graphics* (1999)

Wilkinson formalized a system for describing any statistical graphic as a composition of layers:
- **Data**: The dataset being visualized
- **Aesthetics**: Mappings from data to visual properties
- **Geometries**: The visual marks (points, lines, bars)
- **Statistics**: Transformations of the data
- **Coordinates**: The coordinate system
- **Facets**: Subplots for different data subsets

### ggplot2 and the Tidyverse

Hadley Wickham implemented these ideas in ggplot2 (2005) for R:

```r
ggplot(data, aes(x = income, y = life_exp, color = continent, size = pop)) +
  geom_point() +
  scale_x_log10() +
  facet_wrap(~year)
```

This declarative approach to visualization has influenced:
- Vega and Vega-Lite
- Altair (Python)
- Observable Plot
- Tableau's VizQL

---

# Part V: Data Structures - From Biology to Networks

## Biological Data: From Lists to Images

### Linnaeus and Taxonomy (1735)

Carl Linnaeus created the modern system of biological classification—hierarchical tree structures that organize all living things:

```
Kingdom → Phylum → Class → Order → Family → Genus → Species
```

This is a fundamental **tree data structure**—the same structure used in file systems, XML/HTML, and decision trees.

### DNA as Data: Rosalind Franklin's Photo 51 (1952)

Rosalind Franklin's X-ray crystallography image—Photo 51—revealed the helical structure of DNA. This single image contained the key information Watson and Crick needed to deduce the double helix.

**The Data Journey:**
- **Collection**: 100+ hours of X-ray exposure on DNA fibers
- **Understanding**: The X-pattern revealed helical structure; spacing revealed base pair distances
- **Prediction**: The structure explained how DNA replicates and stores genetic information

**What the Image Shows:**
- The X-shape indicates a helix
- The spacing between lines reveals the pitch
- The missing fourth layer line suggests two strands (double helix)

> "Watson, Crick, and Wilkins repeatedly acknowledged that they could not have solved the structure without the crystallographic evidence."

Franklin died in 1958 and was not included in the 1962 Nobel Prize. Her contributions went largely unrecognized for decades—she's been called "the dark lady of DNA."

### Sources
- [Wikipedia - Rosalind Franklin](https://en.wikipedia.org/wiki/Rosalind_Franklin)
- [Embryo Project - Photo 51](https://embryo.asu.edu/pages/photograph-51-rosalind-franklin-1952)
- [Science History Institute - DNA Discovery](https://www.sciencehistory.org/education/scientific-biographies/francis-crick-rosalind-franklin-james-watson-and-maurice-wilkins/)

---

## Geographic Data: Maps as Data Structures

### GIS: Geographic Information Systems

GIS represents geographic data in layers:
- **Vector data**: Points, lines, polygons with coordinates
- **Raster data**: Grids of values (elevation, temperature, satellite imagery)
- **Attributes**: Data associated with features

The Canada Geographic Information System (CGIS), developed in the 1960s, was the first true GIS. It enabled layered spatial analysis—overlaying maps to find patterns.

### Modern Applications
- Google Maps and navigation
- Urban planning and zoning
- Environmental monitoring
- Disease mapping (echoing John Snow!)
- Precision agriculture

---

## Network Data: Graphs and Connections

### Euler and the Seven Bridges of Königsberg (1736)

Leonhard Euler asked: Can you walk through Königsberg crossing each of its seven bridges exactly once?

His proof that it's impossible founded **graph theory**—the mathematics of networks.

### Network Data Structures

```
Nodes (vertices): Entities
Edges (links): Relationships between entities
```

**Types of Networks:**
- **Undirected**: Facebook friendships
- **Directed**: Twitter follows, web links
- **Weighted**: Strength of connections
- **Bipartite**: Two types of nodes (users and products)

### Famous Network Datasets
- **Zachary's Karate Club** (1977): Social network split by conflict
- **Facebook social network** (anonymized)
- **Wikipedia hyperlinks**
- **Protein-protein interactions**

---

# DEEP DIVE: W.E.B. Du Bois's Data Portraits
**Visualizing Black America at the 1900 Paris Exposition**

## The Story

In 1900, 37 years after the Emancipation Proclamation, sociologist W.E.B. Du Bois traveled to Paris with a radical mission: to show the world the progress of African Americans through the language of data visualization.

### The Context: Post-Reconstruction America

By 1900, the promises of Reconstruction had been dismantled:
- Jim Crow laws enforced segregation across the South
- Lynching was at its peak (over 100 per year)
- African Americans were systematically excluded from political and economic power
- International perception was shaped by racist stereotypes

### The Paris Exposition Universelle

The 1900 World's Fair in Paris drew 50 million visitors. It showcased technological marvels like the Grande Roue (Ferris wheel), moving sidewalks, and talking films.

Thomas J. Calloway, an African American educator, secured space for "The Exhibit of American Negroes"—and invited Du Bois to create a statistical portrait of Black America.

### Du Bois's Vision

Du Bois, then a professor at Atlanta University, saw an opportunity. He would counter racist narratives not with emotion but with **data**—irrefutable evidence of progress despite systematic oppression.

With his students at Atlanta University, Du Bois created **63 hand-drawn data visualizations**—what he called "data portraits."

## The Visualizations

### Two Series

**Series 1: National/International View**
"A Series of Statistical Charts Illustrating the Condition of the Descendants of Former African Slaves Now in Residence in the United States of America"

**Series 2: The Georgia Negro**
Detailed focus on Georgia's Black population—demographics, economics, education, property

### Design Innovation

Du Bois's visualizations were decades ahead of their time:

**Bold Colors:**
- Vibrant reds, greens, golds, and blacks
- Hand-painted with watercolors
- "Bold colors and geometric shapes were decades ahead of modernist graphic design in America"

**Novel Chart Types:**
- Spiral charts showing change over time
- Comparative bar charts challenging racial assumptions
- Geographic visualizations of migration
- Proportional area charts

**Data Sources:**
- U.S. Census data
- Atlanta University sociology research
- State and federal reports

### Examples of Charts

**"City and Rural Population" (1890)**
- Spiral diagram showing urbanization trends
- Striking visual impact

**"Assessed Value of Household and Kitchen Furniture Owned by Georgia Negroes"**
- Shows growth from $21,186 in 1875 to $498,532 in 1899
- Visual proof of economic progress

**"Occupations of Negroes and Whites in Georgia"**
- Comparative bar chart
- Revealed similarities in occupational distribution

**"Illiteracy"**
- Showed dramatic decline in illiteracy rates
- From 90%+ at Emancipation to under 50% by 1900

### The Message

These weren't neutral statistics. Du Bois deliberately chose data that demonstrated:

1. **Progress**: Despite slavery and oppression, Black Americans were building wealth, education, and institutions
2. **Contribution**: Black labor was essential to American prosperity
3. **Humanity**: Statistics humanized a population dehumanized by stereotypes

> "The problem of the Twentieth Century is the problem of the color line." — Du Bois, 1903

### Awards and Reception

The exhibit won a Gold Medal at the Paris Exposition. International visitors saw evidence that contradicted American racist propaganda.

### Rediscovery

The visualizations were shipped to the Library of Congress and largely forgotten for over a century. They were rediscovered and published in full color in 2018:

*W.E.B. Du Bois's Data Portraits: Visualizing Black America* (Princeton Architectural Press)
- Edited by Whitney Battle-Baptiste and Britt Rusert
- First comprehensive collection in color

### Legacy

Du Bois's work anticipates:
- Modern infographics
- Data journalism
- Data activism
- The recognition that visualization is never neutral—it always involves choices about what to show and how

**Key Insight:** Data visualization is always political. The choice of what to measure, how to present it, and who the audience is shapes the message. Du Bois used this power deliberately—to advocate for justice.

### Sources
- [Smithsonian Magazine - Du Bois's Visionary Infographics](https://www.smithsonianmag.com/history/first-time-together-and-color-book-displays-web-du-bois-visionary-infographics-180970826/)
- [Library of Congress - Du Bois Paris Exposition Charts](https://www.loc.gov/pictures/item/2005679642/)
- [Public Domain Review - Du Bois Hand-Drawn Infographics](https://publicdomainreview.org/collection/w-e-b-du-bois-hand-drawn-infographics-of-african-american-life-1900)
- [Hyperallergic - Du Bois Meticulously Visualized 20th-Century Black America](https://hyperallergic.com/476334/how-w-e-b-du-bois-meticulously-visualized-20th-century-black-america/)
- [Tableau Blog - Du Bois and Data Visualization](https://www.tableau.com/blog/how-web-du-bois-used-data-visualization-confront-prejudice-early-20th-century)

---

# Lecture Plan and Hands-On Exercise

## Lecture Plan: "Data Portraits" (75-90 minutes)

### Part 1: The Power of Visualization (20 min)

**Opening Hook:** Show Anscombe's Quartet
- Display the four datasets as tables only
- Ask: "What can you tell me about these datasets?"
- Calculate and show identical summary statistics
- Reveal the plots—dramatically different!

**Key Message:** "Always visualize your data."

**Transition:** But visualization isn't just about understanding data—it's about communication and persuasion.

### Part 2: The Du Bois Story (25 min)

**Historical Context:**
- 1900: 37 years after Emancipation
- Jim Crow, lynching, systematic oppression
- International stereotypes about Black Americans

**The Challenge:**
- How do you change minds?
- Du Bois's answer: Data + Design

**The Visualizations:**
- Show 5-6 key charts from the Paris Exposition
- Discuss design choices: colors, layout, chart types
- What story does each tell?

**Discussion Questions:**
- Why did Du Bois choose these particular statistics?
- How do design choices affect the message?
- What data is NOT included? Why might that matter?

### Part 3: Design Principles (20 min)

**From Bertin's Visual Variables:**
- Position, size, color, shape, etc.
- Which variables work best for which data types?

**From Tufte's Principles:**
- Data-ink ratio
- Chartjunk
- Show students a "before and after" redesign

**Modern Tools:**
- Brief introduction to visualization libraries
- ggplot2, matplotlib, seaborn, Altair

### Part 4: Hands-On Exercise Introduction (10 min)

Introduce the exercise and available datasets.

---

## Hands-On Exercise: "Creating Your Own Data Portrait"

### Objective
Create a data visualization that tells a story about a social, economic, or environmental issue—inspired by Du Bois's approach.

### Duration
2-3 hours (can be homework)

### Materials Provided

**Datasets (choose one):**

1. **Modern Census Data**
   - US Census API: Education, income, housing by race/ethnicity
   - World Bank indicators: Development data by country

2. **Historical Recreation**
   - Du Bois's original Georgia data (available in R's `duboisr` package)
   - Recreate one of his charts with modern tools

3. **Local Data**
   - Lebanon/AUB-specific datasets
   - Regional economic or social indicators

### Tasks

**Task 1: Data Exploration (30 min)**
```python
import pandas as pd
import seaborn as sns

# Load your chosen dataset
data = pd.read_csv('your_data.csv')

# Explore
print(data.describe())
print(data.info())

# What story could this data tell?
```

**Task 2: Sketch Your Story (20 min)**
- On paper, sketch 2-3 possible visualizations
- What is the main message?
- What comparisons are you making?
- Who is your audience?

**Task 3: Create the Visualization (60 min)**

Using Python (matplotlib/seaborn) or R (ggplot2):

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Recreate a Du Bois-style chart
plt.figure(figsize=(10, 8))
plt.style.use('seaborn-whitegrid')

# Your visualization code here

# Add Du Bois-inspired styling
plt.title('YOUR TITLE HERE', fontsize=16, fontweight='bold')
plt.xlabel('X Label')
plt.ylabel('Y Label')

# Color palette inspired by Du Bois
dubois_colors = ['#dc143c', '#00aa00', '#000000', '#ffc107', '#7b3f00']

plt.tight_layout()
plt.savefig('my_data_portrait.png', dpi=300)
plt.show()
```

**Task 4: Reflection (20 min)**

Write a short paragraph (150-200 words) answering:
1. What story does your visualization tell?
2. What design choices did you make and why?
3. What might be misleading or missing from your visualization?
4. How might different audiences interpret it differently?

### Extension: Du Bois Challenge

Recreate one of Du Bois's original visualizations using modern data:
- Find equivalent modern data (Census, World Bank, etc.)
- Create a "then and now" comparison
- What has changed? What hasn't?

**Resources:**
- `duboisr` R package: https://github.com/ajstarks/dubois-data-portraits
- Du Bois Visualization Challenge (annual): https://github.com/ajstarks/dubois-data-portraits

### Evaluation Criteria

| Criterion | Excellent | Good | Needs Work |
|-----------|-----------|------|------------|
| **Story** | Clear, compelling narrative | Clear but conventional | Unclear or no narrative |
| **Design** | Thoughtful choices, minimal chartjunk | Adequate, some clutter | Cluttered, confusing |
| **Accuracy** | Data represented truthfully | Minor issues | Misleading representation |
| **Reflection** | Deep engagement with choices | Surface-level | Missing or minimal |

---

# Recommended Resources

## Books

### Data Visualization Theory
- **Tufte, Edward.** *The Visual Display of Quantitative Information* (1983) - The classic
- **Tufte, Edward.** *Envisioning Information* (1990) - Escaping flatland
- **Cairo, Alberto.** *The Truthful Art* (2016) - Modern principles
- **Cairo, Alberto.** *How Charts Lie* (2019) - Critical data literacy
- **Wilke, Claus.** *Fundamentals of Data Visualization* (2019) - Free online

### History and Context
- **Battle-Baptiste & Rusert.** *W.E.B. Du Bois's Data Portraits* (2018) - The Paris Exposition visualizations
- **Friendly, Michael.** *A History of Data Visualization and Graphic Communication* (2021)
- **Rendgen, Sandra.** *History of Information Graphics* (2019) - Comprehensive visual history

### Practical Guides
- **Wickham, Hadley.** *ggplot2: Elegant Graphics for Data Analysis* (2016)
- **Schwabish, Jonathan.** *Better Data Visualizations* (2021)
- **Knaflic, Cole Nussbaumer.** *Storytelling with Data* (2015)

## Online Courses

- **Coursera: Data Visualization** (University of Illinois)
- **edX: Data Science: Visualization** (Harvard)
- **Observable: Learn D3** - Interactive JavaScript visualization

## Websites and Tools

### Learning Resources
- **Friendly's History of Data Visualization:** https://friendly.github.io/HistDataVis/
- **Data Visualization Catalogue:** https://datavizcatalogue.com/
- **From Data to Viz:** https://www.data-to-viz.com/

### Tools
- **R + ggplot2:** https://ggplot2.tidyverse.org/
- **Python + matplotlib/seaborn:** https://seaborn.pydata.org/
- **Observable:** https://observablehq.com/
- **Flourish:** https://flourish.studio/
- **RAWGraphs:** https://rawgraphs.io/

### Datasets
- **Gapminder:** https://www.gapminder.org/data/
- **Our World in Data:** https://ourworldindata.org/
- **Tidy Tuesday:** https://github.com/rfordatascience/tidytuesday

## Videos

### TED Talks
- **Hans Rosling:** "The Best Stats You've Ever Seen" (2006)
- **David McCandless:** "The Beauty of Data Visualization" (2010)

### YouTube Channels
- **3Blue1Brown** - Mathematical visualization
- **Vox** - Explanatory journalism with data viz
- **StatQuest** - Statistics education

### Documentaries
- **"The Joy of Stats"** (BBC, Hans Rosling)

---

# References

## Data Visualization History
- Friendly, M. (2008). A brief history of data visualization. *Handbook of data visualization*, 15-56.
- Tufte, E. R. (1983). *The visual display of quantitative information*. Graphics Press.
- Bertin, J. (1967). *Sémiologie graphique*. Gauthier-Villars.

## W.E.B. Du Bois
- Battle-Baptiste, W., & Rusert, B. (Eds.). (2018). *W.E.B. Du Bois's data portraits: Visualizing Black America*. Princeton Architectural Press.
- Du Bois, W. E. B. (1900). The exhibit of American Negroes. *Paris Exposition*.

## Visual Perception
- Anscombe, F. J. (1973). Graphs in statistical analysis. *The American Statistician*, 27(1), 17-21.
- Ware, C. (2012). *Information visualization: Perception for design*. Morgan Kaufmann.

## Modern Visualization
- Wickham, H. (2010). A layered grammar of graphics. *Journal of Computational and Graphical Statistics*, 19(1), 3-28.
- Wilkinson, L. (2005). *The grammar of graphics*. Springer.

---

*Document compiled for SCDS DATA 201: Introduction to Data Science I*
*Module 2: Data Structures and Data Visualization*
*"Telling Stories with Data"*
