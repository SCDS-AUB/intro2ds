---
layout: default
title: "Module 12: Machine Learning in Society - Ethics and Responsibility"
---

# Module 12: Machine Learning in Society - Ethics and Responsibility

## Introduction

In May 2016, a team of journalists at ProPublica published an investigation that would reshape how the world thinks about algorithms. They analyzed a risk assessment tool called COMPAS, used in courtrooms across America to predict whether defendants would commit future crimes. Their finding: the algorithm was biased against Black defendants, labeling them as high risk at nearly twice the rate of white defendants—even when controlling for prior crimes.

The COMPAS controversy opened a floodgate of questions that data scientists are still grappling with: Can algorithms be fair? Who is harmed when they're not? What responsibility do we bear for the systems we build? How do we balance efficiency with equity, innovation with precaution?

This module confronts these questions directly. Machine learning is not just a technical discipline—it's a force reshaping society, from who gets hired to who gets medical care to who gets surveilled. Data scientists are not neutral technicians; we are making choices, often hidden in code, that affect millions of lives.

---

## Part 1: When Algorithms Discriminate

### The Faces of Algorithmic Bias

Bias in machine learning takes many forms:

**Historical Bias**: When training data reflects past discrimination. If a hiring algorithm learns from historical hires at a company that discriminated against women, it will learn to discriminate against women.

**Representation Bias**: When certain groups are underrepresented in training data. Facial recognition systems trained primarily on white faces perform poorly on Black faces.

**Measurement Bias**: When the proxy we measure differs from what we care about. Using arrest records as a proxy for crime rates encodes policing patterns, not crime patterns.

**Aggregation Bias**: When a single model is used for groups with different characteristics. A medical algorithm calibrated on average patients may fail for specific populations.

**Evaluation Bias**: When benchmark datasets don't represent real-world populations. ImageNet's categories and images reflect Western perspectives.

### Case Study: Amazon's Hiring Algorithm

In 2018, Reuters revealed that Amazon had developed a machine learning tool to screen job applicants—and then scrapped it when they discovered it was biased against women.

The algorithm was trained on resumes submitted to Amazon over 10 years—a period when the tech industry (and Amazon) was overwhelmingly male. The system learned to penalize resumes containing the word "women's" (as in "women's chess club") and downgraded graduates of all-women's colleges.

Amazon tried to remove the bias by eliminating explicitly gendered terms. It didn't work. The algorithm found other proxies. They eventually abandoned the project entirely.

The lesson: bias isn't a bug you can simply debug. It's baked into data, into features, into the very definition of what we're optimizing for.

### Case Study: Healthcare Allocation

In 2019, a study published in Science revealed that a widely used healthcare algorithm exhibited significant racial bias. The algorithm, used by hospitals to identify patients who need extra care, was much less likely to refer Black patients than equally sick white patients.

The root cause was subtle. The algorithm used healthcare costs as a proxy for healthcare needs. But Black patients in America, facing systemic barriers, historically spent less on healthcare even when equally ill. The algorithm learned that Black patients had lower "need"—when really they had less access.

Replacing the biased label with a better measure of actual health needs reduced the racial disparity by 84%.

---

## Part 2: Fairness - A Moving Target

### What Does "Fair" Mean?

There is no single definition of algorithmic fairness. Researchers have proposed dozens of mathematical definitions, and many are mutually incompatible. Here are the major ones:

**Demographic Parity (Statistical Parity)**: Predictions should be independent of protected group membership. Equal rates of positive predictions across groups.

$$P(\hat{Y}=1 | A=0) = P(\hat{Y}=1 | A=1)$$

**Equalized Odds**: True positive rates and false positive rates should be equal across groups. If the algorithm flags 80% of actual re-offenders in one group, it should flag 80% in all groups.

**Predictive Parity**: Among those predicted positive, equal proportions should actually be positive across groups (equal positive predictive value).

**Individual Fairness**: Similar individuals should receive similar predictions. A person shouldn't be treated worse just because of their group membership.

**Counterfactual Fairness**: A decision is fair if it would have been the same had the individual belonged to a different protected group.

### The Impossibility Theorems

A disturbing mathematical result: when base rates differ between groups (when one group actually has higher rates of the outcome), it is mathematically impossible to satisfy multiple fairness criteria simultaneously.

**Chouldechova's Theorem (2017)**: If a classifier has equal predictive parity across groups, and the base rates differ, then equalized odds cannot hold.

**Kleinberg-Mullainathan-Raghavan (2016)**: Except in trivial cases, it is impossible to simultaneously satisfy calibration, balance for the positive class, and balance for the negative class.

This means fairness involves trade-offs. We cannot have everything. We must choose which definition matters most for a given application—a fundamentally ethical, not technical, choice.

### Fairness Through Awareness vs. Fairness Through Blindness

Two opposing approaches:

**Fairness Through Blindness**: Remove protected attributes (race, gender) from the model. The intuition: if the algorithm doesn't see race, it can't discriminate.

The problem: other features (zip code, name, purchasing patterns) can serve as proxies. Removing race doesn't prevent racial discrimination.

**Fairness Through Awareness**: Explicitly include protected attributes and constrain the model to be fair with respect to them.

The problem: this requires collecting sensitive data, and some jurisdictions prohibit using protected characteristics in decisions.

Neither approach is a complete solution. Fairness requires thoughtful consideration of context, not a mechanical procedure.

---

## Part 3: Privacy in the Age of Data

### The Collapse of Anonymity

Data scientists often promise "anonymity"—removing names and identifiers before analysis. But true anonymization is nearly impossible.

**Netflix Prize (2006)**: Netflix released 100 million movie ratings, anonymized by removing user names. Researchers at UT Austin showed they could re-identify users by cross-referencing with public IMDB ratings.

**AOL Search Data (2006)**: AOL released 20 million search queries from 650,000 users, identified only by numbers. New York Times reporters identified "User 4417749" as a 62-year-old widow in Georgia by analyzing her searches.

**Location Data**: Studies show that just four spatio-temporal points (places and times) are enough to uniquely identify 95% of people in a dataset of 1.5 million.

The lesson: removing obvious identifiers is not enough. Our data is our fingerprint.

### Differential Privacy

**Differential Privacy**, developed by Cynthia Dwork and colleagues, offers a rigorous mathematical guarantee: the output of an analysis should be nearly the same whether or not any single individual is included in the dataset.

The key mechanism: add carefully calibrated random noise to query results. The noise is enough to mask any individual's contribution while preserving overall statistical patterns.

Apple uses differential privacy for usage analytics. The US Census uses it for population data. Google uses it for Chrome usage statistics.

The trade-off: more privacy means more noise, which means less accuracy. Differential privacy formalizes this trade-off mathematically.

### Surveillance and Power

Beyond individual privacy lies systemic surveillance:

**Mass Surveillance**: Government collection of communications, movements, associations
**Corporate Surveillance**: Tech companies tracking online behavior, purchases, locations
**Workplace Surveillance**: Employers monitoring productivity, communications, physical movement

Machine learning amplifies surveillance by making pattern detection automatic. What was once impossible at scale—identifying every face in a crowd, analyzing every phone call—is now trivial.

This creates power asymmetries. Those who control the data and algorithms hold enormous power over those who are observed. Data science can be a tool of liberation or oppression.

---

## Part 4: Accountability and Transparency

### The Black Box Problem

Deep learning models can have billions of parameters. Their decision processes are not easily explained—even by their creators. When a medical AI recommends treatment, or a criminal justice AI recommends detention, who understands why?

This opacity creates problems:
- **Legal**: Many jurisdictions require explanations for consequential decisions
- **Trust**: People reasonably distrust decisions they don't understand
- **Debugging**: How do you fix a biased system you don't understand?
- **Accountability**: Who is responsible when opaque systems fail?

### Explainable AI (XAI)

Researchers have developed techniques to interpret black-box models:

**Feature Importance**: Which input features most influenced the prediction?

**LIME (Local Interpretable Model-agnostic Explanations)**: Approximate the complex model locally with a simple, interpretable model.

**SHAP (SHapley Additive exPlanations)**: Use game theory to attribute prediction contributions to features.

**Attention Visualization**: For transformers, show which parts of the input the model "attends" to.

**Counterfactual Explanations**: What minimal change to the input would change the prediction?

### The Limits of Explanation

But explanations have limits:

- **Faithfulness**: Does the explanation accurately reflect the model's reasoning, or is it a post-hoc rationalization?
- **Completeness**: Can any explanation capture a model with billions of parameters?
- **Comprehension**: Can non-experts understand even "simple" explanations?
- **Gaming**: If explanations are public, can people manipulate inputs to exploit them?

Some researchers argue we should move from "explain this model" to "design models that are inherently interpretable"—even if that means accepting lower accuracy.

---

## Part 5: AI Governance and Regulation

### The Regulatory Landscape

Governments worldwide are developing frameworks for AI governance:

**EU AI Act (2024)**: The first comprehensive AI regulation. Categorizes AI systems by risk level. "Unacceptable risk" (like social scoring) is banned. "High risk" (like hiring, credit) requires conformity assessments, documentation, and human oversight.

**US Executive Order on AI (2023)**: Requires safety testing and transparency for powerful AI systems. Establishes standards for government AI use.

**China's AI Regulations**: Requires algorithmic recommendations to offer opt-outs, mandates security reviews for generative AI.

**GDPR (EU)**: Includes a "right to explanation" for automated decisions (Article 22).

### Professional Ethics

Beyond law, data scientists have professional responsibilities:

**ACM Code of Ethics**: "Avoid harm," "Be honest and trustworthy," "Be fair," "Respect privacy"

**Data Science Oath**: Various proposed oaths modeled on the Hippocratic Oath, emphasizing responsibility and humility

**Organizational Standards**: Companies like Google, Microsoft, and Meta have published AI ethics principles (with varying degrees of enforcement)

### The Challenge of Self-Regulation

Industry self-regulation has a mixed record. Ethics boards have been disbanded (Google's AI ethics board lasted one week). Principles have been ignored when they conflicted with profit. "Ethics washing"—publicly promoting ethics while continuing harmful practices—is common.

This suggests the need for external regulation, independent audits, and meaningful consequences for harm.

---

## Part 6: Societal Impacts

### Labor and Automation

Machine learning enables automation of tasks once thought to require human intelligence:
- Legal document review
- Medical diagnosis
- Customer service
- Content moderation
- Driving vehicles

This creates both opportunities and disruptions:
- Increased productivity and new products
- Job displacement and wage pressure
- Shifts in required skills
- Potential for either greater equality or greater inequality

### Filter Bubbles and Polarization

Recommendation algorithms optimize for engagement—keeping you on the platform. This can create:

**Filter Bubbles**: Showing you content similar to what you've already seen, limiting exposure to diverse perspectives

**Radicalization Pipelines**: Recommendation systems that progressively suggest more extreme content because extreme content drives engagement

**Misinformation Spread**: Algorithms that amplify false but engaging content

### Environmental Impact

Training large AI models consumes enormous energy:
- GPT-3 training: estimated 1,287 MWh, equivalent to ~500 tons of CO2
- Running inference at scale adds continuous energy demand
- Data centers require cooling, often using water

Responsible AI development must consider environmental sustainability.

---

## Part 7: What Data Scientists Can Do

### Building Ethical Practice

**Before Building**:
- Ask: Should this system exist? Who benefits? Who might be harmed?
- Consult affected communities
- Consider alternatives to automated systems

**During Development**:
- Audit training data for bias
- Test on diverse populations
- Document assumptions and limitations
- Include fairness metrics alongside accuracy

**After Deployment**:
- Monitor for disparate impact
- Create channels for feedback and complaints
- Maintain ability to correct or disable systems
- Accept responsibility for failures

### The Importance of Diversity

Homogeneous teams build blind spots into systems. Diversity of background, experience, and perspective helps identify harms that might otherwise be missed.

The data science field has significant diversity problems—particularly in race, gender, and socioeconomic background. Addressing this is both an ethical imperative and a practical necessity for building better systems.

### Resistance and Refusal

Sometimes the ethical choice is not to build. Data scientists have:
- Refused to work on military AI (Google employee protests over Project Maven)
- Leaked information about harmful systems
- Organized collectively for ethical practices

Technical workers have power. Using it responsibly sometimes means saying no.

---

## DEEP DIVE: The COMPAS Controversy - When Algorithms Judge

### The Algorithm in the Courtroom

Eric Loomis was arrested in La Crosse, Wisconsin, in 2013. He had been driving a car used in a drive-by shooting—he wasn't the shooter, but he had a criminal history. At sentencing, the judge consulted a risk assessment score from a proprietary algorithm called COMPAS (Correctional Offender Management Profiling for Alternative Sanctions).

COMPAS, developed by Northpointe (now Equivant), analyzed 137 features about a defendant—criminal history, age, employment, education, housing stability, substance abuse, and more—and produced scores predicting the likelihood of recidivism (committing another crime) and violent recidivism.

Eric Loomis's COMPAS scores were high. The judge sentenced him to six years in prison, explicitly citing the algorithm's assessment.

Loomis appealed, arguing that the use of a secret proprietary algorithm violated his due process rights. In 2016, the Wisconsin Supreme Court ruled against him. Judges could use COMPAS as long as it wasn't the sole factor in sentencing. The algorithm could influence how long someone spent in prison—but neither the defendant nor the court could inspect its workings.

### ProPublica's Investigation

In May 2016, ProPublica published "Machine Bias," an investigation into COMPAS. Their methodology was straightforward:
1. Obtain COMPAS scores for 7,000 defendants in Broward County, Florida
2. Track who actually committed new crimes within two years
3. Analyze errors across racial groups

Their findings were stark:

**Among defendants who did not re-offend**:
- Black defendants: 44.9% were labeled high risk (false positives)
- White defendants: 23.5% were labeled high risk

**Among defendants who did re-offend**:
- Black defendants: 28% were labeled low risk (false negatives)
- White defendants: 47.7% were labeled low risk

In summary: Black defendants who wouldn't re-offend were nearly twice as likely to be wrongly labeled dangerous. White defendants who would re-offend were nearly twice as likely to be wrongly labeled safe.

### Northpointe's Response

Northpointe pushed back, arguing that their algorithm was fair by a different definition: **predictive parity**. Among defendants labeled high risk, similar percentages of Black and white defendants actually re-offended. The algorithm was equally accurate in both groups.

Both claims were true. Both definitions of fairness were valid. But they conflicted—and as the impossibility theorems proved, they could not both be satisfied when base rates differed.

The real question wasn't which definition was "correct." It was which definition mattered more in this context—and who got to decide.

### The Debate Deepens

The COMPAS controversy sparked an explosion of research and debate:

**Base Rates and Fairness**: Black defendants did have higher re-offense rates in the data. But why? Historic discrimination, differential policing, economic inequality—the data reflected societal injustice. Using such data to predict outcomes can perpetuate the very inequalities it encodes.

**What Does "Recidivism" Mean?**: COMPAS predicted rearrest, not actual reoffending. Rearrest rates reflect policing patterns as much as behavior. Black communities are more heavily policed, meaning Black individuals are more likely to be rearrested for equivalent behavior.

**Accuracy vs. Fairness**: COMPAS was reasonably accurate (~65% overall, comparable to expert predictions). But accuracy doesn't guarantee fairness. A system can be accurate on average while causing systematic harm to specific groups.

**Transparency**: COMPAS was proprietary. Defendants couldn't challenge scores based on errors in the algorithm. Courts couldn't audit it. This secrecy undermined due process.

### The Human Alternative

What's the alternative to algorithmic risk assessment? Often, it's judicial intuition—which is also biased, inconsistent, and opaque. Studies have shown that judges are harsher before lunch, influenced by irrelevant information, and subject to racial bias.

The question isn't "algorithms vs. humans" but rather: How do we design systems—whether algorithmic or human—that are as fair and accurate as possible?

Some jurisdictions have moved to simpler, transparent tools based on just a few objective factors. Others have abandoned risk assessment entirely for certain decisions. There is no consensus.

### Lessons for Data Scientists

The COMPAS story teaches critical lessons:

1. **Fairness is not a technical property**: Different definitions of fairness encode different values. Choosing among them is an ethical and political choice, not a mathematical one.

2. **Data encodes history**: When historical data reflects discrimination, models trained on that data will learn discrimination. You cannot simply "remove bias" after the fact.

3. **Context matters**: A risk score that might be acceptable for allocating social services could be unacceptable for determining prison sentences. The stakes matter.

4. **Transparency enables accountability**: When algorithms are secret, they cannot be meaningfully challenged. Due process requires the ability to contest decisions.

5. **Deployment is not the end**: Systems must be continuously monitored for disparate impact, not just validated once at development.

6. **Affected communities must have voice**: Decisions about algorithmic fairness cannot be made only by developers. Those affected—defendants, communities—must have a say.

---

## LECTURE PLAN: Ethics in Machine Learning - Power, Bias, and Responsibility

### Learning Objectives
By the end of this lecture, students will be able to:
1. Identify sources and types of algorithmic bias
2. Explain multiple definitions of fairness and their trade-offs
3. Analyze real-world case studies of algorithmic harm
4. Apply ethical frameworks to data science decisions
5. Propose interventions to make systems more fair

### Lecture Structure (90 minutes)

#### Opening Hook (10 minutes)
**The Sentencing Algorithm**
- Present the Eric Loomis case
- Show the COMPAS controversy in headlines
- Ask: "Should an algorithm influence how long someone goes to prison?"
- Poll: Initial reactions

#### Part 1: Algorithmic Bias (18 minutes)

**What is Bias? (5 minutes)**
- Technical definition vs. social definition
- Historical, representation, measurement bias
- Key insight: bias isn't just prejudice—it's systematic error

**Case Studies of Harm (8 minutes)**
- Amazon hiring: trained on biased history
- Healthcare allocation: wrong proxy variable
- Facial recognition: performance disparities
- Credit scoring: protected classes and proxies

**Where Bias Comes From (5 minutes)**
- Training data reflecting historical discrimination
- Proxy variables and redlining
- Optimization objectives that ignore fairness
- Homogeneous development teams

#### Part 2: Defining Fairness (20 minutes)

**The Challenge of Definition (5 minutes)**
- Ask students: "What would make an algorithm fair?"
- Collect intuitions, show they conflict
- There is no single definition of fairness

**Major Fairness Definitions (10 minutes)**
- Demographic parity: equal rates across groups
- Equalized odds: equal error rates
- Predictive parity: equal accuracy among positives
- Individual fairness: similar treatment for similar individuals
- Mathematical formulation of each

**The Impossibility Result (5 minutes)**
- Present Chouldechova's theorem
- When base rates differ, definitions conflict
- This means fairness requires choices, not just calculation
- Discuss implications: who decides?

#### Part 3: The COMPAS Deep Dive (15 minutes)

**ProPublica's Investigation (7 minutes)**
- Methodology: obtain scores, track outcomes, analyze by race
- Findings: disparate false positive and false negative rates
- Visualize the data

**Northpointe's Response (5 minutes)**
- Their fairness definition: predictive parity
- Why both claims were true
- The impossibility theorem in practice

**Lessons (3 minutes)**
- Context and stakes matter
- Transparency enables accountability
- Affected communities must have voice

#### Part 4: Privacy and Power (12 minutes)

**The End of Anonymity (4 minutes)**
- Netflix, AOL, location data re-identification
- Your data is your fingerprint
- Aggregation enables identification

**Differential Privacy (4 minutes)**
- The mathematical guarantee
- Adding noise to protect individuals
- Trade-off: privacy vs. accuracy

**Surveillance and Power Asymmetries (4 minutes)**
- Mass surveillance capabilities
- Corporate data collection
- Who watches whom?

#### Part 5: What Can We Do? (10 minutes)

**Before, During, After (5 minutes)**
- Before: Should this exist? Who is affected?
- During: Audit data, test on diverse groups, document
- After: Monitor, accept responsibility, enable feedback

**Structural Changes (3 minutes)**
- Diversity in development teams
- External audits and regulation
- Affected community involvement

**The Power of Refusal (2 minutes)**
- Sometimes the answer is no
- Examples of tech worker resistance

#### Wrap-Up (5 minutes)
- Return to opening: What have we learned?
- Fairness requires choices, not just code
- Data scientists have responsibility
- The technical is political
- Discussion questions for reflection

### Materials Needed
- COMPAS data visualizations
- Case study slides with images
- Interactive fairness demo (showing trade-offs)
- Headlines and news clips

### Discussion Questions
1. If base rates differ between groups, which type of fairness should we prioritize? Who should decide?
2. Should companies be required to publish algorithmic impact assessments?
3. Is there a meaningful difference between human bias and algorithmic bias?
4. What would need to change for you to trust an algorithm with sentencing decisions?

---

## HANDS-ON EXERCISE: Auditing an Algorithm for Fairness

### Overview
In this exercise, students will:
1. Analyze a dataset for potential sources of bias
2. Train a classification model
3. Audit the model for disparate impact across groups
4. Explore fairness-accuracy trade-offs
5. Propose and test interventions

### Prerequisites
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, fairlearn

### Setup

```python
# Install required packages
# pip install pandas numpy scikit-learn matplotlib seaborn fairlearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
```

### Part 1: Loading and Exploring the Data (15 minutes)

```python
# We'll use the Adult Income dataset (predicting >$50K income)
# This dataset has known fairness issues

from sklearn.datasets import fetch_openml

# Load dataset
adult = fetch_openml(name='adult', version=2, as_frame=True)
df = adult.frame

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nTarget distribution:")
print(df['income'].value_counts(normalize=True))
```

**Task 1.1**: Examine demographic distributions

```python
# Examine key demographic features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Race distribution
sns.countplot(data=df, x='race', ax=axes[0, 0])
axes[0, 0].set_title('Race Distribution')
axes[0, 0].tick_params(axis='x', rotation=45)

# Sex distribution
sns.countplot(data=df, x='sex', ax=axes[0, 1])
axes[0, 1].set_title('Sex Distribution')

# Income by race
pd.crosstab(df['race'], df['income'], normalize='index').plot(
    kind='bar', ax=axes[1, 0]
)
axes[1, 0].set_title('Income by Race')
axes[1, 0].tick_params(axis='x', rotation=45)

# Income by sex
pd.crosstab(df['sex'], df['income'], normalize='index').plot(
    kind='bar', ax=axes[1, 1]
)
axes[1, 1].set_title('Income by Sex')

plt.tight_layout()
plt.show()

# Calculate base rates
print("\nBase rate (>$50K) by group:")
print(df.groupby('sex')['income'].apply(lambda x: (x == '>50K').mean()))
print(df.groupby('race')['income'].apply(lambda x: (x == '>50K').mean()))
```

### Part 2: Data Preparation (15 minutes)

```python
# Prepare features
# We'll keep sex and race for fairness analysis but explore whether to use in model

# Select features
feature_cols = ['age', 'workclass', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship',
                'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country']

# Create binary target
df['income_binary'] = (df['income'] == '>50K').astype(int)

# Store protected attributes
protected_race = df['race']
protected_sex = df['sex']

# Encode categorical features
df_model = df[feature_cols].copy()
for col in df_model.select_dtypes(include=['object', 'category']).columns:
    df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

X = df_model.values
y = df['income_binary'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Also split protected attributes
_, race_test, _, sex_test = train_test_split(
    protected_race, protected_sex, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)}")
print(f"Test set: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Part 3: Training and Evaluating a Model (15 minutes)

```python
# Train a logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Overall accuracy
print("Overall Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))
```

### Part 4: Fairness Audit (30 minutes)

```python
def fairness_audit(y_true, y_pred, protected_attribute, attribute_name):
    """
    Compute fairness metrics across groups.
    """
    groups = protected_attribute.unique()
    results = []

    for group in groups:
        mask = protected_attribute == group
        if mask.sum() < 10:  # Skip very small groups
            continue

        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]

        # Calculate metrics
        n = mask.sum()
        positive_rate = y_pred_group.mean()  # Selection/prediction rate
        accuracy = accuracy_score(y_true_group, y_pred_group)

        # True positive rate (recall)
        true_positives = ((y_pred_group == 1) & (y_true_group == 1)).sum()
        actual_positives = (y_true_group == 1).sum()
        tpr = true_positives / actual_positives if actual_positives > 0 else 0

        # False positive rate
        false_positives = ((y_pred_group == 1) & (y_true_group == 0)).sum()
        actual_negatives = (y_true_group == 0).sum()
        fpr = false_positives / actual_negatives if actual_negatives > 0 else 0

        results.append({
            'Group': group,
            'N': n,
            'Base Rate': y_true_group.mean(),
            'Positive Pred Rate': positive_rate,
            'Accuracy': accuracy,
            'TPR': tpr,
            'FPR': fpr
        })

    results_df = pd.DataFrame(results)
    print(f"\nFairness Audit for {attribute_name}:")
    print(results_df.round(4).to_string(index=False))

    return results_df


# Audit by sex
sex_audit = fairness_audit(y_test, y_pred, sex_test, "Sex")

# Audit by race
race_audit = fairness_audit(y_test, y_pred, race_test, "Race")
```

**Task 4.1**: Calculate fairness ratios

```python
def calculate_fairness_ratios(audit_df, baseline_group):
    """
    Calculate disparity ratios relative to a baseline group.
    """
    baseline = audit_df[audit_df['Group'] == baseline_group].iloc[0]

    ratios = []
    for _, row in audit_df.iterrows():
        ratios.append({
            'Group': row['Group'],
            'Prediction Rate Ratio': row['Positive Pred Rate'] / baseline['Positive Pred Rate'],
            'TPR Ratio': row['TPR'] / baseline['TPR'],
            'FPR Ratio': row['FPR'] / baseline['FPR']
        })

    ratios_df = pd.DataFrame(ratios)
    print("\nFairness Ratios (1.0 = parity):")
    print(ratios_df.round(4).to_string(index=False))

    # 80% rule (common legal threshold)
    print("\n80% Rule Assessment (ratio should be >= 0.8):")
    for _, row in ratios_df.iterrows():
        pred_ratio = row['Prediction Rate Ratio']
        status = "PASS" if pred_ratio >= 0.8 else "FAIL"
        print(f"  {row['Group']}: {pred_ratio:.4f} ({status})")

    return ratios_df


# Calculate ratios
sex_ratios = calculate_fairness_ratios(sex_audit, 'Male')
race_ratios = calculate_fairness_ratios(race_audit, 'White')
```

**Task 4.2**: Visualize disparities

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Prediction rates by sex
ax = axes[0, 0]
sex_audit.plot(x='Group', y='Positive Pred Rate', kind='bar', ax=ax, legend=False)
ax.axhline(y=sex_audit['Positive Pred Rate'].mean(), color='red', linestyle='--',
           label='Average')
ax.set_title('Positive Prediction Rate by Sex')
ax.set_ylabel('Rate')
ax.tick_params(axis='x', rotation=0)

# TPR and FPR by sex
ax = axes[0, 1]
x = np.arange(len(sex_audit))
width = 0.35
ax.bar(x - width/2, sex_audit['TPR'], width, label='TPR')
ax.bar(x + width/2, sex_audit['FPR'], width, label='FPR')
ax.set_xticks(x)
ax.set_xticklabels(sex_audit['Group'])
ax.legend()
ax.set_title('TPR and FPR by Sex')

# Prediction rates by race
ax = axes[1, 0]
race_audit.plot(x='Group', y='Positive Pred Rate', kind='bar', ax=ax, legend=False)
ax.set_title('Positive Prediction Rate by Race')
ax.set_ylabel('Rate')
ax.tick_params(axis='x', rotation=45)

# TPR and FPR by race
ax = axes[1, 1]
x = np.arange(len(race_audit))
ax.bar(x - width/2, race_audit['TPR'], width, label='TPR')
ax.bar(x + width/2, race_audit['FPR'], width, label='FPR')
ax.set_xticks(x)
ax.set_xticklabels(race_audit['Group'], rotation=45)
ax.legend()
ax.set_title('TPR and FPR by Race')

plt.tight_layout()
plt.show()
```

### Part 5: Fairness Interventions (20 minutes)

```python
# Using Fairlearn library for fairness-aware learning
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import MetricFrame, selection_rate

# Fit a fairness-constrained model
# This uses demographic parity constraint
constraint = DemographicParity()
mitigator = ExponentiatedGradient(
    LogisticRegression(max_iter=1000),
    constraints=constraint
)

# Need to provide protected attribute during training
mitigator.fit(X_train_scaled, y_train,
              sensitive_features=sex_test.iloc[:len(X_train_scaled)].reset_index(drop=True))

# Predictions from fair model
y_pred_fair = mitigator.predict(X_test_scaled)

# Compare
print("Original Model:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  Male selection rate: {y_pred[sex_test.reset_index(drop=True) == 'Male'].mean():.4f}")
print(f"  Female selection rate: {y_pred[sex_test.reset_index(drop=True) == 'Female'].mean():.4f}")

print("\nFair Model (Demographic Parity):")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_fair):.4f}")
print(f"  Male selection rate: {y_pred_fair[sex_test.reset_index(drop=True) == 'Male'].mean():.4f}")
print(f"  Female selection rate: {y_pred_fair[sex_test.reset_index(drop=True) == 'Female'].mean():.4f}")
```

**Task 5.1**: Explore the fairness-accuracy trade-off

```python
# Vary constraint strength and observe trade-off
# This is conceptual; actual implementation varies

# Post-processing: adjust thresholds per group
def threshold_adjustment(y_prob, protected, thresholds):
    """
    Apply different thresholds to different groups.
    """
    y_pred_adjusted = np.zeros(len(y_prob))

    for group, threshold in thresholds.items():
        mask = protected == group
        y_pred_adjusted[mask] = (y_prob[mask] >= threshold).astype(int)

    return y_pred_adjusted


# Find thresholds that equalize rates
from scipy.optimize import minimize_scalar

def find_equalizing_threshold(y_prob, protected, target_rate):
    """Find threshold that achieves target selection rate."""
    results = {}

    for group in protected.unique():
        mask = protected == group
        probs = y_prob[mask]

        def diff(t):
            return abs((probs >= t).mean() - target_rate)

        result = minimize_scalar(diff, bounds=(0, 1), method='bounded')
        results[group] = result.x

    return results


# Target the overall selection rate
target = y_pred.mean()
thresholds = find_equalizing_threshold(y_prob, sex_test.reset_index(drop=True), target)
print(f"Thresholds to achieve {target:.4f} selection rate:")
print(thresholds)

y_pred_adjusted = threshold_adjustment(y_prob, sex_test.reset_index(drop=True), thresholds)
print(f"\nAdjusted selection rates:")
for group in ['Male', 'Female']:
    mask = sex_test.reset_index(drop=True) == group
    print(f"  {group}: {y_pred_adjusted[mask].mean():.4f}")
```

### Challenge Questions

1. **Trade-offs**: What was the accuracy cost of achieving demographic parity? Is this trade-off acceptable?

2. **Which Definition?**: If you had to choose between equalizing prediction rates and equalizing error rates, which would you choose for this application? Why?

3. **Feature Decisions**: We excluded sex from the model features but the model still shows gender disparities. Why? What should be done?

4. **Historical Context**: The training data reflects historical inequities. If we train on this data, we perpetuate those inequities. What alternatives exist?

5. **Stakeholder Perspectives**: How might different stakeholders (employers, job applicants, regulators) view these fairness trade-offs differently?

### Expected Outputs

Students should submit:
1. Exploratory analysis showing demographic distributions and base rates
2. Fairness audit of baseline model with disparities quantified
3. Implementation of at least one fairness intervention
4. Comparison of original and fair models on accuracy and fairness metrics
5. Written reflection on trade-offs and appropriate fairness definitions

### Evaluation Rubric

| Criteria | Points |
|----------|--------|
| Data exploration and bias identification | 15 |
| Correct fairness metric calculation | 20 |
| Fairness audit visualization and interpretation | 20 |
| Intervention implementation | 20 |
| Trade-off analysis and reflection | 15 |
| Code quality and documentation | 10 |
| **Total** | **100** |

---

## Recommended Resources

### Books

**Technical**
- *Fairness and Machine Learning* by Barocas, Hardt, Narayanan - The comprehensive textbook (free online)
- *Interpretable Machine Learning* by Christoph Molnar - Free online, excellent coverage
- *The Ethical Algorithm* by Kearns and Roth - Accessible introduction
- *Weapons of Math Destruction* by Cathy O'Neil - Case studies of algorithmic harm

**Philosophical and Social**
- *Algorithms of Oppression* by Safiya Noble - Race and search algorithms
- *Race After Technology* by Ruha Benjamin - How technology perpetuates racism
- *Automating Inequality* by Virginia Eubanks - Algorithms and poverty
- *The Alignment Problem* by Brian Christian - AI safety and values

### Academic Papers

- **Angwin et al. (2016)**. "Machine Bias" - ProPublica's COMPAS investigation
- **Chouldechova (2017)**. "Fair Prediction with Disparate Impact" - The impossibility theorem
- **Kleinberg et al. (2017)**. "Inherent Trade-Offs in the Fair Determination of Risk Scores"
- **Buolamwini & Gebru (2018)**. "Gender Shades" - Facial recognition disparities
- **Obermeyer et al. (2019)**. "Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations"
- **Bender et al. (2021)**. "On the Dangers of Stochastic Parrots" - LLM risks

### Video Lectures and Talks

- **Joy Buolamwini: "How I'm fighting bias in algorithms"** - TED Talk
- **Cathy O'Neil: "The era of blind faith in big data must end"** - TED Talk
- **Kate Crawford: "The Trouble with Bias"** - NeurIPS keynote
- **Timnit Gebru**: Various talks on AI ethics and racial justice
- **Arvind Narayanan**: "21 Fairness Definitions" - Tutorial

### Tools and Libraries

- **Fairlearn** (https://fairlearn.org/) - Microsoft's fairness toolkit
- **AI Fairness 360** (https://aif360.mybluemix.net/) - IBM's comprehensive toolkit
- **What-If Tool** (https://pair-code.github.io/what-if-tool/) - Google's exploration tool
- **SHAP** (https://shap.readthedocs.io/) - Model explanations
- **Aequitas** (http://aequitas.dssg.io/) - Bias audit toolkit

### Organizations and Resources

- **Partnership on AI** (https://partnershiponai.org/)
- **AI Now Institute** (https://ainowinstitute.org/)
- **Algorithmic Justice League** (https://www.ajl.org/)
- **Data & Society** (https://datasociety.net/)
- **ACM FAccT Conference** - Academic fairness, accountability, transparency

---

## References

1. Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). "Machine Bias." *ProPublica*.

2. Chouldechova, A. (2017). "Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments." *Big Data*, 5(2), 153-163.

3. Kleinberg, J., Mullainathan, S., & Raghavan, M. (2017). "Inherent Trade-Offs in the Fair Determination of Risk Scores." *Proceedings of ITCS*.

4. Obermeyer, Z., et al. (2019). "Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations." *Science*, 366(6464), 447-453.

5. Buolamwini, J., & Gebru, T. (2018). "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification." *Conference on Fairness, Accountability and Transparency*.

6. Dwork, C., et al. (2012). "Fairness Through Awareness." *Proceedings of ITCS*.

7. Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning*. fairmlbook.org.

8. O'Neil, C. (2016). *Weapons of Math Destruction*. Crown Publishing.

9. Benjamin, R. (2019). *Race After Technology*. Polity Press.

10. Noble, S.U. (2018). *Algorithms of Oppression*. NYU Press.

11. Eubanks, V. (2018). *Automating Inequality*. St. Martin's Press.

12. Dwork, C., & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy." *Foundations and Trends in Theoretical Computer Science*, 9(3-4).

---

*Module 12 confronts the ethical dimensions of machine learning—the biases, harms, and power dynamics that arise when algorithms make consequential decisions about human lives. Through the COMPAS controversy, we learn that fairness is not a technical property but a contested value, and that data scientists bear responsibility for the systems they build.*
