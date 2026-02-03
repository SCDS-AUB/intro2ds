---
layout: default
title: "Module 6: Language"
---

# Module 6: Language
## "Machines that Speak"

*Research Document for DATA 201 Course Development*

---

# Table of Contents

1. [Introduction](#introduction)
2. [Part I: The Dream of Machine Translation](#part-i-the-dream-of-machine-translation)
3. [Part II: The Statistical Turn](#part-ii-the-statistical-turn)
4. [Part III: Word Embeddings - Language as Geometry](#part-iii-word-embeddings---language-as-geometry)
5. [Part IV: The Transformer Revolution](#part-iv-the-transformer-revolution)
6. [Part V: Arabic NLP - Challenges and Opportunities](#part-v-arabic-nlp---challenges-and-opportunities)
7. [DEEP DIVE: Word2Vec and the "King - Man + Woman = Queen" Discovery](#deep-dive-word2vec-and-the-king---man--woman--queen-discovery)
8. [Lecture Plan and Hands-On Exercise](#lecture-plan-and-hands-on-exercise)
9. [Recommended Resources](#recommended-resources)
10. [References](#references)

---

# Introduction

Human language is remarkably complex—full of ambiguity, context, and subtle meaning. This module explores:

- How computers learned to process text
- The journey from rules to statistics to deep learning
- The surprising discovery that words have geometry

**Core Question:** Can machines understand language, or just manipulate symbols?

---

# Part I: The Dream of Machine Translation

## The Georgetown-IBM Experiment (1954)

On January 7, 1954, researchers demonstrated the first public machine translation system at IBM's New York headquarters.

### The System

- Translated 60 Russian sentences into English
- Used 250 words and 6 grammar rules
- Completely rule-based

### Example

**Russian:** "Ми передаем мисли посредством речи"
**Translation:** "We transmit thoughts by means of speech"

### The Hype

The *New York Times* predicted fluent machine translation within "three or five years."

### The Reality: ALPAC Report (1966)

After 10 years and millions of dollars, the Automatic Language Processing Advisory Committee concluded:
- Machine translation was far harder than expected
- Human translators were still much better
- Funding should be cut

The field entered a "winter" lasting decades.

### The Lesson

Rule-based approaches couldn't capture the complexity of natural language. Progress required a fundamentally different approach.

---

## Warren Weaver's Memo (1949)

Warren Weaver, a mathematician at the Rockefeller Foundation, wrote a famous memo proposing machine translation. His insight:

> "When I look at an article in Russian, I say: 'This is really written in English, but it has been coded in some strange symbols. I will now proceed to decode.'"

This framing—translation as decoding—planted seeds for the statistical approach.

---

# Part II: The Statistical Turn

## Claude Shannon's Language Model (1948)

In his information theory paper, Shannon asked: How much information is in English text?

### The Experiment

Shannon estimated the entropy of English by having people predict the next letter in sequences:

- Given nothing: ~4.7 bits/letter
- Given previous letters: ~1.3 bits/letter

English is highly redundant and predictable.

### N-gram Models

Shannon introduced **n-gram models**: predict the next word based on the previous n-1 words.

$$P(w_n | w_1, ..., w_{n-1}) \approx P(w_n | w_{n-2}, w_{n-1})$$

**Trigram example:**
"The cat sat on the ___"

Likely: "mat," "floor," "chair"
Unlikely: "elephant," "quantum," "democracy"

### Impact

N-gram models dominated NLP for decades:
- Speech recognition
- Spelling correction
- Machine translation

---

## IBM Models for Translation (1990s)

Peter Brown and colleagues at IBM developed statistical translation models:

### The Key Insight

Treat translation as finding the most probable target sentence given the source:

$$\hat{e} = \arg\max_e P(e|f) = \arg\max_e P(f|e) \cdot P(e)$$

- P(e): Language model (is this good English?)
- P(f|e): Translation model (does f translate to e?)

### Training

Train on parallel corpora (same texts in multiple languages):
- UN proceedings
- European Parliament debates
- Canadian Hansard

### The Rise of Data

The IBM approach showed: **more data beats better algorithms.** With enough parallel text, statistical models outperformed rule-based systems.

---

## Google Translate (2006)

Google launched statistical machine translation using:
- Massive parallel corpora from the web
- Phrase-based models (translate chunks, not words)
- Huge computing resources

### The Transformation

By 2016, Google switched to neural machine translation (NMT)—a dramatic improvement in quality.

---

# Part III: Word Embeddings - Language as Geometry

## Distributional Semantics

"You shall know a word by the company it keeps." — J.R. Firth (1957)

### The Idea

Words that appear in similar contexts have similar meanings:
- "The ____ barked loudly" → dog, puppy, hound
- "I ate a delicious ____" → apple, meal, sandwich

### Co-occurrence Matrices

Count how often words appear together:

|         | eat | bark | computer |
|---------|-----|------|----------|
| dog     | 5   | 100  | 2        |
| cat     | 8   | 3    | 1        |
| laptop  | 0   | 0    | 200      |

Words with similar rows are semantically similar.

---

## Latent Semantic Analysis (1990)

Deerwester et al. applied SVD to term-document matrices:
- Reduce dimensions
- Similar documents cluster together
- Similar words cluster together

This was an early form of word embeddings, but high-dimensional and computationally expensive.

---

## Word2Vec: The Breakthrough (2013)

Tomas Mikolov at Google published Word2Vec—a neural network that learned dense word vectors.

### Two Architectures

**Skip-gram:** Predict context words from center word
**CBOW:** Predict center word from context words

### The Magic

Word2Vec produced 300-dimensional vectors where:
- Similar words are nearby (dog ≈ cat)
- Relationships are directions (king - man + woman ≈ queen)

### Efficiency

Clever tricks (negative sampling, hierarchical softmax) made training feasible on huge corpora.

### Impact

Word2Vec transformed NLP:
- Pre-trained vectors became standard
- Transfer learning emerged
- Words became mathematical objects

---

# Part IV: The Transformer Revolution

## Attention Is All You Need (2017)

Vaswani et al. at Google introduced the **Transformer** architecture.

### The Problem with RNNs

Recurrent neural networks processed sequences one element at a time:
- Slow (can't parallelize)
- Hard to learn long-range dependencies
- Information "forgets" over distance

### The Solution: Self-Attention

Let every position attend to every other position directly:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Key Innovations

- **Positional encoding:** Since there's no recurrence, positions are encoded explicitly
- **Multi-head attention:** Multiple attention patterns in parallel
- **Layer normalization:** Stable training

### Impact

Transformers became the foundation for:
- BERT (2018)
- GPT series (2018-2023)
- T5, PaLM, LLaMA, Claude

---

## BERT: Bidirectional Understanding (2018)

Google's BERT (Bidirectional Encoder Representations from Transformers):

### Pre-training Tasks

1. **Masked Language Modeling:** Predict masked words from context
   - "The [MASK] sat on the mat" → "cat"

2. **Next Sentence Prediction:** Is sentence B likely to follow sentence A?

### Transfer Learning

Fine-tune pre-trained BERT on downstream tasks:
- Question answering
- Sentiment analysis
- Named entity recognition

### State-of-the-Art

BERT achieved SOTA on 11 NLP benchmarks simultaneously.

---

## GPT: Generative Language Models (2018-2023)

OpenAI's GPT series (Generative Pre-trained Transformer):

### The Approach

- Predict the next token (autoregressive)
- Scale up: more data, more parameters, more compute

### The Scaling Laws

OpenAI discovered: Performance improves predictably with:
- Model size (parameters)
- Dataset size (tokens)
- Compute (FLOPs)

### GPT-3 (2020)

175 billion parameters. Could:
- Write essays
- Answer questions
- Generate code
- Perform few-shot learning

### ChatGPT (2022)

GPT fine-tuned with RLHF (Reinforcement Learning from Human Feedback):
- More helpful
- Less harmful
- Conversational

---

# Part V: Arabic NLP - Challenges and Opportunities

## The Challenges

### Morphological Complexity

Arabic words contain enormous information:
- Root + pattern system
- Prefixes and suffixes encode pronouns, tense, gender
- وكتبتهالهم (wakatabtuhālahum) = "and I wrote it to them"

### Dialectal Variation

- Modern Standard Arabic (MSA) for formal writing
- Regional dialects for speech and social media
- Egyptian, Levantine, Gulf, Maghrebi dialects

### Orthographic Issues

- Short vowels often omitted
- Same spelling, different meanings depending on vowels
- Diacritics rarely used in informal text

### Limited Resources

- Less training data than English
- Fewer annotated datasets
- Arabic Wikipedia: ~1.2M articles vs English: ~6.7M

---

## Opportunities

### AraBERT and CAMeL Tools

Arabic-specific models trained on large Arabic corpora:
- Better morphological handling
- Dialect-aware processing
- Named entity recognition for Arabic

### Social Media Analysis

Arabic is widely used online:
- Twitter/X analysis
- Sentiment analysis
- Detecting misinformation

### Historical Document Processing

Rich Arabic manuscript tradition:
- OCR for historical texts
- Digital humanities applications
- Preserving cultural heritage

### Research at AUB and Lebanon

Active research in:
- Arabic sentiment analysis
- Dialect identification
- Code-switching (Arabic + English/French)
- Arabic speech recognition

---

# DEEP DIVE: Word2Vec and the "King - Man + Woman = Queen" Discovery

## The Paper

In 2013, Tomas Mikolov and colleagues at Google published "Efficient Estimation of Word Representations in Vector Space."

## The Discovery

When they trained Word2Vec on billions of words, something remarkable emerged:

```
vector("King") - vector("Man") + vector("Woman") ≈ vector("Queen")
```

### What Does This Mean?

The vector from "Man" to "King" captures something like "royalty."
Apply that same vector to "Woman," and you get "Queen."

**The relationship is a direction in vector space.**

### More Examples

```
Paris - France + Italy = Rome  (capitals)
Walking - Walk + Swim = Swimming  (verb forms)
Brother - Man + Woman = Sister  (gender relations)
```

### The Geometry of Meaning

Words aren't just points—the directions between them encode semantic relationships:
- Gender: woman - man
- Tense: walked - walk
- Country-capital: France - Paris

## How It Works

### The Training Objective

**Skip-gram:** Given a word, predict surrounding words.

For the sentence "The quick brown fox jumps":
- Input: "brown"
- Target: "The", "quick", "fox", "jumps"

### Why Relationships Emerge

Words that share relationships appear in similar contexts in similar ways:

"The king sat on his throne"
"The queen sat on her throne"

The model learns that "king" and "queen" have similar contexts, differing mainly in gender-related words (his/her).

## Hands-On Demo

```python
import gensim.downloader as api

# Load pre-trained Word2Vec
model = api.load('word2vec-google-news-300')

# The famous analogy
result = model.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)
print(f"King - Man + Woman = {result[0][0]}")  # Queen!

# More analogies
def analogy(a, b, c):
    """a is to b as c is to ???"""
    result = model.most_similar(
        positive=[b, c],
        negative=[a],
        topn=1
    )
    return result[0][0]

print(f"Paris:France :: Rome:{analogy('Paris', 'France', 'Rome')}")
print(f"Man:Woman :: King:{analogy('man', 'woman', 'king')}")
```

## The Limitations

### Biases

Word embeddings capture biases in training data:

```
Man:Computer_Programmer :: Woman:Homemaker
```

This reflects societal biases in the text, not truth. Debiasing embeddings is an active research area.

### Not True Understanding

The model doesn't "know" that queens are female monarchs. It learns statistical patterns that happen to align with semantic relationships.

### Analogy Failures

Many analogies don't work:
```
Doctor - Man + Woman ≠ Nurse (hopefully!)
```

The relationships are approximate and sometimes reflect stereotypes.

## The Impact

### NLP Transformation

Word2Vec showed that:
- Distributional semantics works at scale
- Pre-training on large data transfers to downstream tasks
- Words can be mathematical objects

### Path to Transformers

Word2Vec → ELMo (contextualized embeddings) → BERT → GPT

The progression: from static word vectors to contextualized representations to massive language models.

## The Data Journey

- **Collection:** Billions of words from Google News
- **Understanding:** Neural network learns word relationships
- **Prediction:** Analogies, similarity, clustering—all emerge from the geometry

---

# Lecture Plan and Hands-On Exercise

## Lecture Plan: "Teaching Machines to Read" (75-90 minutes)

### Part 1: The Language Problem (15 min)

**Opening Activity:** Give students ambiguous sentences:
- "Time flies like an arrow; fruit flies like a banana"
- "I saw the man with the telescope"

Why is language hard for computers?
- Ambiguity
- Context dependence
- World knowledge

### Part 2: From Rules to Statistics (20 min)

**Timeline:**
- 1950s: Rule-based systems (Georgetown-IBM)
- 1966: ALPAC Report—rules don't scale
- 1990s: Statistical revolution (IBM models)
- 2006: Google Translate

**Key Insight:** More data beats better rules.

### Part 3: Words as Vectors (25 min)

**The Word2Vec Story:**
- Distributional hypothesis: context = meaning
- Neural network training
- The "King - Man + Woman = Queen" moment

**Live Demo:**
- Load pre-trained embeddings
- Try analogies
- Visualize word clusters

### Part 4: The Transformer Era (15 min)

- Attention mechanism intuition
- BERT: Understanding context
- GPT: Generating text
- The scaling revolution

**Discussion:** What can/can't language models do?

---

## Hands-On Exercise: "Exploring Word Embeddings"

### Objective

Discover semantic relationships through word vector arithmetic.

### Duration

2 hours

### Setup

```python
# Install if needed
# !pip install gensim

import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load pre-trained embeddings (this takes a few minutes)
print("Loading Word2Vec model...")
model = api.load('word2vec-google-news-300')
print("Loaded!")
```

### Task 1: Word Similarity (20 min)

```python
# Find similar words
word = "computer"
similar = model.most_similar(word, topn=10)
print(f"Words similar to '{word}':")
for w, score in similar:
    print(f"  {w}: {score:.3f}")

# Try: "happy", "algorithm", "university", "Lebanon"
```

**Questions:**
1. Are the similar words what you expected?
2. Try a word with multiple meanings (e.g., "bank"). What happens?

### Task 2: Analogies (30 min)

```python
def complete_analogy(a, b, c):
    """a is to b as c is to ???"""
    try:
        result = model.most_similar(
            positive=[b, c],
            negative=[a],
            topn=5
        )
        return result
    except KeyError as e:
        return f"Word not in vocabulary: {e}"

# Classic examples
print("man:woman :: king:?")
print(complete_analogy('man', 'woman', 'king'))

print("\nParis:France :: Tokyo:?")
print(complete_analogy('Paris', 'France', 'Tokyo'))
```

**Create your own analogies:**
- Countries and capitals
- Past and present tense
- Singular and plural
- Professions and tools

Which work? Which fail?

### Task 3: Bias Detection (30 min)

```python
# Examine gender associations
def gender_association(word):
    """How associated is a word with 'man' vs 'woman'?"""
    man_sim = model.similarity(word, 'man')
    woman_sim = model.similarity(word, 'woman')
    return woman_sim - man_sim  # Positive = more female-associated

professions = ['doctor', 'nurse', 'engineer', 'teacher',
               'programmer', 'secretary', 'scientist', 'receptionist']

for prof in professions:
    score = gender_association(prof)
    direction = "female" if score > 0 else "male"
    print(f"{prof}: {score:+.3f} ({direction}-leaning)")
```

**Discussion Questions:**
1. What biases do you observe?
2. Where do these biases come from?
3. How might this affect real applications?

### Task 4: Visualization (30 min)

```python
# Visualize word relationships with PCA
words = ['king', 'queen', 'man', 'woman', 'prince', 'princess',
         'father', 'mother', 'son', 'daughter', 'husband', 'wife']

# Get vectors
vectors = np.array([model[w] for w in words])

# Reduce to 2D
pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(12, 8))
for i, word in enumerate(words):
    plt.scatter(coords[i, 0], coords[i, 1])
    plt.annotate(word, (coords[i, 0]+0.02, coords[i, 1]+0.02), fontsize=12)

# Draw arrows showing relationships
# man→woman should parallel king→queen
plt.arrow(coords[2, 0], coords[2, 1],
          coords[3, 0]-coords[2, 0], coords[3, 1]-coords[2, 1],
          head_width=0.05, color='blue', alpha=0.5)
plt.arrow(coords[0, 0], coords[0, 1],
          coords[1, 0]-coords[0, 0], coords[1, 1]-coords[0, 1],
          head_width=0.05, color='blue', alpha=0.5)

plt.title("Word Embeddings: Gender and Royalty")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```

### Task 5: Sentiment Analysis Preview (20 min)

```python
# Simple sentiment using word similarities
positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing']
negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor']

def simple_sentiment(sentence):
    """Compute sentiment based on word similarities."""
    words = sentence.lower().split()
    scores = []

    for word in words:
        try:
            pos_sim = np.mean([model.similarity(word, p) for p in positive_words])
            neg_sim = np.mean([model.similarity(word, n) for n in negative_words])
            scores.append(pos_sim - neg_sim)
        except KeyError:
            continue  # Skip words not in vocabulary

    return np.mean(scores) if scores else 0

# Test sentences
sentences = [
    "This movie was absolutely wonderful",
    "The food was terrible and disgusting",
    "It was an okay experience nothing special"
]

for sent in sentences:
    score = simple_sentiment(sent)
    sentiment = "Positive" if score > 0 else "Negative"
    print(f"{sentiment} ({score:+.3f}): {sent}")
```

---

# Recommended Resources

## Books

- **Jurafsky, D. & Martin, J.H.** *Speech and Language Processing* (3rd ed., draft online) - The NLP textbook
- **Goldberg, Y.** *Neural Network Methods for Natural Language Processing* (2017)
- **Eisenstein, J.** *Introduction to Natural Language Processing* (2019, free online)

## Online Courses

- **Stanford CS224N:** Natural Language Processing with Deep Learning
- **fast.ai:** Practical NLP
- **Hugging Face Course:** Transformers

## Tools

- **spaCy:** Industrial-strength NLP
- **Hugging Face Transformers:** Pre-trained models
- **NLTK:** Classic NLP toolkit
- **Gensim:** Word embeddings

## Arabic NLP

- **CAMeL Tools:** Arabic NLP toolkit from NYU Abu Dhabi
- **AraBERT:** Arabic BERT models
- **Farasa:** Arabic NLP toolkit

## Videos

- **3Blue1Brown:** Neural networks and attention
- **Yannic Kilcher:** Paper explanations
- **Stanford Online:** CS224N lectures on YouTube

---

# References

## Historical
- Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*.
- Brown, P. F., et al. (1990). A Statistical Approach to Machine Translation. *Computational Linguistics*.

## Word Embeddings
- Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. *arXiv*.
- Mikolov, T., et al. (2013). Distributed Representations of Words and Phrases. *NeurIPS*.

## Transformers and Modern NLP
- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. *arXiv*.
- Brown, T., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.

## Arabic NLP
- Obeid, O., et al. (2020). CAMeL Tools: An Open Source Python Toolkit for Arabic NLP. *LREC*.
- Antoun, W., et al. (2020). AraBERT: Transformer-based Model for Arabic Language Understanding. *arXiv*.

---

*Document compiled for SCDS DATA 201: Introduction to Data Science I*
*Module 6: Language*
*"Machines that Speak"*
