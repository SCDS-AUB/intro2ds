---
layout: default
title: "DATA 202 Module 7: Networks and Graph Data"
---

# DATA 202 Module 7: Networks and Graph Data

## Introduction

Many real-world systems are best understood as networks: social connections, biological interactions, transportation routes, financial transactions, knowledge structures. Graph data science provides tools to analyze these relational systems—finding communities, identifying influential nodes, predicting missing connections, and propagating information through networks.

This module explores network analysis and graph machine learning, from classical algorithms to modern graph neural networks.

---

## Part 1: Network Fundamentals

### Graphs and Networks

A **graph** G = (V, E) consists of:
- **Vertices (Nodes)** V: Entities (people, proteins, cities)
- **Edges (Links)** E: Relationships (friendships, interactions, routes)

**Types**:
- **Undirected**: Symmetric relationships (friendship)
- **Directed**: Asymmetric (following, citations)
- **Weighted**: Edges have values (distance, strength)
- **Bipartite**: Two node types, edges between types (users and products)
- **Multiplex**: Multiple edge types between same nodes

### Network Representations

**Adjacency Matrix**: N×N matrix A where A[i,j] = 1 if edge exists
- Dense storage, fast operations
- O(N²) space

**Edge List**: List of (source, target) pairs
- Sparse storage
- Efficient for large networks

**Adjacency List**: For each node, list of neighbors
- Efficient neighbor access
- Good balance

### NetworkX Basics

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)])

# Basic properties
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Degree of node 3: {G.degree(3)}")

# Visualization
nx.draw(G, with_labels=True, node_color='lightblue')
plt.show()
```

---

## Part 2: Network Analysis

### Centrality Measures

**Degree Centrality**: Number of connections
```python
nx.degree_centrality(G)
```

**Betweenness Centrality**: How often node lies on shortest paths
```python
nx.betweenness_centrality(G)
```

**Closeness Centrality**: Average distance to all other nodes
```python
nx.closeness_centrality(G)
```

**PageRank**: Importance based on importance of neighbors
```python
nx.pagerank(G)
```

**Eigenvector Centrality**: Connected to well-connected nodes
```python
nx.eigenvector_centrality(G)
```

### Community Detection

Finding densely connected groups:

**Louvain Algorithm**: Optimize modularity greedily
```python
from community import community_louvain
partition = community_louvain.best_partition(G)
```

**Label Propagation**: Iterative label spreading
```python
communities = nx.community.label_propagation_communities(G)
```

**Girvan-Newman**: Edge betweenness removal
```python
communities = nx.community.girvan_newman(G)
```

### Network Properties

**Clustering Coefficient**: Probability neighbors are connected
**Path Length**: Average distance between nodes
**Diameter**: Maximum shortest path
**Density**: Fraction of possible edges that exist
**Assortativity**: Tendency of similar nodes to connect

---

## Part 3: Graph Machine Learning

### Node Embeddings

Learn vector representations of nodes:

**DeepWalk**: Random walks + Word2Vec
1. Generate random walks from each node
2. Treat walks as sentences
3. Apply Skip-gram to learn embeddings

**Node2Vec**: Biased random walks
- Parameters control exploration vs. exploitation
- Captures structural roles and communities

```python
from node2vec import Node2Vec

node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200)
model = node2vec.fit(window=10, min_count=1)
embeddings = {node: model.wv[str(node)] for node in G.nodes()}
```

### Link Prediction

Predict missing or future edges:

**Classical Features**:
- Common Neighbors
- Jaccard Coefficient
- Adamic-Adar Index
- Preferential Attachment

```python
preds = nx.adamic_adar_index(G, [(1, 4), (2, 5)])
for u, v, score in preds:
    print(f"Edge ({u}, {v}): {score:.3f}")
```

### Graph Neural Networks

**Message Passing**: Nodes aggregate information from neighbors

**GCN (Graph Convolutional Network)**:
$$H^{(l+1)} = \sigma(\tilde{A} H^{(l)} W^{(l)})$$

**GraphSAGE**: Sample and aggregate
**GAT (Graph Attention Network)**: Attention over neighbors
**GIN (Graph Isomorphism Network)**: More expressive aggregation

---

## Part 4: Applications

### Social Network Analysis

- Identify influencers
- Detect communities
- Track information spread
- Recommend connections

### Biological Networks

- Protein-protein interactions
- Gene regulatory networks
- Drug-target interactions
- Predict drug side effects

### Knowledge Graphs

- Entities and relationships
- Question answering
- Recommendation
- Examples: Wikidata, DBpedia, Google Knowledge Graph

### Financial Networks

- Transaction networks for fraud detection
- Interbank lending for systemic risk
- Supply chain analysis

---

## DEEP DIVE: The PageRank Revolution

### How Google Ranked the Web

In 1996, the web was chaotic. Search engines ranked pages by keyword matching—easily gamed with invisible text and keyword stuffing. Finding quality information was a nightmare.

Two Stanford PhD students, **Larry Page** and **Sergey Brin**, had a different idea: use the structure of the web itself. Their insight was simple yet profound: a page is important if important pages link to it.

### The Algorithm

**PageRank** treats the web as a graph where pages are nodes and hyperlinks are directed edges. The importance of a page is defined recursively:

$$PR(p) = \frac{1-d}{N} + d \sum_{q \in B_p} \frac{PR(q)}{L(q)}$$

Where:
- d is a damping factor (~0.85)
- B_p is the set of pages linking to p
- L(q) is the number of outgoing links from q

Interpretation: Imagine a random surfer following links. PageRank is the probability they land on each page in the long run.

### Why It Worked

PageRank was:
- **Democratic**: Every link is a vote
- **Weighted**: Votes from important pages matter more
- **Robust**: Hard to game without getting many important sites to link to you
- **Scalable**: Computed efficiently via iteration

### Beyond Search

PageRank principles now appear everywhere:
- Twitter's "Who to Follow" suggestions
- Academic citation analysis
- Fraud detection
- Biological network analysis

The insight that network structure encodes importance transformed not just search but our understanding of complex systems.

---

## HANDS-ON EXERCISE: Social Network Analysis

### Part 1: Load and Explore Network

```python
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Load a social network (Zachary's Karate Club)
G = nx.karate_club_graph()

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.4f}")
print(f"Clustering: {nx.average_clustering(G):.4f}")

# Degree distribution
degrees = [d for n, d in G.degree()]
plt.hist(degrees, bins=range(max(degrees)+2))
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Degree Distribution')
plt.show()
```

### Part 2: Centrality Analysis

```python
# Compute centralities
centralities = pd.DataFrame({
    'degree': nx.degree_centrality(G),
    'betweenness': nx.betweenness_centrality(G),
    'closeness': nx.closeness_centrality(G),
    'eigenvector': nx.eigenvector_centrality(G),
    'pagerank': nx.pagerank(G)
})

print("Top 5 nodes by PageRank:")
print(centralities.nlargest(5, 'pagerank'))

# Visualize with size by centrality
pos = nx.spring_layout(G, seed=42)
node_size = [3000 * centralities.loc[n, 'pagerank'] for n in G.nodes()]
nx.draw(G, pos, node_size=node_size, with_labels=True)
plt.title('Node size = PageRank')
plt.show()
```

### Part 3: Community Detection

```python
from community import community_louvain

# Detect communities
partition = community_louvain.best_partition(G)

# Visualize
colors = [partition[n] for n in G.nodes()]
nx.draw(G, pos, node_color=colors, cmap=plt.cm.Set1, with_labels=True)
plt.title('Communities (Louvain)')
plt.show()

# Compare to ground truth (club split)
club_labels = [G.nodes[n]['club'] for n in G.nodes()]
print("Ground truth vs detected:")
for n in G.nodes():
    print(f"Node {n}: {club_labels[n-1]} -> Community {partition[n]}")
```

---

## Recommended Resources

### Libraries
- **NetworkX**: Pure Python, extensive algorithms
- **igraph**: Fast C library with Python bindings
- **PyTorch Geometric**: Graph neural networks
- **DGL**: Deep graph library
- **Gephi**: Graph visualization software

### Books
- *Networks, Crowds, and Markets* by Easley and Kleinberg (free online)
- *Network Science* by Barabási (free online)
- *Graph Representation Learning* by Hamilton

### Datasets
- Stanford Network Analysis Project (SNAP)
- Konect
- Network Repository

---

*Module 7 explores networks and graph data—from classical network analysis to modern graph neural networks. Understanding the structure of relationships unlocks insights impossible to gain from tabular data alone.*
