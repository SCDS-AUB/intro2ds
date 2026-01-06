#!/usr/bin/env python3
"""
Find potential collaborators at AUB for each DATA 201/202 module
based on publication relevance.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# Module definitions with search keywords (ordered by priority)
MODULES = {
    "Module 1: Data Science & Systems Thinking": {
        "keywords": ["data science", "big data", "data analytics", "information systems",
                     "data collection", "database", "data management"],
        "departments": []
    },
    "Module 2: Data Visualization": {
        "keywords": ["visualization", "visual", "graphics", "mapping", "GIS", "geographic",
                     "network visualization", "infographic", "chart", "plot"],
        "departments": []
    },
    "Module 3: Statistical Thinking": {
        "keywords": ["statistics", "statistical", "probability", "inference", "correlation",
                     "regression", "hypothesis", "bayesian", "sampling", "confidence interval"],
        "departments": []
    },
    "Module 4: Optimization & ML Intro": {
        "keywords": ["optimization", "linear programming", "gradient", "machine learning",
                     "model fitting", "objective function", "constraint", "minimize", "maximize"],
        "departments": []
    },
    "Module 5: Clustering & Dimensionality Reduction": {
        "keywords": ["clustering", "cluster analysis", "k-means", "PCA", "dimensionality reduction",
                     "classification", "segmentation", "pattern recognition", "unsupervised"],
        "departments": []
    },
    "Module 6: Language (NLP)": {
        "keywords": ["natural language", "NLP", "text mining", "sentiment analysis", "arabic",
                     "language model", "text classification", "tokenization", "word embedding",
                     "transformer", "BERT", "GPT"],
        "departments": []
    },
    "Module 7: Vision (Computer Vision)": {
        "keywords": ["computer vision", "image processing", "image analysis", "object detection",
                     "image classification", "CNN", "convolutional", "visual recognition",
                     "deep learning", "neural network"],
        "departments": []
    },
    "Module 8: Audio & Speech": {
        "keywords": ["audio", "speech", "signal processing", "acoustic", "sound", "voice",
                     "spectral", "fourier", "frequency analysis", "speech recognition"],
        "departments": []
    },
    "Module 9: Time-Series & Forecasting": {
        "keywords": ["time series", "forecasting", "temporal", "prediction", "sensor",
                     "monitoring", "trend analysis", "seasonality", "ARIMA", "stochastic"],
        "departments": []
    },
    "Module 10: Regression & Classification": {
        "keywords": ["regression", "classification", "prediction", "supervised learning",
                     "logistic regression", "decision tree", "random forest", "feature"],
        "departments": []
    },
    "Module 11: Deep Learning": {
        "keywords": ["deep learning", "neural network", "transformer", "attention mechanism",
                     "foundation model", "transfer learning", "fine-tuning"],
        "departments": []
    },
    "Module 12: Ethics & Bias": {
        "keywords": ["ethics", "ethical", "bias", "fairness", "privacy", "responsible AI",
                     "algorithmic bias", "discrimination", "transparency", "accountability"],
        "departments": []
    },
    "Module (202) 2: Big Data & Cloud": {
        "keywords": ["big data", "cloud computing", "distributed", "spark", "hadoop",
                     "scalability", "parallel processing", "database", "SQL"],
        "departments": []
    },
    "Module (202) 4: OCR & Documents": {
        "keywords": ["OCR", "optical character recognition", "document", "handwriting",
                     "text extraction", "archive", "digitization", "manuscript"],
        "departments": []
    },
    "Module (202) 7: Networks & Graphs": {
        "keywords": ["network analysis", "graph", "social network", "citation", "centrality",
                     "community detection", "network science", "connectivity"],
        "departments": []
    },
}

def load_articles(filepath):
    """Load articles from JSON lines file."""
    articles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                articles.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return articles

def filter_recent(articles, min_year=2021):
    """Filter articles from recent years."""
    recent = []
    for article in articles:
        date = article.get('date_issued', '')
        try:
            year = int(date[:4]) if date else 0
            if year >= min_year:
                recent.append(article)
        except (ValueError, TypeError):
            continue
    return recent

def search_relevance(article, keywords):
    """Calculate relevance score based on keyword matches."""
    text = ' '.join([
        article.get('title', ''),
        article.get('abstract', ''),
        article.get('subjects', ''),
        article.get('keywords', '')
    ]).lower()

    score = 0
    matched_keywords = []
    for kw in keywords:
        if kw.lower() in text:
            # Higher score for title matches
            if kw.lower() in article.get('title', '').lower():
                score += 3
            else:
                score += 1
            matched_keywords.append(kw)

    return score, matched_keywords

def parse_authors(authors_str):
    """Parse author string into list of names."""
    if not authors_str:
        return []
    return [a.strip() for a in authors_str.split('||') if a.strip()]

def find_collaborators(articles, modules):
    """Find potential collaborators for each module."""
    results = {}

    for module_name, module_info in modules.items():
        keywords = module_info['keywords']

        # Track authors and departments
        author_scores = defaultdict(lambda: {'score': 0, 'articles': [], 'departments': set()})
        dept_scores = defaultdict(lambda: {'score': 0, 'articles': [], 'authors': set()})

        for article in articles:
            score, matched = search_relevance(article, keywords)
            if score > 0:
                authors = parse_authors(article.get('authors', ''))
                dept = article.get('department', 'Unknown')
                faculty = article.get('faculty', 'Unknown')

                for author in authors:
                    author_scores[author]['score'] += score
                    author_scores[author]['articles'].append({
                        'title': article.get('title', ''),
                        'year': article.get('date_issued', ''),
                        'matched_keywords': matched,
                        'score': score
                    })
                    author_scores[author]['departments'].add(dept)

                dept_scores[dept]['score'] += score
                dept_scores[dept]['articles'].append(article.get('title', ''))
                for author in authors:
                    dept_scores[dept]['authors'].add(author)

        # Sort and get top collaborators
        sorted_authors = sorted(
            author_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:15]  # Top 15 authors per module

        sorted_depts = sorted(
            dept_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:10]  # Top 10 departments per module

        results[module_name] = {
            'top_authors': [
                {
                    'name': name,
                    'relevance_score': data['score'],
                    'num_relevant_articles': len(data['articles']),
                    'departments': list(data['departments']),
                    'sample_articles': data['articles'][:3]  # Top 3 articles
                }
                for name, data in sorted_authors
            ],
            'top_departments': [
                {
                    'department': dept,
                    'relevance_score': data['score'],
                    'num_articles': len(data['articles']),
                    'num_authors': len(data['authors']),
                    'top_authors': list(data['authors'])[:5]
                }
                for dept, data in sorted_depts
            ]
        }

    return results

def generate_markdown_report(results, output_path):
    """Generate a markdown report of collaborators."""
    lines = ["# Potential Collaborators by Module\n"]
    lines.append("*Based on AUB ScholarWorks publications (2021-present)*\n\n")

    for module_name, data in results.items():
        lines.append(f"## {module_name}\n")

        # Top Departments
        lines.append("### Relevant Departments\n")
        for dept in data['top_departments'][:5]:
            lines.append(f"- **{dept['department']}** ({dept['num_articles']} relevant articles)")
            if dept['top_authors']:
                authors_str = ', '.join(dept['top_authors'][:3])
                lines.append(f"  - Key researchers: {authors_str}")
        lines.append("")

        # Top Authors
        lines.append("### Key Researchers\n")
        for author in data['top_authors'][:5]:
            depts = ', '.join(author['departments'])
            lines.append(f"- **{author['name']}** ({depts})")
            lines.append(f"  - {author['num_relevant_articles']} relevant articles, relevance score: {author['relevance_score']}")
            if author['sample_articles']:
                sample = author['sample_articles'][0]
                lines.append(f"  - Sample: \"{sample['title'][:80]}...\" ({sample['year']})")
        lines.append("\n---\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def generate_json_output(results, output_path):
    """Save results as JSON for web use."""
    # Convert sets to lists for JSON serialization
    json_results = {}
    for module, data in results.items():
        json_results[module] = {
            'top_authors': data['top_authors'],
            'top_departments': [
                {
                    'department': d['department'],
                    'relevance_score': d['relevance_score'],
                    'num_articles': d['num_articles'],
                    'num_authors': d['num_authors'],
                    'top_authors': d['top_authors']
                }
                for d in data['top_departments']
            ]
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

def main():
    # Paths
    data_path = Path(__file__).parent.parent.parent / 'aub-network' / 'data' / 'data-analysis' / 'discovered_cluster_data' / 'articles_enriched.json'
    output_dir = Path(__file__).parent.parent

    print(f"Loading articles from {data_path}...")
    articles = load_articles(data_path)
    print(f"Loaded {len(articles)} articles")

    # Filter to recent articles (last 4 years)
    recent_articles = filter_recent(articles, min_year=2021)
    print(f"Filtered to {len(recent_articles)} articles from 2021+")

    print("Finding collaborators for each module...")
    results = find_collaborators(recent_articles, MODULES)

    # Generate outputs
    md_path = output_dir / 'collaborators.md'
    json_path = output_dir / 'data' / 'collaborators.json'

    # Ensure data directory exists
    (output_dir / 'data').mkdir(exist_ok=True)

    generate_markdown_report(results, md_path)
    print(f"Markdown report saved to {md_path}")

    generate_json_output(results, json_path)
    print(f"JSON data saved to {json_path}")

if __name__ == '__main__':
    main()
