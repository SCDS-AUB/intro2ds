#!/usr/bin/env python3
"""
Find potential collaborators at AUB for each DATA 201/202 module
based on publication relevance.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# Departments to deprioritize (medical/clinical - still include but don't dominate)
MEDICAL_DEPTS = {
    'internal medicine', 'surgery', 'pediatrics', 'anesthesiology', 'ophthalmology',
    'obstetrics', 'gynecology', 'dermatology', 'cardiology', 'oncology', 'radiology',
    'pathology', 'emergency medicine', 'family medicine', 'otolaryngology', 'urology',
    'gastroenterology', 'nephrology', 'neurology', 'psychiatry', 'hematology'
}

# Departments to prioritize for technical/computational modules
TECHNICAL_DEPTS = {
    'computer science', 'electrical', 'mechanical engineering', 'industrial engineering',
    'mathematics', 'physics', 'statistics', 'biomedical engineering'
}

# Departments to prioritize for social sciences/humanities
SOCIAL_HUMANITIES_DEPTS = {
    'history', 'philosophy', 'sociology', 'psychology', 'economics', 'political',
    'anthropology', 'archaeology', 'education', 'english', 'arabic', 'design',
    'architecture', 'art', 'communications', 'media', 'public policy', 'business'
}

# Module definitions with search keywords
MODULES = {
    "1": {
        "name": "Introduction to Data Science and Systems Thinking",
        "keywords": ["data science", "big data", "data analytics", "information systems",
                     "data collection", "database", "data management", "digital", "computational"],
        "prefer_depts": ["history", "computer science", "philosophy", "sociology"]
    },
    "2": {
        "name": "Data Structures and Data Visualization",
        "keywords": ["visualization", "visual", "graphics", "mapping", "GIS", "geographic",
                     "network visualization", "infographic", "data representation", "design"],
        "prefer_depts": ["design", "geography", "biology", "computer science", "architecture"]
    },
    "3": {
        "name": "Statistical Thinking",
        "keywords": ["statistics", "statistical", "probability", "inference", "correlation",
                     "regression", "hypothesis", "bayesian", "sampling", "survey", "quantitative"],
        "prefer_depts": ["statistics", "economics", "sociology", "psychology", "business", "epidemiology"]
    },
    "4": {
        "name": "Optimization and Introduction to Machine Learning",
        "keywords": ["optimization", "linear programming", "gradient", "machine learning",
                     "model fitting", "objective function", "constraint", "minimize", "maximize",
                     "operations research", "scheduling"],
        "prefer_depts": ["industrial engineering", "mechanical engineering", "computer science", "business"]
    },
    "5": {
        "name": "Unsupervised Learning (Clustering and Dimensionality Reduction)",
        "keywords": ["clustering", "cluster analysis", "k-means", "PCA", "dimensionality reduction",
                     "segmentation", "pattern recognition", "unsupervised", "grouping", "taxonomy"],
        "prefer_depts": ["computer science", "biology", "sociology", "marketing", "genetics"]
    },
    "6": {
        "name": "Language (NLP)",
        "keywords": ["natural language", "NLP", "text mining", "sentiment analysis", "arabic",
                     "language model", "text classification", "linguistics", "corpus", "translation",
                     "computational linguistics"],
        "prefer_depts": ["computer science", "arabic", "english", "linguistics", "history", "psychology"]
    },
    "7": {
        "name": "Vision (Computer Vision)",
        "keywords": ["computer vision", "image processing", "image analysis", "object detection",
                     "image classification", "CNN", "convolutional", "visual recognition",
                     "deep learning", "neural network", "pattern recognition"],
        "prefer_depts": ["computer science", "electrical", "mechanical engineering", "biology"]
    },
    "8": {
        "name": "Audio and Speech",
        "keywords": ["audio", "speech", "signal processing", "acoustic", "sound", "voice",
                     "spectral", "fourier", "frequency", "speech recognition", "music"],
        "prefer_depts": ["electrical", "computer science", "music", "communications", "biomedical"]
    },
    "9": {
        "name": "Time-Series Data and Forecasting",
        "keywords": ["time series", "forecasting", "temporal", "prediction", "sensor",
                     "monitoring", "trend", "seasonality", "dynamics", "longitudinal"],
        "prefer_depts": ["economics", "physics", "environmental", "mechanical engineering", "agriculture"]
    },
    "10": {
        "name": "Learning Functions: Regression and Classification",
        "keywords": ["regression", "classification", "prediction", "supervised learning",
                     "logistic regression", "decision tree", "feature engineering", "model"],
        "prefer_depts": ["computer science", "economics", "psychology", "sociology", "epidemiology"]
    },
    "11": {
        "name": "Deep Learning for Vision and Language",
        "keywords": ["deep learning", "neural network", "transformer", "attention",
                     "foundation model", "transfer learning", "fine-tuning", "GPT", "BERT"],
        "prefer_depts": ["computer science", "electrical", "design"]
    },
    "12": {
        "name": "Machine Learning in Society (Ethics and Impact)",
        "keywords": ["ethics", "ethical", "bias", "fairness", "privacy", "responsible",
                     "discrimination", "transparency", "accountability", "social impact", "policy"],
        "prefer_depts": ["philosophy", "sociology", "psychology", "public policy", "law", "business"]
    },
    "13": {
        "name": "Project – Application and Integration",
        "keywords": ["project", "application", "integration", "real-world", "case study",
                     "industry", "deployment", "product"],
        "prefer_depts": ["design", "business", "computer science"]
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

def is_medical_dept(dept_str):
    """Check if department string indicates primarily medical affiliation."""
    if not dept_str:
        return False
    dept_lower = dept_str.lower()
    medical_count = sum(1 for m in MEDICAL_DEPTS if m in dept_lower)
    total_depts = len(dept_str.split('||'))
    return medical_count > total_depts * 0.5

def is_preferred_dept(dept_str, prefer_list):
    """Check if department matches preferred departments for module."""
    if not dept_str:
        return False
    dept_lower = dept_str.lower()
    return any(p in dept_lower for p in prefer_list)

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

def get_primary_dept(dept_str):
    """Extract primary department from concatenated string."""
    if not dept_str:
        return "Unknown"
    parts = dept_str.split('||')
    # Return shortest non-empty part (usually most specific)
    valid = [p.strip() for p in parts if p.strip()]
    if valid:
        return min(valid, key=len)
    return "Unknown"

def find_collaborators(articles, modules):
    """Find potential collaborators for each module."""
    results = {}

    for module_id, module_info in modules.items():
        keywords = module_info['keywords']
        prefer_depts = module_info.get('prefer_depts', [])

        # Track authors
        author_data = defaultdict(lambda: {
            'score': 0,
            'articles': [],
            'departments': set(),
            'is_medical': False,
            'is_preferred': False
        })

        for article in articles:
            score, matched = search_relevance(article, keywords)
            if score > 0:
                authors = parse_authors(article.get('authors', ''))
                dept = article.get('department', 'Unknown')
                primary_dept = get_primary_dept(dept)

                for author in authors:
                    author_data[author]['score'] += score
                    author_data[author]['articles'].append({
                        'title': article.get('title', ''),
                        'year': article.get('date_issued', ''),
                        'matched_keywords': matched,
                        'score': score,
                        'department': primary_dept
                    })
                    author_data[author]['departments'].add(primary_dept)

                    if is_medical_dept(dept):
                        author_data[author]['is_medical'] = True
                    if is_preferred_dept(dept, prefer_depts):
                        author_data[author]['is_preferred'] = True

        # Sort with preference for non-medical and preferred departments
        def sort_key(item):
            name, data = item
            # Boost preferred departments, penalize medical-only
            dept_bonus = 10 if data['is_preferred'] else 0
            medical_penalty = -5 if data['is_medical'] and not data['is_preferred'] else 0
            return data['score'] + dept_bonus + medical_penalty

        sorted_authors = sorted(
            author_data.items(),
            key=sort_key,
            reverse=True
        )

        # Select diverse set: max 2 from medical, prioritize others
        selected = []
        medical_count = 0
        for name, data in sorted_authors:
            if len(selected) >= 8:
                break
            if data['is_medical'] and not data['is_preferred']:
                if medical_count >= 2:
                    continue
                medical_count += 1
            selected.append((name, data))

        results[module_id] = {
            'name': module_info['name'],
            'collaborators': [
                {
                    'name': name,
                    'relevance_score': data['score'],
                    'departments': list(data['departments']),
                    'articles': sorted(data['articles'], key=lambda x: x['score'], reverse=True)[:3]
                }
                for name, data in selected
            ]
        }

    return results

def generate_json_output(results, output_path):
    """Save results as JSON for web use."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

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

    # Generate JSON output
    json_path = output_dir / 'data' / 'collaborators.json'
    (output_dir / 'data').mkdir(exist_ok=True)
    generate_json_output(results, json_path)
    print(f"JSON data saved to {json_path}")

if __name__ == '__main__':
    main()
