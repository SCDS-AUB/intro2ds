---
layout: default
title: Syllabus
permalink: /syllabus/
---

<style>
.syllabus-intro {
    max-width: 800px;
    margin: 0 auto 2em;
    line-height: 1.7;
}

.module-card {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    margin: 2em 0;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.module-header {
    background: linear-gradient(135deg, var(--primary-color), #3d566e);
    color: white;
    padding: 1.2em 1.5em;
}

.module-number {
    font-size: 0.85em;
    opacity: 0.9;
    margin-bottom: 0.3em;
}

.module-title {
    font-size: 1.4em;
    font-weight: 600;
    margin: 0;
}

.module-title a {
    color: white;
    text-decoration: none;
}

.module-title a:hover {
    text-decoration: underline;
}

.module-storyline {
    font-style: italic;
    opacity: 0.9;
    margin-top: 0.5em;
    font-size: 0.95em;
}

.module-body {
    padding: 1.5em;
}

.module-section {
    margin-bottom: 1.5em;
}

.module-section:last-child {
    margin-bottom: 0;
}

.section-title {
    font-weight: 600;
    color: var(--primary-color);
    font-size: 0.9em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.8em;
    padding-bottom: 0.3em;
    border-bottom: 2px solid var(--accent-color);
}

.topic-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5em;
}

.topic-item {
    background: #f0f4f8;
    padding: 0.4em 0.8em;
    border-radius: 4px;
    font-size: 0.9em;
    color: #444;
}

.computing-box {
    background: #e8f4f8;
    border-left: 3px solid var(--accent-color);
    padding: 0.8em 1em;
    border-radius: 0 4px 4px 0;
    font-size: 0.95em;
}

.module-link {
    display: inline-block;
    margin-top: 1em;
    padding: 0.5em 1em;
    background: var(--primary-color);
    color: white;
    text-decoration: none;
    border-radius: 4px;
    font-size: 0.9em;
}

.module-link:hover {
    background: #3d566e;
}

.course-divider {
    text-align: center;
    margin: 3em 0;
    position: relative;
}

.course-divider::before {
    content: '';
    position: absolute;
    left: 0;
    right: 0;
    top: 50%;
    border-top: 2px solid #e0e0e0;
}

.course-divider h2 {
    background: white;
    display: inline-block;
    padding: 0 1.5em;
    position: relative;
    color: var(--primary-color);
    font-size: 1.6em;
}
</style>

<div class="syllabus-intro">

# Course Syllabus

This curriculum introduces students to the foundations of data science through hands-on projects and real-world applications. Each module connects computing skills with domain expertise through interdisciplinary collaboration.

</div>

<div class="course-divider">
<h2>DATA 201: Introduction to Data Science I</h2>
</div>

<!-- Module 1 -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 1</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data201/module01-data-revolution">Introduction to Data Science and Systems Thinking</a></div>
<div class="module-storyline">"Understanding the Data Revolution"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">History and impact of data collection</li>
<li class="topic-item">Structured vs. unstructured data</li>
<li class="topic-item">Basics of data science</li>
<li class="topic-item">Role of statistics and computing</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Python, Jupyter, Pandas, Basic Visualization</div>
</div>

<a href="{{ site.baseurl }}/modules/data201/module01-data-revolution" class="module-link">View Full Module</a>

</div>
</div>

<!-- Module 2 -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 2</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data201/module02-visualization">Data Structures and Data Visualization</a></div>
<div class="module-storyline">"Telling Stories with Data"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Data from various disciplines</li>
<li class="topic-item">Lists, matrices, vectors, images, video</li>
<li class="topic-item">Visualization techniques</li>
<li class="topic-item">Interactive graphics</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Interactive visualizations with Matplotlib, Seaborn, Plotly</div>
</div>

<a href="{{ site.baseurl }}/modules/data201/module02-visualization" class="module-link">View Full Module</a>

</div>
</div>

<!-- Module 3 -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 3</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data201/module03-statistics">Statistical Thinking</a></div>
<div class="module-storyline">"Probabilistic Thinking"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Real-world statistical examples</li>
<li class="topic-item">Probability theory</li>
<li class="topic-item">Statistical inference</li>
<li class="topic-item">Correlations and causation</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Simulations and statistical analysis in Python (scipy, statsmodels)</div>
</div>

<a href="{{ site.baseurl }}/modules/data201/module03-statistics" class="module-link">View Full Module</a>

</div>
</div>

<!-- Module 4 -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 4</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data201/module04-optimization">Optimization and Introduction to Machine Learning</a></div>
<div class="module-storyline">"Optimization and Model Fitting"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Optimization problems</li>
<li class="topic-item">Linear programming</li>
<li class="topic-item">Gradient descent</li>
<li class="topic-item">Supervised learning basics</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Python implementations of basic optimization (scipy.optimize, sklearn)</div>
</div>

<a href="{{ site.baseurl }}/modules/data201/module04-optimization" class="module-link">View Full Module</a>

</div>
</div>

<!-- Module 5 -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 5</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data201/module05-clustering">Unsupervised Learning</a></div>
<div class="module-storyline">"Discovering Patterns"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Customer segmentation</li>
<li class="topic-item">Clustering methods (K-means, GMM)</li>
<li class="topic-item">Dimensionality reduction (PCA, t-SNE)</li>
<li class="topic-item">Anomaly detection</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Applying clustering techniques using sklearn, visualization with UMAP</div>
</div>

<a href="{{ site.baseurl }}/modules/data201/module05-clustering" class="module-link">View Full Module</a>

</div>
</div>

<!-- Module 6 -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 6</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data201/module06-language">Language</a></div>
<div class="module-storyline">"Machines that Speak"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Text/language datasets</li>
<li class="topic-item">Natural language processing</li>
<li class="topic-item">Tokenization and embeddings</li>
<li class="topic-item">Transformers</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Hands-on NLP with NLTK, spaCy, Hugging Face transformers</div>
</div>

<a href="{{ site.baseurl }}/modules/data201/module06-language" class="module-link">View Full Module</a>

</div>
</div>

<!-- Module 7 -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 7</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data201/module07-vision">Vision</a></div>
<div class="module-storyline">"Machines that See"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Image datasets</li>
<li class="topic-item">Computer vision fundamentals</li>
<li class="topic-item">Neural network basics</li>
<li class="topic-item">CNN architectures</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Deep neural networks with PyTorch, image processing with OpenCV</div>
</div>

<a href="{{ site.baseurl }}/modules/data201/module07-vision" class="module-link">View Full Module</a>

</div>
</div>

<!-- Module 8 -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 8</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data201/module08-audio">Audio</a></div>
<div class="module-storyline">"Machines that Hear"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Raw audio signals</li>
<li class="topic-item">Digital signal processing</li>
<li class="topic-item">Spectral analysis</li>
<li class="topic-item">Speech recognition</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Fast Fourier Transform, librosa, speech synthesis</div>
</div>

<a href="{{ site.baseurl }}/modules/data201/module08-audio" class="module-link">View Full Module</a>

</div>
</div>

<!-- Module 9 -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 9</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data201/module09-timeseries">Time-Series Data and Forecasting</a></div>
<div class="module-storyline">"Measuring things in time"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Sensors and measurement devices</li>
<li class="topic-item">Forecasting methods</li>
<li class="topic-item">Chaos theory and stochastic processes</li>
<li class="topic-item">Trend analysis</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Python-based forecasting (statsmodels, Prophet)</div>
</div>

<a href="{{ site.baseurl }}/modules/data201/module09-timeseries" class="module-link">View Full Module</a>

</div>
</div>

<!-- Module 10 -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 10</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data201/module10-regression">Learning Functions: Regression and Classification</a></div>
<div class="module-storyline">"Learning from examples"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Regression datasets</li>
<li class="topic-item">Classification datasets</li>
<li class="topic-item">Linear and logistic regression</li>
<li class="topic-item">Feature engineering</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Model training and evaluation in Python (sklearn, cross-validation)</div>
</div>

<a href="{{ site.baseurl }}/modules/data201/module10-regression" class="module-link">View Full Module</a>

</div>
</div>

<!-- Module 11 -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 11</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data201/module11-deeplearning">Deep Learning for Vision and Language</a></div>
<div class="module-storyline">"Advanced Neural Networks"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Real-world AUB datasets</li>
<li class="topic-item">Research-driven projects</li>
<li class="topic-item">Integrating ML, visualization, optimization</li>
<li class="topic-item">Team collaboration</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Transfer learning, fine-tuning pre-trained models (PyTorch, Hugging Face)</div>
</div>

<a href="{{ site.baseurl }}/modules/data201/module11-deeplearning" class="module-link">View Full Module</a>

</div>
</div>

<!-- Module 12 -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 12</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data201/module12-ethics">Machine Learning in Society</a></div>
<div class="module-storyline">"Responsible Innovation"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">AI ethics case studies</li>
<li class="topic-item">Ethical frameworks</li>
<li class="topic-item">Bias in ML</li>
<li class="topic-item">Fairness and accountability</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Evaluating models for fairness (Fairlearn, AIF360)</div>
</div>

<a href="{{ site.baseurl }}/modules/data201/module12-ethics" class="module-link">View Full Module</a>

</div>
</div>

<!-- Module 13 -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 13</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data201/module13-project">Project - Application and Integration</a></div>
<div class="module-storyline">"Real-World Impact"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Real-world AUB datasets</li>
<li class="topic-item">Research or product-driven project</li>
<li class="topic-item">Integrating learned techniques</li>
<li class="topic-item">Team work and problem formulation</li>
</ul>
</div>

<a href="{{ site.baseurl }}/modules/data201/module13-project" class="module-link">View Full Module</a>

</div>
</div>

<!-- Module 14 -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 14</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data201/module14-presentations">Project Presentations and Reflections</a></div>
<div class="module-storyline">"Sharing Your Insights"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Final presentations</li>
<li class="topic-item">Peer feedback</li>
<li class="topic-item">Synthesis and reflection</li>
<li class="topic-item">Learning outcomes assessment</li>
</ul>
</div>

<a href="{{ site.baseurl }}/modules/data201/module14-presentations" class="module-link">View Full Module</a>

</div>
</div>

<div class="course-divider">
<h2>DATA 202: Introduction to Data Science II</h2>
</div>

<!-- DATA 202 Modules -->
<div class="module-card">
<div class="module-header">
<div class="module-number">Module 1</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data202/module01-acquisition">Data Acquisition and Cleaning</a></div>
<div class="module-storyline">"Transforming Raw Data into Insights"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Web scraping</li>
<li class="topic-item">API integration</li>
<li class="topic-item">Handling missing data</li>
<li class="topic-item">Handling biased data</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Automated data cleaning (pandas, BeautifulSoup, requests)</div>
</div>

<a href="{{ site.baseurl }}/modules/data202/module01-acquisition" class="module-link">View Full Module</a>

</div>
</div>

<div class="module-card">
<div class="module-header">
<div class="module-number">Module 2</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data202/module02-databases">Large-Scale Databases and Cloud Computing</a></div>
<div class="module-storyline">"Working with Large and Complex Datasets"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Big datasets handling</li>
<li class="topic-item">Data cleaning pipelines</li>
<li class="topic-item">Storage strategies</li>
<li class="topic-item">Real-world scalability</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Distributed processing with Spark, SQL databases</div>
</div>

<a href="{{ site.baseurl }}/modules/data202/module02-databases" class="module-link">View Full Module</a>

</div>
</div>

<div class="module-card">
<div class="module-header">
<div class="module-number">Module 3</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data202/module03-streaming">Real-Time Time-Series Data</a></div>
<div class="module-storyline">"Forecasting and Pattern Discovery in Temporal Data"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Real-time data streams</li>
<li class="topic-item">Streaming architectures</li>
<li class="topic-item">Online learning</li>
<li class="topic-item">Edge computing</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Implementing real-time time-series models, streaming pipelines</div>
</div>

<a href="{{ site.baseurl }}/modules/data202/module03-streaming" class="module-link">View Full Module</a>

</div>
</div>

<div class="module-card">
<div class="module-header">
<div class="module-number">Module 4</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data202/module04-ocr">OCR and Document Processing</a></div>
<div class="module-storyline">"Bringing Text from the Physical to the Digital World"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Scanned documents</li>
<li class="topic-item">Handwritten texts</li>
<li class="topic-item">Historical archives</li>
<li class="topic-item">Image-to-text processing</li>
<li class="topic-item">Deep learning OCR models</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">OCR implementation with Tesseract, deep learning approaches</div>
</div>

<a href="{{ site.baseurl }}/modules/data202/module04-ocr" class="module-link">View Full Module</a>

</div>
</div>

<div class="module-card">
<div class="module-header">
<div class="module-number">Module 5</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data202/module05-music-speech">Music and Speech</a></div>
<div class="module-storyline">"Analyzing and Generating Sound with AI"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Audio signal processing</li>
<li class="topic-item">Speech recognition</li>
<li class="topic-item">Music generation</li>
<li class="topic-item">Voice synthesis</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Speech-to-text, music analysis with librosa, generative audio models</div>
</div>

<a href="{{ site.baseurl }}/modules/data202/module05-music-speech" class="module-link">View Full Module</a>

</div>
</div>

<div class="module-card">
<div class="module-header">
<div class="module-number">Module 6</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data202/module06-video">Videos and Fields</a></div>
<div class="module-storyline">"Processing Spatial and Temporal Visual Data"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Video processing</li>
<li class="topic-item">Spatial field data</li>
<li class="topic-item">3D data representations</li>
<li class="topic-item">Scientific visualization</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Video analysis with OpenCV, 3D data processing</div>
</div>

<a href="{{ site.baseurl }}/modules/data202/module06-video" class="module-link">View Full Module</a>

</div>
</div>

<div class="module-card">
<div class="module-header">
<div class="module-number">Module 7</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data202/module07-networks">Graph Data Science and Network Analysis</a></div>
<div class="module-storyline">"Understanding networks in datapoints"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Social networks</li>
<li class="topic-item">Citation graphs</li>
<li class="topic-item">Biological networks</li>
<li class="topic-item">Centrality measures</li>
<li class="topic-item">Community detection</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">NetworkX, graph algorithms, Dijkstra's algorithm</div>
</div>

<a href="{{ site.baseurl }}/modules/data202/module07-networks" class="module-link">View Full Module</a>

</div>
</div>

<div class="module-card">
<div class="module-header">
<div class="module-number">Module 8</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data202/module08-foundation">Foundation Models and Scaling</a></div>
<div class="module-storyline">"How big data and big models work"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Text datasets (news, medical, legal, scientific)</li>
<li class="topic-item">Language models</li>
<li class="topic-item">N-gram models</li>
<li class="topic-item">Search and retrieval</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Training LLMs (GPT), prompt engineering, RAG</div>
</div>

<a href="{{ site.baseurl }}/modules/data202/module08-foundation" class="module-link">View Full Module</a>

</div>
</div>

<div class="module-card">
<div class="module-header">
<div class="module-number">Module 9</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data202/module09-deployment">GPUs and Model Deployment</a></div>
<div class="module-storyline">"Taking AI from Research to Production"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Large-scale datasets for training</li>
<li class="topic-item">Foundation models</li>
<li class="topic-item">Model optimization</li>
<li class="topic-item">GPU parallelization</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Fine-tuning models with APIs, GPU programming basics, model serving</div>
</div>

<a href="{{ site.baseurl }}/modules/data202/module09-deployment" class="module-link">View Full Module</a>

</div>
</div>

<div class="module-card">
<div class="module-header">
<div class="module-number">Module 10</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data202/module10-ethics">Ethics, Bias, and Fairness in Data Science</a></div>
<div class="module-storyline">"Ensuring Responsible AI Development"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Case studies on biased datasets</li>
<li class="topic-item">Ethical frameworks</li>
<li class="topic-item">Fairness constraints</li>
<li class="topic-item">Interpretability</li>
</ul>
</div>

<div class="module-section">
<div class="section-title">Computing</div>
<div class="computing-box">Evaluating bias in AI models, explainability tools (SHAP, LIME)</div>
</div>

<a href="{{ site.baseurl }}/modules/data202/module10-ethics" class="module-link">View Full Module</a>

</div>
</div>

<div class="module-card">
<div class="module-header">
<div class="module-number">Module 11</div>
<div class="module-title"><a href="{{ site.baseurl }}/modules/data202/module11-capstone">Capstone Project</a></div>
<div class="module-storyline">"Solving a Large-Scale Data Challenge"</div>
</div>
<div class="module-body">

<div class="module-section">
<div class="section-title">Topics</div>
<ul class="topic-list">
<li class="topic-item">Dataset selection (music, healthcare, astronomy, finance)</li>
<li class="topic-item">Complete data processing pipeline</li>
<li class="topic-item">AI pipeline design</li>
<li class="topic-item">Research-style report and presentation</li>
</ul>
</div>

<a href="{{ site.baseurl }}/modules/data202/module11-capstone" class="module-link">View Full Module</a>

</div>
</div>

---

## Discussion & Feedback

Have suggestions for module content or datasets? Leave a comment below.

<script src="https://giscus.app/client.js"
        data-repo="SCDS-AUB/intro2ds"
        data-repo-id="R_kgDOQ0050g"
        data-category="General"
        data-category-id="DIC_kwDOQ0050s4C0pnD"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="preferred_color_scheme"
        data-lang="en"
        crossorigin="anonymous"
        async>
</script>
