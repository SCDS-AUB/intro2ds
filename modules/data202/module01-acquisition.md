---
layout: default
title: "DATA 202 Module 1: Data Acquisition - From Web to Warehouse"
---

# DATA 202 Module 1: Data Acquisition - From Web to Warehouse

## Introduction

The most sophisticated machine learning algorithm is useless without data. And in the real world, data rarely arrives in clean CSV files ready for analysis. It must be hunted, gathered, scraped, requested, negotiated, and often fought for.

This module explores the art and science of data acquisition—the methods and tools for obtaining data from the wild web, from APIs, from documents, and from streams. We'll learn not just the technical skills but also the legal, ethical, and practical considerations that separate responsible data collection from digital trespass.

---

## Part 1: The Data Landscape

### Where Data Lives

Data exists in many forms across the digital landscape:

**Structured APIs**: Services that provide data in organized formats (JSON, XML) through programmatic interfaces. Twitter, weather services, financial data providers.

**Semi-Structured Web Pages**: HTML containing embedded data—product listings, news articles, social media posts—that must be extracted.

**Documents**: PDFs, images of text, scanned forms that require optical character recognition and parsing.

**Databases**: Corporate and government databases, some accessible, many locked behind authentication.

**Real-Time Streams**: Continuous flows of data—stock tickers, sensor readings, social media firehoses.

**Unstructured Media**: Images, audio, video containing information that must be extracted through recognition systems.

### The Data Acquisition Hierarchy

From easiest to most challenging:

1. **Published Datasets**: Already collected, cleaned, documented
2. **API Data**: Structured, documented, but requires coding
3. **Web Scraping**: Requires parsing, more fragile
4. **Document Extraction**: Requires OCR, handling noise
5. **Real-Time Streams**: Requires infrastructure, handling velocity
6. **Original Collection**: Surveys, sensors, experiments

---

## Part 2: Working with APIs

### What is an API?

An **Application Programming Interface (API)** is a contract between software systems. Web APIs allow programs to request data from servers over HTTP, receiving structured responses.

When you visit a weather website, your browser makes requests to their servers and receives HTML to display. When a program uses their API, it makes similar requests but receives structured data (typically JSON) instead of a webpage.

### REST APIs

**REST (Representational State Transfer)** is the dominant architectural style for web APIs:

**Resources**: Things you want to access (users, tweets, products) identified by URLs
- `https://api.example.com/users/123`
- `https://api.example.com/products?category=electronics`

**HTTP Methods**:
- `GET`: Retrieve data
- `POST`: Create new data
- `PUT`: Update existing data
- `DELETE`: Remove data

**Responses**: Typically JSON with status codes
- `200 OK`: Success
- `401 Unauthorized`: Need authentication
- `404 Not Found`: Resource doesn't exist
- `429 Too Many Requests`: Rate limited

### Authentication

Most valuable APIs require authentication:

**API Keys**: Simple tokens passed in headers or query parameters
```python
headers = {'Authorization': 'Bearer YOUR_API_KEY'}
response = requests.get(url, headers=headers)
```

**OAuth**: More complex, used when accessing user data
- Redirect user to service login
- Receive temporary token
- Exchange for access token
- Use access token for requests

### Rate Limiting

APIs protect themselves from overuse:

**Rate limits**: Maximum requests per time period (e.g., 100/hour)

**Handling**:
- Track your usage
- Implement backoff when limited
- Cache responses to avoid repeated requests
- Respect `Retry-After` headers

### Working with JSON

JSON (JavaScript Object Notation) is the lingua franca of APIs:

```python
import requests

response = requests.get('https://api.example.com/data')
data = response.json()  # Parse JSON to Python dict/list

# Navigate the structure
users = data['users']
for user in users:
    print(user['name'], user['email'])
```

### Common APIs for Data Science

**General Data**:
- OpenWeatherMap: Weather data
- Twitter/X API: Social media (limited free tier)
- Reddit API: Discussions and posts
- Wikipedia API: Encyclopedia data

**Financial**:
- Alpha Vantage: Stock prices
- Coinbase: Cryptocurrency
- Yahoo Finance: Market data

**Geographic**:
- OpenStreetMap: Maps and locations
- Google Maps: Geocoding, places
- Mapbox: Mapping services

**Government**:
- data.gov API: US government data
- World Bank API: Development indicators
- Census API: Population data

---

## Part 3: Web Scraping

### When APIs Aren't Available

Many websites don't offer APIs, or their APIs are limited. Web scraping extracts data directly from HTML pages.

### The Ethics and Legality of Scraping

Web scraping exists in legal gray areas:

**Potentially Allowed**:
- Public data for personal use
- Research and journalism
- Data your organization owns
- When explicitly permitted

**Potentially Problematic**:
- Violating Terms of Service
- Overloading servers
- Accessing data behind logins
- Commercial use of scraped data
- Republishing copyrighted content

**Check First**:
- `robots.txt`: Machine-readable crawling rules
- Terms of Service: Legal restrictions
- Rate limits: How much traffic is reasonable

### Tools and Libraries

**Requests**: Make HTTP requests
```python
import requests
html = requests.get('https://example.com').text
```

**Beautiful Soup**: Parse HTML
```python
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')
titles = soup.find_all('h1')
```

**Selenium**: Automate browsers for JavaScript-rendered content
```python
from selenium import webdriver
driver = webdriver.Chrome()
driver.get('https://example.com')
content = driver.page_source
```

**Scrapy**: Full-featured scraping framework for larger projects

### HTML Parsing Strategies

**Find by tag**: `soup.find('div')`, `soup.find_all('a')`

**Find by class**: `soup.find_all('div', class_='product')`

**Find by ID**: `soup.find(id='main-content')`

**CSS selectors**: `soup.select('div.product > span.price')`

**Navigate structure**: `element.parent`, `element.next_sibling`

### Handling Common Challenges

**JavaScript-Rendered Content**: Use Selenium or APIs like Splash

**Pagination**: Follow "next" links or construct page URLs

**Rate Limiting**: Add delays between requests
```python
import time
for url in urls:
    response = requests.get(url)
    time.sleep(2)  # Wait 2 seconds
```

**Changing Structures**: Build robust selectors, add error handling

**CAPTCHAs and Blocks**: Respect them—they're telling you to stop

---

## Part 4: Data Formats and Parsing

### Common Formats

**CSV/TSV**: Comma or tab-separated values
- Simple, widely supported
- No nested structures
- Encoding issues common

**JSON**: JavaScript Object Notation
- Nested structures
- Standard for APIs
- Human-readable

**XML**: eXtensible Markup Language
- Hierarchical
- Schema validation
- Verbose

**Parquet**: Columnar binary format
- Efficient for analytics
- Preserves types
- Not human-readable

**Excel**: Microsoft spreadsheet
- Multiple sheets
- Formatting metadata
- pandas can read directly

### Parsing Strategies

**Pandas for tabular data**:
```python
df = pd.read_csv('data.csv')
df = pd.read_json('data.json')
df = pd.read_excel('data.xlsx')
df = pd.read_parquet('data.parquet')
```

**json module for nested structures**:
```python
import json
with open('data.json') as f:
    data = json.load(f)
```

**xml.etree for XML**:
```python
import xml.etree.ElementTree as ET
tree = ET.parse('data.xml')
root = tree.getroot()
```

### Handling Messy Data

Real-world data is messy:

**Encoding issues**: Specify encoding explicitly
```python
pd.read_csv('data.csv', encoding='utf-8')
```

**Missing values**: Identify and handle
```python
pd.read_csv('data.csv', na_values=['NA', 'N/A', ''])
```

**Inconsistent formats**: Clean as you parse
```python
df['date'] = pd.to_datetime(df['date'], errors='coerce')
```

---

## Part 5: Building Data Pipelines

### From One-Off Script to Reliable Pipeline

One-off data collection is easy. Reliable, repeatable collection is engineering.

### Pipeline Components

**Extraction**: Get data from sources
**Transformation**: Clean and structure
**Loading**: Store in destination
**Monitoring**: Track success/failure
**Scheduling**: Run automatically

### Tools for Pipelines

**Simple scheduling**: cron (Unix), Task Scheduler (Windows)

**Workflow orchestration**:
- Apache Airflow: Full-featured, Python-based
- Prefect: Modern alternative
- Luigi: Simpler predecessor

**Cloud services**:
- AWS Glue, Lambda
- Google Cloud Dataflow
- Azure Data Factory

### Error Handling and Reliability

**Retry logic**: Temporary failures should retry
```python
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def fetch_data(url):
    return requests.get(url)
```

**Logging**: Record what happened
```python
import logging
logging.info(f"Fetched {len(records)} records from {url}")
```

**Alerting**: Know when things break

**Idempotency**: Running twice should give same result

---

## DEEP DIVE: The Rise and Fall of the Twitter API

### The Birth of the API Economy

In 2006, Twitter launched with a simple premise: share 140-character messages with the world. But what made Twitter revolutionary wasn't just the product—it was the decision to open their data through an API.

The Twitter API allowed anyone to build applications that read and wrote tweets. This spawned an ecosystem of clients, analytics tools, and integrations. TweetDeck, the popular client, was built entirely on the API. Researchers could study information spread in real-time. Businesses monitored brand mentions. Developers experimented with new social applications.

This openness was strategic. Twitter was a small startup competing against Facebook. By opening the API, they gained developers as allies, extending Twitter's reach far beyond what they could build alone.

### The Golden Age (2006-2012)

The early Twitter API was remarkably permissive:

**Firehose**: The complete stream of all tweets—billions of messages—was available to partners.

**Search API**: Query historical tweets freely.

**Streaming API**: Real-time filtered streams of tweets.

**High Rate Limits**: Generous quotas for authenticated developers.

Researchers published thousands of papers using Twitter data—studying elections, disasters, public health, language, and behavior at unprecedented scale. The Arab Spring protests of 2011 were analyzed in real-time. COVID-19 misinformation would later be tracked through the platform.

Companies built entire businesses on Twitter data:
- **DataSift**: Sold filtered firehose access
- **Gnip**: Provided historical data (later acquired by Twitter)
- **Social media monitoring tools**: Brandwatch, Sprinklr, Hootsuite

### The Closing Begins (2012-2022)

As Twitter grew and monetized, the open API became a liability:

**2012**: Twitter restricted API access, killing third-party clients that competed with their official app.

**2013-2018**: Gradual tightening—lower rate limits, more restrictive terms, fewer features.

**2020**: Academic Research Product Track launched—providing better access for researchers, but with application requirements.

**2022**: Twitter started charging for API access that was previously free.

The shift reflected a fundamental tension: Twitter's data was valuable, and giving it away undermined their business model. They could charge for data access or use it for their own advertising targeting—not both.

### The Musk Era (2023-Present)

When Elon Musk acquired Twitter in 2022, the API transformation accelerated dramatically:

**Free tier eliminated**: Basic API access that was free now costs hundreds of dollars per month.

**Research access restricted**: The Academic Research track was discontinued.

**Rate limits slashed**: Even paid tiers received far fewer requests.

**Pricing restructured**: The firehose, once available to partners, now costs millions of dollars annually.

For researchers, the change was catastrophic. Decades of methodology built on Twitter data became impossible to replicate. Studies tracking misinformation, public health, and political discourse lost their data source.

### Lessons for Data Scientists

The Twitter API saga teaches essential lessons:

1. **Data access is not guaranteed**: APIs can change or disappear. Don't assume today's access will continue.

2. **Terms of Service matter**: Using an API creates a contractual relationship. Violations have consequences.

3. **Data dependency is risk**: Building on others' data creates dependency. Consider alternatives and backups.

4. **Store what you collect**: If you have legitimate access, archive data you might need later.

5. **The economics drive decisions**: API access is a business decision. Understand the provider's incentives.

6. **Reproducibility challenges**: Research built on proprietary data may not be reproducible.

### The Post-Twitter Landscape

As Twitter closed, alternatives emerged:

**Mastodon/Fediverse**: Decentralized, open protocols
**Bluesky**: Open API from the start
**Reddit API**: Also restricted in 2023
**Common Crawl**: Web archive data
**Original collection**: Building your own datasets

The era of freely available social media data may be ending. Future data science may require more direct data collection, better relationships with data providers, and more attention to data sustainability.

---

## LECTURE PLAN: Data Acquisition in the Wild

### Learning Objectives
By the end of this lecture, students will be able to:
1. Design an API data collection strategy
2. Build a basic web scraper
3. Handle common data formats
4. Understand legal and ethical considerations
5. Plan for reliable data pipelines

### Lecture Structure (90 minutes)

#### Opening Hook (8 minutes)
**The Data Hunt**
- Present a scenario: "Your organization needs social media data about a topic. Where do you start?"
- Show the landscape of options
- Introduce the Twitter API story as cautionary tale
- Frame: "Data acquisition is the foundation—and often the bottleneck"

#### Part 1: The API Approach (20 minutes)

**Understanding APIs (7 minutes)**
- What is an API? Restaurant metaphor (menu, order, kitchen)
- REST principles: Resources, URLs, HTTP methods
- Live demo: Simple API call in Python
- Show: JSON response structure

**Authentication and Limits (7 minutes)**
- API keys: The simplest case
- OAuth: When you need user permissions
- Rate limiting: Why and how to handle
- Demo: authenticated request, handle rate limit

**Practical Strategies (6 minutes)**
- Read documentation first
- Test interactively before scripting
- Handle errors gracefully
- Cache to avoid redundant calls

#### Part 2: Web Scraping (25 minutes)

**When and Whether to Scrape (5 minutes)**
- When APIs aren't available
- Legal considerations: robots.txt, ToS
- Ethical considerations: Server load, privacy
- The gray areas

**HTML Parsing (10 minutes)**
- How web pages are structured
- Beautiful Soup walkthrough
- Demo: scrape a simple page
- Common patterns: lists, tables, links

**Advanced Techniques (10 minutes)**
- JavaScript-rendered content: Selenium
- Pagination and crawling
- Handling failures and changes
- When to stop: signs you're being blocked

#### Part 3: Data Formats (12 minutes)

**The Format Zoo (5 minutes)**
- CSV, JSON, XML, Parquet, Excel
- When to use which
- Common pitfalls: encoding, types

**Parsing Strategies (7 minutes)**
- Pandas for everything tabular
- JSON parsing for nested data
- XML when you must
- Demo: Load messy real-world data

#### Part 4: Building Reliable Pipelines (15 minutes)

**From Script to Pipeline (5 minutes)**
- One-off vs. recurring collection
- Components: extract, transform, load
- Scheduling and orchestration

**Reliability Engineering (5 minutes)**
- Retry logic
- Logging and monitoring
- Error handling
- Idempotency

**The Twitter Lesson (5 minutes)**
- Story of API access changes
- Implications for research
- Strategies for robustness
- Importance of data archiving

#### Part 5: Hands-On Preview (5 minutes)
- Introduce the exercise
- Show the target data sources
- Expectations and tips

#### Wrap-Up (5 minutes)
- Recap key points
- Data acquisition is foundational
- Legal and ethical awareness
- Pipeline thinking
- Preview next module

### Materials Needed
- Jupyter notebook with API examples
- Sample websites for scraping practice
- Various data format samples
- Pipeline architecture diagrams

### Discussion Questions
1. When is it okay to scrape data? When isn't it?
2. What would you do if an API you depend on shut down?
3. How do you balance collecting enough data with respecting rate limits?
4. What makes a data pipeline reliable?

---

## HANDS-ON EXERCISE: Building a Multi-Source Data Collector

### Overview
In this exercise, students will:
1. Collect data from a public API
2. Scrape data from a website
3. Parse and combine multiple data sources
4. Build a basic data pipeline with error handling

### Prerequisites
- Python 3.8+
- Libraries: requests, beautifulsoup4, pandas, schedule

### Setup

```python
# Install required packages
# pip install requests beautifulsoup4 pandas lxml

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
```

### Part 1: Working with APIs (20 minutes)

```python
# We'll use the Open-Meteo weather API (free, no auth required)

def get_weather(latitude, longitude):
    """
    Fetch current weather data from Open-Meteo API.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True,
        "hourly": "temperature_2m,precipitation",
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return None


# Test the function
weather = get_weather(33.89, 35.50)  # Beirut coordinates
if weather:
    print(f"Current temperature: {weather['current_weather']['temperature']}°C")
    print(f"Wind speed: {weather['current_weather']['windspeed']} km/h")
```

**Task 1.1**: Build a multi-city weather collector

```python
cities = [
    {"name": "Beirut", "lat": 33.89, "lon": 35.50},
    {"name": "Dubai", "lat": 25.27, "lon": 55.30},
    {"name": "Cairo", "lat": 30.04, "lon": 31.24},
    {"name": "Amman", "lat": 31.95, "lon": 35.93},
    {"name": "Riyadh", "lat": 24.77, "lon": 46.74}
]

def collect_regional_weather(cities):
    """
    Collect weather for multiple cities with rate limiting.
    """
    results = []

    for city in cities:
        logging.info(f"Fetching weather for {city['name']}")
        weather = get_weather(city['lat'], city['lon'])

        if weather:
            results.append({
                'city': city['name'],
                'temperature': weather['current_weather']['temperature'],
                'windspeed': weather['current_weather']['windspeed'],
                'timestamp': datetime.now().isoformat()
            })

        time.sleep(0.5)  # Rate limiting: 500ms between requests

    return pd.DataFrame(results)


weather_df = collect_regional_weather(cities)
print(weather_df)
```

### Part 2: Web Scraping (25 minutes)

```python
# We'll scrape a simple, scrapable website for practice
# Example: Books to Scrape (http://books.toscrape.com)

def scrape_books(url="http://books.toscrape.com"):
    """
    Scrape book information from Books to Scrape.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch page: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    books = []

    # Find all book containers
    for article in soup.find_all('article', class_='product_pod'):
        # Extract title
        title = article.h3.a['title']

        # Extract price
        price_text = article.find('p', class_='price_color').text
        price = float(price_text.replace('£', ''))

        # Extract rating
        rating_class = article.find('p', class_='star-rating')['class'][1]
        rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
        rating = rating_map.get(rating_class, 0)

        # Extract availability
        availability = article.find('p', class_='instock availability')
        in_stock = 'In stock' in availability.text if availability else False

        books.append({
            'title': title,
            'price': price,
            'rating': rating,
            'in_stock': in_stock
        })

    return books


# Test the scraper
books = scrape_books()
books_df = pd.DataFrame(books)
print(books_df.head(10))
```

**Task 2.1**: Add pagination to scrape multiple pages

```python
def scrape_all_books(base_url="http://books.toscrape.com"):
    """
    Scrape all books across multiple pages.
    """
    all_books = []
    page = 1

    while True:
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}/catalogue/page-{page}.html"

        logging.info(f"Scraping page {page}")

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 404:
                logging.info("No more pages")
                break
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch page {page}: {e}")
            break

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article', class_='product_pod')

        if not articles:
            break

        for article in articles:
            title = article.h3.a['title']
            price = float(article.find('p', class_='price_color').text.replace('£', ''))
            rating_class = article.find('p', class_='star-rating')['class'][1]
            rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}

            all_books.append({
                'title': title,
                'price': price,
                'rating': rating_map.get(rating_class, 0),
                'page': page
            })

        page += 1
        time.sleep(1)  # Be respectful

        # Limit for exercise purposes
        if page > 5:
            break

    return pd.DataFrame(all_books)


all_books_df = scrape_all_books()
print(f"Scraped {len(all_books_df)} books")
print(all_books_df.describe())
```

### Part 3: Handling Multiple Data Formats (15 minutes)

```python
# Simulate receiving data in different formats

# CSV data
csv_data = """name,value,category
item1,100,A
item2,200,B
item3,150,A
"""

# JSON data
json_data = '''
{
    "items": [
        {"name": "item4", "value": 175, "category": "B"},
        {"name": "item5", "value": 225, "category": "C"}
    ],
    "metadata": {"source": "api", "version": "1.0"}
}
'''

# Parse different formats
from io import StringIO

df_csv = pd.read_csv(StringIO(csv_data))
print("CSV Data:")
print(df_csv)

json_parsed = json.loads(json_data)
df_json = pd.DataFrame(json_parsed['items'])
print("\nJSON Data:")
print(df_json)

# Combine data sources
combined_df = pd.concat([df_csv, df_json], ignore_index=True)
print("\nCombined Data:")
print(combined_df)
```

### Part 4: Building a Data Pipeline (20 minutes)

```python
class DataPipeline:
    """
    A simple data collection pipeline with error handling and logging.
    """

    def __init__(self, output_dir="."):
        self.output_dir = output_dir
        self.run_log = []

    def log_run(self, source, status, records=0, error=None):
        """Log pipeline run information."""
        self.run_log.append({
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'status': status,
            'records': records,
            'error': str(error) if error else None
        })

    def fetch_weather_data(self, cities):
        """Fetch weather data with error handling."""
        try:
            df = collect_regional_weather(cities)
            self.log_run('weather', 'success', len(df))
            return df
        except Exception as e:
            self.log_run('weather', 'failed', error=e)
            logging.error(f"Weather collection failed: {e}")
            return pd.DataFrame()

    def fetch_books_data(self, max_pages=3):
        """Fetch books data with error handling."""
        try:
            all_books = []
            for page in range(1, max_pages + 1):
                if page == 1:
                    url = "http://books.toscrape.com"
                else:
                    url = f"http://books.toscrape.com/catalogue/page-{page}.html"

                response = requests.get(url, timeout=10)
                if response.status_code == 404:
                    break

                soup = BeautifulSoup(response.text, 'html.parser')
                for article in soup.find_all('article', class_='product_pod'):
                    all_books.append({
                        'title': article.h3.a['title'],
                        'price': float(article.find('p', class_='price_color').text.replace('£', '')),
                    })
                time.sleep(0.5)

            df = pd.DataFrame(all_books)
            self.log_run('books', 'success', len(df))
            return df
        except Exception as e:
            self.log_run('books', 'failed', error=e)
            logging.error(f"Books collection failed: {e}")
            return pd.DataFrame()

    def run(self):
        """Execute the full pipeline."""
        logging.info("Starting data pipeline")

        # Collect from all sources
        weather_df = self.fetch_weather_data(cities)
        books_df = self.fetch_books_data()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not weather_df.empty:
            weather_file = f"weather_{timestamp}.csv"
            weather_df.to_csv(weather_file, index=False)
            logging.info(f"Saved weather data to {weather_file}")

        if not books_df.empty:
            books_file = f"books_{timestamp}.csv"
            books_df.to_csv(books_file, index=False)
            logging.info(f"Saved books data to {books_file}")

        # Return summary
        return pd.DataFrame(self.run_log)


# Run the pipeline
pipeline = DataPipeline()
run_summary = pipeline.run()
print("\nPipeline Run Summary:")
print(run_summary)
```

### Part 5: Advanced Topics (10 minutes)

**Task 5.1**: Add retry logic

```python
from functools import wraps

def retry(max_attempts=3, delay=1):
    """Decorator to retry failed functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logging.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
            raise last_exception
        return wrapper
    return decorator


@retry(max_attempts=3)
def fetch_with_retry(url):
    """Fetch URL with automatic retry."""
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response


# Test retry logic (this will fail gracefully)
try:
    result = fetch_with_retry("http://httpstat.us/500")  # Always returns 500
except Exception as e:
    print(f"All retries failed: {e}")
```

### Challenge Questions

1. **API Design**: Design an API collection strategy for a service that has a 100 requests/hour limit and you need 10,000 records. How would you structure this?

2. **Scraping Ethics**: You want to collect data from a website for academic research. The robots.txt doesn't prohibit it, but the ToS says "no automated collection." What do you do?

3. **Data Freshness**: You're building a weather dashboard that needs hourly updates. Design a system that handles temporary API outages gracefully.

4. **Deduplication**: Your pipeline runs every hour and might collect overlapping data. How would you handle duplicates?

5. **Scaling Up**: Your single-threaded scraper takes 10 hours to collect all the data you need. How might you speed it up while remaining respectful to the source?

### Expected Outputs

Students should submit:
1. Working code for API data collection with error handling
2. A web scraper with pagination support
3. A data pipeline combining multiple sources
4. Documentation of legal/ethical considerations for chosen data sources
5. Run logs demonstrating successful collection

### Evaluation Rubric

| Criteria | Points |
|----------|--------|
| API collection implementation | 20 |
| Web scraping with error handling | 25 |
| Data format handling | 15 |
| Pipeline design and reliability | 25 |
| Ethics/legal awareness documentation | 10 |
| Code quality | 5 |
| **Total** | **100** |

---

## Recommended Resources

### Documentation
- [Requests Library](https://requests.readthedocs.io/)
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Scrapy](https://docs.scrapy.org/)
- [Selenium](https://selenium-python.readthedocs.io/)

### Tutorials
- [Real Python: Web Scraping with Python](https://realpython.com/python-web-scraping-practical-introduction/)
- [APIs for Beginners](https://www.freecodecamp.org/news/apis-for-beginners/)

### Legal and Ethics
- [Terms of Service; Didn't Read](https://tosdr.org/)
- [EFF: Scraping and the Law](https://www.eff.org/)
- [Ethics of Web Scraping](https://towardsdatascience.com/ethics-in-web-scraping-b96b18136f01)

### API Directories
- [Public APIs](https://github.com/public-apis/public-apis)
- [RapidAPI](https://rapidapi.com/)
- [Any API](https://any-api.com/)

---

## References

1. Mitchell, R. (2018). *Web Scraping with Python* (2nd ed.). O'Reilly Media.

2. Masse, M. (2011). *REST API Design Rulebook*. O'Reilly Media.

3. Krotov, V., & Silva, L. (2018). "Legality and Ethics of Web Scraping." *Proceedings of AMCIS*.

4. Freelon, D. (2018). "Computational Research in the Post-API Age." *Political Communication*, 35(4).

5. Bruns, A. (2019). "After the 'APIcalypse': Social Media Platforms and Their Fight Against Critical Scholarly Research." *Information, Communication & Society*, 22(11).

---

*Module 1 of DATA 202 explores the foundational skill of data acquisition—the methods and tools for obtaining data from APIs, websites, and other sources. Through the story of the Twitter API's rise and fall, we learn that data access is never guaranteed, and responsible data science requires understanding both the technical and ethical dimensions of data collection.*
