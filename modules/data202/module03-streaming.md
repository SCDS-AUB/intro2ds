---
layout: default
title: "DATA 202 Module 3: Real-Time and Streaming Data"
---

# DATA 202 Module 3: Real-Time and Streaming Data

## Introduction

The world generates data continuously—every sensor reading, every click, every transaction, every heartbeat. Traditional batch processing assumes data sits still while you analyze it. But increasingly, data arrives in streams that must be processed as they flow.

This module explores streaming data paradigms: the technologies, architectures, and use cases for processing data in motion. From IoT sensors to financial markets to social media firehoses, we'll learn to build systems that respond in real-time.

---

## Part 1: Batch vs. Stream Processing

### The Batch Paradigm

Traditional data processing is batch-oriented:
1. Collect data over a period
2. Store in a data warehouse or lake
3. Run analysis on the complete dataset
4. Generate reports

**Limitations**:
- Latency: Hours or days between event and insight
- Staleness: Decisions based on yesterday's data
- Missed Opportunities: Real-time patterns invisible

### The Streaming Paradigm

Stream processing treats data as a continuous flow:
- Process each event as it arrives
- Maintain running calculations (counts, averages, windows)
- Trigger actions immediately
- Handle unbounded datasets

**Trade-offs**:
- Complexity: Harder to reason about stateful processing
- Ordering: Events may arrive out of order
- Exactly-once: Guaranteeing each event processed exactly once is hard
- State management: Where do running aggregates live?

### Use Cases for Streaming

**Financial Services**: Fraud detection, algorithmic trading, risk monitoring
**IoT**: Sensor monitoring, predictive maintenance, smart cities
**Social Media**: Trending topics, content moderation, recommendations
**E-commerce**: Real-time personalization, inventory tracking
**Transportation**: Vehicle tracking, routing optimization, ETA calculation

---

## Part 2: Streaming Concepts

### Events and Streams

An **event** is an immutable record of something that happened:
```json
{
    "event_type": "purchase",
    "user_id": "user123",
    "product_id": "prod456",
    "amount": 29.99,
    "timestamp": "2024-01-15T14:30:00Z"
}
```

A **stream** is an unbounded sequence of events, ordered by time.

### Windowing

Since streams are unbounded, we need ways to group events:

**Tumbling Windows**: Fixed, non-overlapping intervals
- "Count events per minute"
- Each event belongs to exactly one window

**Sliding Windows**: Overlapping intervals
- "Average over last 5 minutes, updated every second"
- Events may belong to multiple windows

**Session Windows**: Dynamic windows based on activity
- Group events within idle gaps
- "All actions in a browsing session"

### Time Semantics

**Event Time**: When the event actually occurred
**Processing Time**: When the system processes the event
**Ingestion Time**: When the event enters the system

Late-arriving data complicates event-time processing—an event timestamped 10:00 may arrive at 10:05.

### State and Checkpointing

Streaming computations maintain **state**:
- Running counts and aggregates
- Session information
- Machine learning model parameters

**Checkpointing** saves state periodically for fault tolerance—if a processor fails, restart from the last checkpoint.

---

## Part 3: Streaming Technologies

### Apache Kafka

**Kafka** is the dominant streaming platform:
- **Topics**: Categories of messages
- **Producers**: Publish messages to topics
- **Consumers**: Subscribe to topics and process messages
- **Partitions**: Enable parallel processing and scaling
- **Retention**: Messages stored for configurable time

Kafka is more "log" than "queue"—messages persist and can be replayed.

```python
from kafka import KafkaProducer, KafkaConsumer
import json

# Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
producer.send('events', {'user': 'john', 'action': 'click'})

# Consumer
consumer = KafkaConsumer(
    'events',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)
for message in consumer:
    print(message.value)
```

### Apache Spark Streaming

Spark Streaming processes micro-batches—small batch jobs run frequently:
- Leverages Spark's batch capabilities
- Unified API for batch and stream
- Structured Streaming uses DataFrames

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("StreamExample").getOrCreate()

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "events") \
    .load()

# Process
events = df.selectExpr("CAST(value AS STRING)")

# Write results
query = events.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()
```

### Apache Flink

Flink is a true stream processor (not micro-batch):
- Event-at-a-time processing
- Rich windowing and event-time handling
- Exactly-once semantics
- Low latency

### Cloud Services

**AWS**: Kinesis Data Streams, Kinesis Data Analytics
**Google Cloud**: Pub/Sub, Dataflow
**Azure**: Event Hubs, Stream Analytics

---

## Part 4: IoT and Sensor Data

### The IoT Data Challenge

IoT devices generate continuous streams:
- Thousands or millions of devices
- High frequency (sensor readings per second)
- Often unreliable connectivity
- Edge processing may be required

### Edge Computing

Process data at the **edge** (on or near the device):
- Reduce bandwidth by filtering/aggregating
- Lower latency for time-critical decisions
- Handle intermittent connectivity

### Time-Series Databases

IoT data is inherently time-series:
- **InfluxDB**: Purpose-built for time-series
- **TimescaleDB**: PostgreSQL extension
- **Prometheus**: Monitoring-focused
- **QuestDB**: High-performance ingestion

---

## DEEP DIVE: The Lambda Architecture and Its Evolution

### Handling Two Speeds of Truth

In 2011, **Nathan Marz**, working on Twitter's analytics, faced a dilemma. Batch processing (Hadoop) gave accurate, complete results but took hours. Real-time processing gave immediate results but lacked completeness and accuracy. How to get both?

His answer was the **Lambda Architecture**:

**Batch Layer**: Process complete historical data periodically
- Master dataset: immutable, append-only
- Batch views: Precomputed from complete data
- Slow but accurate

**Speed Layer**: Process real-time data as it arrives
- Compensate for batch latency
- Approximate views
- Fast but potentially incomplete

**Serving Layer**: Merge batch and real-time views
- Answer queries from both
- Real-time view replaced by batch when available

The Lambda Architecture solved real problems and was widely adopted. But it had a significant flaw: you had to maintain two separate codebases doing essentially the same computation.

### The Kappa Architecture

**Jay Kreps**, while at LinkedIn developing Kafka, proposed a simpler approach:

**Use streaming for everything**. Store all data in a log (Kafka). Run streaming jobs to produce views. If you need to recompute from scratch, replay the log.

The Kappa Architecture eliminates code duplication—one codebase, one paradigm.

### Modern Reality

Today, most organizations use hybrid approaches:
- Kafka for real-time ingestion
- Cloud data warehouses for historical analysis
- Streaming for real-time views
- The distinction between batch and stream blurs with tools like Spark and Flink that handle both

### Lesson for Data Scientists

Understand both paradigms. Many problems need real-time responses built on historical context. The architecture choice depends on latency requirements, accuracy needs, and operational complexity tolerance.

---

## HANDS-ON EXERCISE: Building a Real-Time Data Pipeline

### Overview
Students will:
1. Set up a local Kafka environment
2. Build a producer that generates streaming events
3. Create a consumer that processes and aggregates
4. Implement windowed analytics

### Setup (Using Docker)

```bash
# docker-compose.yml for Kafka
# Start with: docker-compose up -d
```

### Part 1: Event Producer

```python
import json
import random
import time
from datetime import datetime
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_event():
    return {
        'event_id': random.randint(1, 1000000),
        'user_id': f'user_{random.randint(1, 100)}',
        'action': random.choice(['view', 'click', 'purchase']),
        'amount': round(random.uniform(1, 100), 2) if random.random() > 0.7 else 0,
        'timestamp': datetime.now().isoformat()
    }

while True:
    event = generate_event()
    producer.send('user_events', event)
    print(f"Sent: {event}")
    time.sleep(0.1)
```

### Part 2: Streaming Consumer with Aggregation

```python
from kafka import KafkaConsumer
from collections import defaultdict
from datetime import datetime, timedelta
import json

consumer = KafkaConsumer(
    'user_events',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest'
)

# Track events in current window
window_size = timedelta(seconds=10)
window_start = datetime.now()
window_counts = defaultdict(int)

for message in consumer:
    event = message.value
    window_counts[event['action']] += 1

    # Check if window expired
    if datetime.now() - window_start > window_size:
        print(f"\n=== Window Summary ===")
        for action, count in window_counts.items():
            print(f"{action}: {count}")

        # Reset window
        window_start = datetime.now()
        window_counts = defaultdict(int)
```

---

## Recommended Resources

### Books
- *Streaming Systems* by Tyler Akidau, et al.
- *Kafka: The Definitive Guide* by Neha Narkhede, et al.
- *Designing Data-Intensive Applications* by Martin Kleppmann

### Documentation
- [Apache Kafka](https://kafka.apache.org/documentation/)
- [Apache Flink](https://flink.apache.org/)
- [Spark Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)

---

*Module 3 explores real-time and streaming data processing—the technologies and paradigms for handling data in motion. From Lambda and Kappa architectures to modern streaming platforms, we learn to build systems that respond as events unfold.*
