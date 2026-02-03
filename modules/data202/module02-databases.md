---
layout: default
title: "DATA 202 Module 2: Databases and Data Storage"
---

# DATA 202 Module 2: Databases and Data Storage

## Introduction

Where does data live? The answer has evolved dramatically—from filing cabinets to magnetic tape, from hierarchical databases to the relational model, from SQL to NoSQL to data lakes. Understanding databases is essential for any data scientist working with data at scale.

This module explores database fundamentals, SQL for data querying, NoSQL alternatives, and modern data infrastructure. We'll learn not just how to use these systems but how to choose among them for different applications.

---

## Part 1: The Relational Revolution

### Before Relational Databases

In the 1960s, data management was a mess. Early computer systems used hierarchical databases (like trees) or network databases (like graphs). To find data, you had to know exactly how it was stored and navigate physical pointers between records.

**Problems**:
- Changing the data structure broke applications
- Queries had to be hardcoded into programs
- Performance optimization was manual and complex
- Data redundancy and inconsistency were common

### Edgar Codd's Vision (1970)

In 1970, **Edgar F. Codd** (1923-2003), a mathematician and researcher at IBM, published a paper that would transform computing: "A Relational Model of Data for Large Shared Data Banks."

Codd's insight was radical: separate the logical organization of data from its physical storage. Data should be organized into **relations** (tables) with **rows** (tuples) and **columns** (attributes), and users should query data using a high-level language without needing to know how it was stored.

This abstraction had profound implications:
- **Data Independence**: Change how data is stored without changing applications
- **Declarative Queries**: Say *what* you want, not *how* to get it
- **Mathematical Foundation**: Relational algebra provided rigorous semantics
- **Optimization**: The database, not the programmer, optimizes query execution

### The Rise of SQL

Codd's ideas needed a practical query language. IBM researchers developed SEQUEL (Structured English Query Language), later renamed **SQL** (Structured Query Language).

SQL became the standard way to interact with relational databases:

```sql
SELECT customer_name, SUM(order_total)
FROM customers
JOIN orders ON customers.id = orders.customer_id
WHERE order_date > '2024-01-01'
GROUP BY customer_name
ORDER BY SUM(order_total) DESC;
```

SQL's power lies in its declarative nature: you describe the result you want, and the database figures out how to compute it efficiently.

---

## Part 2: Relational Database Fundamentals

### Tables, Keys, and Relationships

**Table (Relation)**: A collection of related data entries
- Rows represent individual records
- Columns represent attributes
- Each row has a unique identifier (primary key)

**Primary Key**: Unique identifier for each row
- Can be a single column or combination of columns
- Cannot be NULL

**Foreign Key**: Reference to a primary key in another table
- Creates relationships between tables
- Enforces referential integrity

**Relationships**:
- One-to-One: One customer has one profile
- One-to-Many: One customer has many orders
- Many-to-Many: Many students take many courses (requires junction table)

### Normalization

**Normalization** organizes data to reduce redundancy and improve integrity:

**1NF (First Normal Form)**: No repeating groups; atomic values
**2NF**: No partial dependencies on composite keys
**3NF**: No transitive dependencies

Denormalization trades some redundancy for query performance—common in analytics.

### SQL Fundamentals

**Creating Tables**:
```sql
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Inserting Data**:
```sql
INSERT INTO customers (name, email)
VALUES ('John Doe', 'john@example.com');
```

**Querying Data**:
```sql
SELECT name, email
FROM customers
WHERE created_at > '2024-01-01'
ORDER BY name;
```

**Joining Tables**:
```sql
SELECT c.name, o.order_date, o.total
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE o.total > 100;
```

**Aggregating**:
```sql
SELECT customer_id, COUNT(*) as order_count, SUM(total) as total_spent
FROM orders
GROUP BY customer_id
HAVING SUM(total) > 1000;
```

---

## Part 3: Advanced SQL for Data Science

### Window Functions

Window functions perform calculations across related rows without collapsing them:

```sql
SELECT
    customer_id,
    order_date,
    total,
    SUM(total) OVER (PARTITION BY customer_id ORDER BY order_date) as running_total,
    RANK() OVER (PARTITION BY customer_id ORDER BY total DESC) as order_rank
FROM orders;
```

### Common Table Expressions (CTEs)

CTEs make complex queries readable:

```sql
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', order_date) as month,
        SUM(total) as sales
    FROM orders
    GROUP BY 1
),
growth AS (
    SELECT
        month,
        sales,
        LAG(sales) OVER (ORDER BY month) as prev_sales
    FROM monthly_sales
)
SELECT
    month,
    sales,
    (sales - prev_sales) / prev_sales * 100 as growth_pct
FROM growth;
```

### Subqueries

Queries within queries:

```sql
SELECT * FROM customers
WHERE id IN (
    SELECT customer_id FROM orders
    WHERE total > (SELECT AVG(total) FROM orders)
);
```

### SQL for Analytics

**Percentiles**:
```sql
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total) as median_order
FROM orders;
```

**Date Operations**:
```sql
SELECT
    DATE_TRUNC('week', order_date) as week,
    COUNT(*) as orders
FROM orders
WHERE order_date > CURRENT_DATE - INTERVAL '30 days'
GROUP BY 1;
```

---

## Part 4: NoSQL Databases

### Beyond Relational

Relational databases excel at structured data with complex relationships. But they struggle with:
- Massive scale (billions of records, petabytes)
- Unstructured or semi-structured data
- High write throughput
- Flexible schemas

**NoSQL** ("Not Only SQL") databases emerged to address these needs.

### Document Databases (MongoDB)

Store data as flexible JSON-like documents:

```json
{
    "_id": "user123",
    "name": "John Doe",
    "email": "john@example.com",
    "orders": [
        {"product": "Widget", "price": 29.99},
        {"product": "Gadget", "price": 49.99}
    ],
    "preferences": {
        "newsletter": true,
        "theme": "dark"
    }
}
```

**Use cases**: Content management, user profiles, product catalogs

### Key-Value Stores (Redis)

Simple but extremely fast: store and retrieve by key.

```
SET user:123:session "abc123"
GET user:123:session
EXPIRE user:123:session 3600
```

**Use cases**: Caching, sessions, real-time analytics

### Column-Family Stores (Cassandra)

Optimized for write-heavy workloads and horizontal scaling.

**Use cases**: Time-series data, IoT, large-scale logging

### Graph Databases (Neo4j)

Store nodes and relationships explicitly; optimized for graph queries.

```cypher
MATCH (u:User)-[:FOLLOWS]->(friend)-[:FOLLOWS]->(foaf)
WHERE u.name = "John"
RETURN foaf.name, COUNT(*) as mutual_friends
```

**Use cases**: Social networks, recommendation engines, fraud detection

---

## Part 5: Modern Data Infrastructure

### Data Warehouses

**Data warehouses** store historical data optimized for analytical queries:
- Columnar storage for fast aggregations
- Star and snowflake schemas for dimensional modeling
- Separation of storage and compute

**Major Platforms**:
- **Snowflake**: Cloud-native, scales storage/compute independently
- **Google BigQuery**: Serverless, SQL interface
- **Amazon Redshift**: AWS-integrated, columnar
- **Databricks**: Combines warehouse and data lake

### Data Lakes

**Data lakes** store raw data in native format:
- Schema-on-read (structure applied at query time)
- Handles structured, semi-structured, unstructured data
- Cost-effective for large volumes
- Often built on cloud object storage (S3, GCS, Azure Blob)

### The Lakehouse Architecture

Modern systems combine warehouse and lake benefits:
- Raw data stored in lakes
- Structured layer for analytics
- Delta Lake, Iceberg, Hudi for ACID transactions on lakes

### ETL and ELT

**ETL** (Extract, Transform, Load): Transform data before loading
**ELT** (Extract, Load, Transform): Load raw data, transform in warehouse

Modern trend is toward ELT with powerful cloud warehouses.

---

## DEEP DIVE: Edgar Codd and the Invention of Relational Databases

### The Man Who Abstracted Data

In 1970, the IBM research lab in San Jose housed some of the world's best computer scientists. Among them was a quiet British mathematician who would fundamentally change how humanity stores and accesses information.

**Edgar Frank Codd** (1923-2003) had an unusual path to computing. Born in England, he served as a pilot in the Royal Air Force during World War II. After the war, he studied mathematics at Oxford, then emigrated to the United States to work in computing—first at IBM, where he programmed early machines, then at the University of Michigan for a PhD, and back to IBM's research division.

By the late 1960s, Codd had grown frustrated with how computers handled data. The dominant approaches—hierarchical and network databases—were engineering marvels that nonetheless failed to separate logical data organization from physical storage. To retrieve customer orders, a program had to navigate pointer chains hardcoded based on how data was physically arranged. Change the physical layout, and every program broke.

### The Paper That Changed Everything

In June 1970, Codd published "A Relational Model of Data for Large Shared Data Banks" in Communications of the ACM. The paper was densely mathematical, drawing on set theory and predicate logic. Its core claims were radical:

1. **Data should be organized into relations (tables)**—two-dimensional structures with rows and columns.

2. **All data access should be through high-level operations** on relations: select rows meeting criteria, project columns, join relations on shared values.

3. **The physical storage should be completely hidden**—users and applications should know nothing about how data is stored on disk.

4. **A query optimizer should translate logical queries into efficient physical operations**—the system, not the programmer, should figure out the best execution path.

### IBM's Reluctant Revolution

You might expect IBM to embrace Codd's ideas. Instead, they resisted. IBM had invested heavily in IMS, a hierarchical database that powered major customers. Codd's relational model threatened to obsolete their successful product.

The conflict within IBM became legendary. Codd lobbied internally, gave talks, wrote more papers. IBM eventually funded a research prototype—System R—but kept it separate from product development. Meanwhile, a small Berkeley project called Ingres demonstrated that relational ideas could be practical.

The market eventually forced IBM's hand. When Oracle (founded as Software Development Laboratories in 1977) beat IBM to market with a commercial relational database, IBM finally released their own product—DB2—in 1983.

### The Impact

Today, relational databases are everywhere:
- Nearly every web application uses MySQL, PostgreSQL, or similar
- Enterprise systems run on Oracle, SQL Server, DB2
- Even NoSQL systems often support SQL-like queries
- Cloud data warehouses are fundamentally relational

Codd received the Turing Award in 1981 for his work. He continued advocating for proper relational implementation (publishing "Codd's 12 Rules" to define what qualified as truly relational), but by then the industry he created had moved beyond his control.

### Lessons for Data Science

1. **Abstraction is powerful**: Separating what you want from how to get it enables both user simplicity and system optimization.

2. **Theory precedes practice**: Codd's mathematical foundation allowed decades of optimization and extension.

3. **Incumbents resist disruption**: Even IBM, the company that employed Codd, was slow to adopt his ideas.

4. **Good ideas spread**: Despite resistance, relational databases became dominant because they solved real problems better.

5. **Know your foundations**: Modern data scientists use SQL daily; understanding its theoretical basis deepens practice.

---

## LECTURE PLAN: From Codd to Cloud - Understanding Data Storage

### Learning Objectives
By the end of this lecture, students will be able to:
1. Explain the relational model and its advantages
2. Write SQL queries for data analysis
3. Choose appropriate database types for different use cases
4. Understand modern data infrastructure (warehouses, lakes)

### Lecture Structure (90 minutes)

#### Opening Hook (7 minutes)
- Show the chaos of 1960s data management
- Present Codd's radical idea: abstract logical from physical
- Demo: A simple SQL query retrieving complex joins
- "This ease of use revolutionized computing"

#### Part 1: Relational Foundations (20 minutes)
- Tables, keys, relationships
- Normalization concepts
- SQL basics: SELECT, FROM, WHERE, JOIN, GROUP BY
- Live demo: Build and query a simple database

#### Part 2: SQL for Analytics (20 minutes)
- Aggregations and grouping
- Window functions
- CTEs for readability
- Subqueries
- Demo: Analyze a realistic dataset

#### Part 3: Beyond Relational (15 minutes)
- When SQL isn't enough
- Document databases (MongoDB)
- Key-value stores (Redis)
- Graph databases (Neo4j)
- Choosing the right tool

#### Part 4: Modern Data Infrastructure (15 minutes)
- Data warehouses: Snowflake, BigQuery, Redshift
- Data lakes and object storage
- The lakehouse pattern
- ETL vs. ELT

#### Part 5: Codd's Story (8 minutes)
- The man who abstracted data
- IBM's resistance
- Legacy and lessons

#### Wrap-Up (5 minutes)
- Recap key concepts
- SQL is foundational
- Modern infrastructure builds on relational ideas
- Preview hands-on exercise

---

## HANDS-ON EXERCISE: SQL for Data Science

### Overview
Students will:
1. Design and create a relational database schema
2. Write analytical SQL queries
3. Compare SQL and pandas approaches
4. Work with a cloud database

### Setup

```python
# Using SQLite for local practice
import sqlite3
import pandas as pd

# Create an in-memory database
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()
```

### Part 1: Creating Tables

```sql
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    country TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    price REAL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER,
    order_date DATETIME,
    total REAL
);
```

### Part 2: Analytical Queries

```sql
-- Top customers by total spending
SELECT c.name, SUM(o.total) as total_spent
FROM customers c
JOIN orders o ON c.id = o.customer_id
GROUP BY c.id
ORDER BY total_spent DESC
LIMIT 10;

-- Monthly sales trend
SELECT
    strftime('%Y-%m', order_date) as month,
    SUM(total) as sales,
    COUNT(*) as order_count
FROM orders
GROUP BY 1
ORDER BY 1;

-- Product category analysis
SELECT
    p.category,
    COUNT(DISTINCT o.customer_id) as unique_customers,
    SUM(o.quantity) as units_sold,
    SUM(o.total) as revenue
FROM products p
JOIN orders o ON p.id = o.product_id
GROUP BY p.category;
```

### Challenge: Customer Segmentation Query

```sql
WITH customer_stats AS (
    SELECT
        customer_id,
        COUNT(*) as order_count,
        SUM(total) as total_spent,
        AVG(total) as avg_order,
        MAX(order_date) as last_order
    FROM orders
    GROUP BY customer_id
)
SELECT
    c.name,
    cs.order_count,
    cs.total_spent,
    CASE
        WHEN cs.total_spent > 1000 THEN 'Gold'
        WHEN cs.total_spent > 500 THEN 'Silver'
        ELSE 'Bronze'
    END as segment
FROM customer_stats cs
JOIN customers c ON cs.customer_id = c.id;
```

---

## Recommended Resources

### Books
- *Database System Concepts* by Silberschatz, Korth, Sudarshan
- *SQL for Data Scientists* by Renee Teate
- *Designing Data-Intensive Applications* by Martin Kleppmann

### Online
- [SQLZoo](https://sqlzoo.net/) - Interactive SQL tutorial
- [Mode Analytics SQL Tutorial](https://mode.com/sql-tutorial/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

---

## References

1. Codd, E.F. (1970). "A Relational Model of Data for Large Shared Data Banks." *Communications of the ACM*, 13(6), 377-387.

2. Date, C.J. (2003). *An Introduction to Database Systems* (8th ed.). Addison-Wesley.

3. Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media.

---

*Module 2 explores databases and data storage—from Edgar Codd's revolutionary relational model to modern cloud data infrastructure. Understanding where and how data lives is foundational to working with data at scale.*
