---
layout: default
title: "DATA 202 Module 8: Foundation Models and Generative AI"
---

# DATA 202 Module 8: Foundation Models and Generative AI

## Introduction

A paradigm shift has occurred in AI. Instead of training specialized models for each task, we now train massive models on vast datasets and adapt them for downstream applications. These **foundation models**—GPT, BERT, CLIP, Stable Diffusion—serve as the foundation for countless applications.

This module explores the foundation model paradigm: how these models work, how to use them effectively, and what their emergence means for data science practice.

---

## Part 1: The Foundation Model Paradigm

### What Makes a Foundation Model?

**Foundation models** share key characteristics:
- Trained on massive, diverse datasets
- Large scale (billions of parameters)
- Self-supervised learning (no manual labels)
- Transfer to many downstream tasks
- Emergent capabilities at scale

### The Training Recipe

**Pre-training**: Learn general representations from unlabeled data
- Language models: Predict next token
- Vision models: Contrastive learning, masked image modeling
- Multimodal: Align representations across modalities

**Fine-tuning**: Adapt to specific tasks with labeled data
- Full fine-tuning: Update all parameters
- Parameter-efficient: Update subset (LoRA, adapters)
- Prompt tuning: Learn task-specific prompts

**In-context learning**: Task specification through examples
- No gradient updates
- Model learns from context provided in prompt

### The Scaling Laws

Research has shown predictable relationships:
- More data → better performance
- More parameters → better performance
- More compute → better performance

**Chinchilla scaling**: Optimal to scale data and parameters together

---

## Part 2: Large Language Models

### How LLMs Work

**Tokenization**: Break text into tokens
```python
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer.encode("Hello, world!")
# [15496, 11, 995, 0]
```

**Architecture**: Transformer decoder
- Self-attention: Each position attends to all previous positions
- Feed-forward layers: Transform representations
- Layer normalization: Stabilize training

**Training**: Predict next token
$$\mathcal{L} = -\sum_t \log P(x_t | x_{<t})$$

### Using LLMs

**Through APIs**:
```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain gradient descent in simple terms."}
    ]
)
print(response.choices[0].message.content)
```

**Local inference**:
```python
from transformers import pipeline

generator = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf")
output = generator("The key to machine learning is", max_length=50)
```

### Prompt Engineering

The art of crafting effective prompts:

**Zero-shot**: Direct instruction
```
Classify this review as positive or negative: "Great product!"
```

**Few-shot**: Provide examples
```
Review: "Terrible service" → Negative
Review: "Love it!" → Positive
Review: "Waste of money" → ?
```

**Chain-of-thought**: Request reasoning
```
Solve step by step: If a train travels 60 mph for 2.5 hours, how far does it go?
```

**Role prompting**: Set context
```
You are an expert data scientist. Explain...
```

---

## Part 3: Multimodal Models

### Vision-Language Models

**CLIP** (OpenAI, 2021): Connects images and text
- Train on 400M image-text pairs
- Contrastive learning aligns embeddings
- Zero-shot image classification via text prompts

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image,
    return_tensors="pt"
)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)
```

### Vision-Language Understanding

**GPT-4V, Gemini, Claude**: LLMs that understand images
- Describe images
- Answer questions about images
- Reason about visual content

### Text-to-Image Generation

**Stable Diffusion, DALL-E, Midjourney**: Generate images from text
- Diffusion models: Gradually denoise random noise
- Text conditioning: CLIP embeddings guide generation

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
image = pipe("A serene lake at sunset, oil painting style").images[0]
image.save("output.png")
```

---

## Part 4: RAG and Knowledge Integration

### Retrieval-Augmented Generation

LLMs have knowledge cutoffs and can hallucinate. **RAG** grounds generation in retrieved documents:

1. **Index**: Embed documents into vector database
2. **Retrieve**: Find relevant documents for query
3. **Generate**: Provide documents as context for LLM

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Create vector store from documents
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever()
)

answer = qa.run("What is the company's return policy?")
```

### Vector Databases

Store and search embeddings efficiently:
- **Pinecone**: Managed vector database
- **Weaviate**: Open-source, multimodal
- **Chroma**: Lightweight, Python-native
- **Faiss**: Facebook's similarity search
- **Milvus**: Scalable open-source

### Agents and Tool Use

LLMs as reasoning engines that use tools:
- Search the web
- Execute code
- Query databases
- Call APIs

```python
from langchain.agents import create_openai_functions_agent

tools = [search_tool, calculator_tool, code_executor_tool]
agent = create_openai_functions_agent(llm, tools, prompt)
result = agent.invoke({"input": "What's the current weather in Beirut?"})
```

---

## Part 5: Fine-Tuning and Customization

### When to Fine-Tune

Fine-tuning makes sense when:
- Task requires specialized knowledge
- Consistent style or format needed
- Performance gains justify cost
- Data privacy requires local model

### Parameter-Efficient Fine-Tuning

**LoRA (Low-Rank Adaptation)**:
- Add small trainable matrices to attention layers
- Freeze original weights
- Dramatic reduction in trainable parameters

```python
from peft import get_peft_model, LoraConfig

config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
model = get_peft_model(base_model, config)
# Only ~0.1% of parameters are trainable
```

### Instruction Tuning

Train models to follow instructions:
- Collect instruction-output pairs
- Fine-tune on diverse instructions
- Results in more helpful, harmless models

---

## DEEP DIVE: The GPT Series and the AI Revolution

### From GPT-1 to GPT-4

**GPT-1 (2018)**: 117M parameters. Showed language model pre-training + fine-tuning works.

**GPT-2 (2019)**: 1.5B parameters. Zero-shot task performance. OpenAI initially withheld due to misuse concerns.

**GPT-3 (2020)**: 175B parameters. Few-shot learning. The paper "Language Models are Few-Shot Learners" demonstrated emergent capabilities at scale.

**GPT-4 (2023)**: Unknown size (rumored 1T+). Multimodal (vision). Passes professional exams. Powers ChatGPT Plus.

### The Emergence of Capabilities

At certain scales, new abilities appear suddenly:
- Arithmetic: Emerges around 10B parameters
- Chain-of-thought reasoning: Emerges around 100B
- Theory of mind: Debated, but appears in larger models

This **emergence** is both exciting and concerning—we can't predict what abilities will appear at the next scale.

### The Industry Transformation

ChatGPT's release in November 2022 was a watershed:
- 100M users in 2 months (fastest ever)
- Microsoft invested $10B in OpenAI
- Google declared "code red"
- Startups raised billions
- Every company exploring AI integration

The foundation model paradigm is reshaping software development, knowledge work, and potentially education, healthcare, and law.

---

## HANDS-ON EXERCISE: Building with Foundation Models

### Part 1: Text Generation with Open Models

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "microsoft/phi-2"  # Small but capable
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

def generate(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate("Explain machine learning to a 10-year-old:"))
```

### Part 2: Image Understanding with CLIP

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def classify_image(image_path, labels):
    image = Image.open(image_path)
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return {label: prob.item() for label, prob in zip(labels, probs[0])}

labels = ["a cat", "a dog", "a bird", "a car"]
result = classify_image("image.jpg", labels)
```

### Part 3: RAG System

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load and split documents
loader = TextLoader("documents.txt")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000)
docs = splitter.split_documents(documents)

# Create vector store
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# Query
query = "What is the main topic?"
results = db.similarity_search(query, k=3)
for doc in results:
    print(doc.page_content[:200])
```

---

## Recommended Resources

### Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [LangChain](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs)

### Courses
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- [DeepLearning.AI Generative AI Courses](https://www.deeplearning.ai/)

### Papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Few-Shot Learners" (Brown et al., 2020)
- "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)

---

*Module 8 explores foundation models and generative AI—the paradigm shift from task-specific models to massive pre-trained systems that can be adapted for countless applications. Understanding how to use and customize these models is essential for modern data science.*
