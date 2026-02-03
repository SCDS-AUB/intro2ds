---
layout: default
title: "DATA 202 Module 4: OCR and Document AI"
---

# DATA 202 Module 4: OCR and Document AI

## Introduction

Much of the world's knowledge is locked in documents—scanned papers, PDFs, handwritten notes, historical records, receipts, and forms. Extracting information from these documents requires **Optical Character Recognition (OCR)** and increasingly sophisticated **Document AI** systems.

This module explores the technologies for converting document images to structured data: from classical OCR to modern deep learning approaches that understand document layout, tables, and meaning.

---

## Part 1: From Images to Text

### The OCR Pipeline

Traditional OCR follows a pipeline:

1. **Image Preprocessing**: Deskew, denoise, binarize
2. **Layout Analysis**: Detect text regions, columns, paragraphs
3. **Line Segmentation**: Separate individual text lines
4. **Character Segmentation**: Isolate characters
5. **Character Recognition**: Classify each character
6. **Post-processing**: Language models, spell checking

### Classical Approaches

**Template Matching**: Compare character images against known templates
- Fast but brittle
- Fails with font variation

**Feature-Based**: Extract features (strokes, curves), classify
- More robust to variation
- Still struggles with noise

**Hidden Markov Models**: Model character sequences
- Better context integration
- Foundation for early commercial systems

### The Tesseract Story

**Tesseract**, now the world's most-used open-source OCR engine, has a remarkable history:

- Developed at HP Labs 1985-1994
- Considered best OCR engine of its era
- Abandoned as HP exited the OCR market
- Released as open source in 2005
- Now maintained by Google, powered by LSTM

```python
import pytesseract
from PIL import Image

# Basic OCR
text = pytesseract.image_to_string(Image.open('document.png'))

# With language specification
text = pytesseract.image_to_string(
    Image.open('arabic_doc.png'),
    lang='ara'
)

# Get detailed information
data = pytesseract.image_to_data(
    Image.open('document.png'),
    output_type=pytesseract.Output.DICT
)
```

---

## Part 2: Deep Learning Revolution

### End-to-End Recognition

Modern OCR uses deep learning for end-to-end recognition:

**CNNs for Visual Features**: Extract visual representations
**RNNs/Transformers for Sequences**: Model character sequences
**CTC Loss**: Handle variable-length sequences without explicit alignment

### Key Architectures

**CRNN (Convolutional Recurrent Neural Network)**:
- CNN extracts features from image strips
- BiLSTM processes sequence
- CTC decodes to text

**Attention-Based Models**:
- Encoder processes full image
- Decoder attends to relevant regions
- Generates characters autoregressively

**Transformer-Based OCR**:
- Vision Transformer (ViT) for encoding
- Text decoder for generation
- TrOCR, Donut, etc.

### Cloud OCR Services

**Google Cloud Vision**: General OCR, document parsing
**AWS Textract**: Form and table extraction
**Azure Computer Vision**: OCR and document analysis
**Google Document AI**: Specialized document processors

---

## Part 3: Document Understanding

### Beyond OCR: Document AI

OCR extracts text. Document AI understands structure:

**Layout Analysis**: Headers, paragraphs, captions, tables
**Table Extraction**: Rows, columns, cells
**Form Understanding**: Field-value pairs
**Key Information Extraction**: Specific data from documents

### LayoutLM and Document Transformers

**LayoutLM** (Microsoft) revolutionized document understanding:
- Combines text, layout position, and visual features
- Pre-trained on large document datasets
- Fine-tunes for specific extraction tasks

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base"
)

# Process document image
encoding = processor(image, return_tensors="pt")
outputs = model(**encoding)
```

### Handwriting Recognition

Handwriting recognition (HTR) is harder than printed text:
- Huge variation between writers
- Connected cursive scripts
- Historical documents have archaic styles

Modern approaches:
- Transformer-based models
- Writer adaptation techniques
- Large-scale pretraining

---

## Part 4: Practical Document Processing

### PDF Extraction

PDFs may be:
- **Text-based**: Text extractable directly
- **Image-based**: Scanned images requiring OCR
- **Mixed**: Combination of text and images

```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    for page in pdf.pages:
        # Extract text
        text = page.extract_text()

        # Extract tables
        tables = page.extract_tables()
        for table in tables:
            df = pd.DataFrame(table[1:], columns=table[0])
```

### Preprocessing for Better OCR

Image quality dramatically affects OCR:

```python
import cv2
import numpy as np

def preprocess_for_ocr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Denoise
    img = cv2.fastNlMeansDenoising(img, h=10)

    # Binarize (Otsu's method)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    return img
```

### Handling Arabic Script

Arabic presents unique challenges:
- Right-to-left direction
- Connected letters with multiple forms
- Diacritics (harakat)
- Mixed with left-to-right elements

```python
# Tesseract with Arabic
text = pytesseract.image_to_string(
    image,
    lang='ara',
    config='--psm 3'  # Automatic page segmentation
)
```

---

## DEEP DIVE: The Digitization of Human Knowledge

### The Dream of Universal Access

In 2004, Google announced an audacious project: scan every book ever printed and make them searchable. The Google Books project would eventually scan over 40 million books—a significant fraction of humanity's written knowledge.

But scanning is just imaging. Making books searchable required OCR at unprecedented scale:
- Degraded historical texts
- Hundreds of languages
- Gothic typefaces, handwriting, marginalia
- Billions of pages

### reCAPTCHA: Humans as OCR Workers

When computers struggled with difficult words, Google found an ingenious solution: **reCAPTCHA**.

Remember those distorted word puzzles used to prove you're human? They served a dual purpose. One word tested you; the other was an OCR failure that needed human help. By distributing difficult words across millions of CAPTCHA challenges, Google crowdsourced OCR correction.

reCAPTCHA digitized 13 million articles from the New York Times archives and helped correct Google Books. Users unknowingly contributed to the digitization of human knowledge, one CAPTCHA at a time.

### The Continuing Challenge

Despite advances, OCR remains imperfect:
- Historical documents with unfamiliar fonts
- Handwritten manuscripts
- Low-quality scans
- Languages without training data
- Complex layouts

Each archive, each language, each historical period presents new challenges. The dream of universal access to recorded knowledge remains a work in progress.

---

## HANDS-ON EXERCISE: Building a Document Processing Pipeline

### Overview
Students will:
1. Preprocess document images
2. Apply OCR with Tesseract
3. Extract structured information
4. Handle multi-language documents

### Part 1: Basic OCR

```python
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Load and preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# OCR with confidence scores
def ocr_with_confidence(image_path):
    img = preprocess_image(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    results = []
    for i, word in enumerate(data['text']):
        if word.strip():
            results.append({
                'word': word,
                'confidence': data['conf'][i],
                'bbox': (data['left'][i], data['top'][i],
                        data['width'][i], data['height'][i])
            })
    return results

results = ocr_with_confidence('document.png')
high_conf = [r for r in results if r['confidence'] > 80]
```

### Part 2: Table Extraction

```python
import pdfplumber
import pandas as pd

def extract_tables_from_pdf(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            for j, table in enumerate(page_tables):
                if table and len(table) > 1:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    df['page'] = i + 1
                    df['table_num'] = j + 1
                    tables.append(df)
    return tables
```

### Part 3: Key Information Extraction

```python
import re

def extract_invoice_info(text):
    """Extract key fields from invoice text."""
    info = {}

    # Invoice number
    inv_match = re.search(r'Invoice\s*#?\s*:?\s*(\w+)', text, re.IGNORECASE)
    if inv_match:
        info['invoice_number'] = inv_match.group(1)

    # Date
    date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
    if date_match:
        info['date'] = date_match.group()

    # Total amount
    total_match = re.search(r'Total:?\s*\$?\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
    if total_match:
        info['total'] = float(total_match.group(1).replace(',', ''))

    return info
```

---

## Recommended Resources

### Libraries
- **Tesseract OCR**: https://github.com/tesseract-ocr/tesseract
- **EasyOCR**: Simple deep learning OCR
- **PaddleOCR**: Multilingual OCR
- **pdfplumber**: PDF extraction
- **LayoutParser**: Document layout analysis

### Cloud Services
- Google Cloud Vision API
- AWS Textract
- Azure Form Recognizer
- Google Document AI

### Papers
- "LayoutLMv3" - Microsoft's document understanding model
- "TrOCR" - Transformer OCR
- "Donut" - Document understanding transformer

---

*Module 4 explores OCR and Document AI—the technologies for extracting information from document images. From classical character recognition to modern deep learning systems that understand document structure, we learn to unlock the knowledge trapped in unstructured documents.*
