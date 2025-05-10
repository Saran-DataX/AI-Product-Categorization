# 🧠 AI-Powered Product Categorization using NLP & BERT

This repository presents an end-to-end solution for **automated product categorization** using advanced Natural Language Processing (NLP) techniques. Leveraging **BERT** and **TensorFlow**, the system classifies product titles into predefined e-commerce categories with high accuracy, achieving over **93% validation accuracy** on a balanced synthetic dataset.

---

## 📌 Objective

Automated product categorization is crucial for enhancing user experience, improving search relevance, and optimizing inventory management in e-commerce. This project fine-tunes a transformer-based language model (BERT) to classify product titles into 10 distinct categories.

---

## 📊 Dataset

A synthetically generated and balanced dataset containing **10,000 product titles**, equally distributed across 10 categories:

- 📱 Mobile Phones  
- 💻 Laptops  
- 📺 Televisions  
- 🎧 Headphones  
- 📷 Cameras  
- 👟 Footwear  
- 👔 Clothing  
- 🪑 Furniture  
- 🍳 Kitchen Appliances  
- 📚 Books  

> File: `products_dataset.csv`

---

## 🧠 Model Overview

- **Model Architecture:** `TFBertForSequenceClassification` (based on BERT base uncased)
- **Tokenizer:** Hugging Face `BertTokenizer`
- **Framework:** TensorFlow 2.x
- **Optimizer:** Hugging Face `create_optimizer` (AdamW)
- **Loss Function:** `SparseCategoricalCrossentropy`
- **Accuracy Achieved:** ~93% on validation data

---

## 🛠️ Project Structure

```bash
├── products_dataset.csv                # Balanced synthetic dataset
├── product_categorization.ipynb        # End-to-end Google Colab notebook
└── README.md                           # Project documentation
