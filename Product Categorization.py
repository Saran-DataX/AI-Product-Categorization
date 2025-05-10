import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import Dataset
import re
from nltk.stem import WordNetLemmatizer
import nltk
from torch.optim import AdamW  # Corrected import to torch.optim
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, get_linear_schedule_with_warmup

# Ensure required NLTK resources are available
nltk.download('wordnet')
nltk.download('punkt')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a fake product dataset using Faker
def generate_fake_data(num_samples=10000):
    categories = ["Electronics", "Clothing", "Books", "Home & Kitchen", "Beauty"]
    data = []

    for _ in range(num_samples):
        title = "Sample Product " + str(np.random.randint(1000, 9999))
        description = "This is a description for product " + str(np.random.randint(1000, 9999))
        category = np.random.choice(categories)
        data.append([title, description, category])

    return pd.DataFrame(data, columns=["title", "description", "category"])

# Generate the dataset
df = generate_fake_data(10000)

# Combine the title and description into one text field
df["text"] = df["title"] + " " + df["description"]

# Encode the categories
category_map = {category: idx for idx, category in enumerate(df["category"].unique())}
df["label"] = df["category"].map(category_map)

# Split into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"])

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(category_map))

# Tokenize the texts
def clean_text(text):
    """Text cleaning function to preprocess and remove unnecessary words/special chars"""
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetical characters
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Apply text cleaning
train_df['text'] = train_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)

# Tokenizer function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)

train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Define optimizer and scheduler
def get_optimizer_scheduler(model, train_loader, lr=5e-5, warmup_steps=500):
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    
    total_steps = len(train_loader) * training_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    return optimizer, scheduler

# Simplified TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',              # output directory
    num_train_epochs=5,                  # number of epochs (increase for fine-tuning)
    per_device_train_batch_size=32,      # batch size for training
    per_device_eval_batch_size=64,       # batch size for evaluation
    warmup_steps=500,                    # warmup steps for learning rate scheduler
    weight_decay=0.01,                   # strength of weight decay
    logging_dir='./logs',                # directory for storing logs
    logging_steps=200,                   # log every 200 steps
    save_steps=1000,                     # save model every 1000 steps         # evaluate every X steps              # save model every X steps
    load_best_model_at_end=True,         # load best model when finished training
    metric_for_best_model="accuracy",    # use accuracy to find the best model
)

# Trainer
trainer = Trainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.predictions.argmax(axis=-1), p.label_ids),
        'f1': f1_score(p.label_ids, p.predictions.argmax(axis=-1), average='weighted'),
        'precision': precision_score(p.label_ids, p.predictions.argmax(axis=-1), average='weighted'),
        'recall': recall_score(p.label_ids, p.predictions.argmax(axis=-1), average='weighted')
    },
    optimizers=get_optimizer_scheduler(model, train_dataset)[0],
    lr_scheduler=get_optimizer_scheduler(model, train_dataset)[1],
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained('./distilbert-product-model')
tokenizer.save_pretrained('./distilbert-product-model')

# Evaluate the model
results = trainer.evaluate()
print(f"Evaluation Results: {results}")

# Save evaluation results to a CSV
eval_df = pd.DataFrame(results, index=[0])
eval_df.to_csv('./evaluation_results.csv', index=False)

# Optional: Save checkpoint
trainer.save_model('./final_model_checkpoint')
tokenizer.save_pretrained('./final_model_checkpoint')
