import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

EPOCHS = 10
LEARNING_RATE = 2e-5

# Short exploration with pandas
dataframe = pd.read_csv("./processed_data/Data_Security.csv")

# Encode labels and split data
train_df, eval_df = train_test_split(dataframe, test_size=0.2, random_state=42)

# Initialize a label encoder for each target column
encoder = LabelEncoder()

# encode training dataset
train_df['Security Measure'] = encoder.fit_transform(train_df['Security Measure'])

# Encode eval dataset
eval_df['Security Measure'] = encoder.fit_transform(eval_df['Security Measure'])

# transform to huggingface dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)


# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['segment'], padding="max_length", truncation=True)

# Tokenize the dataset
train_tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
eval_tokenized_datasets = eval_dataset.map(tokenize_function, batched=True)

# Prepare DataLoader
small_train_dataset = train_tokenized_datasets.shuffle(seed=42).select(range(200))  # Small subset for example
small_eval_dataset = eval_tokenized_datasets.shuffle(seed=42).select(range(100))

train_dataloader = DataLoader(train_tokenized_datasets, batch_size=8)
test_dataloader = DataLoader(eval_tokenized_datasets, batch_size=8)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=10)


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs= EPOCHS,
    weight_decay=0.01,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)

    # Calculate precision, recall, and F1-score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, # MAKE SURE YOU ARE USING CORRECT DATASET
    eval_dataset=eval_dataset,  # MAKE SURE YOU ARE USING CORRECT DATASET
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model

eval_results = trainer.evaluate()
print(eval_results)

# Save the model
model.save_pretrained(f"./data_security_model_{EPOCHS}_epochs")
tokenizer.save_pretrained("./data_security_model")
