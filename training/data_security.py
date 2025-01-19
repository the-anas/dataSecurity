import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import wandb


EPOCHS = 2
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
logging_dir = "./training_metrics_logs"


# wandb set up
wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="Annotating Privacy Policies", name= "Test run, not tracking anything",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "Batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    },
)

# set up logger
logging.basicConfig(
    filename=f"{logging_dir}/data_security_test_run.txt",  # Log file location
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(message)s",  # Log format
)

logger = logging.getLogger()

# Custom callback to log metrics
class LoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print(f"Metrics received in the callback: {metrics}")
        if metrics:
            epoch = state.epoch
            eval_loss = metrics.get('eval_loss') # YOU CAN ADD    ,None as default here to avoid issues
            accuracy = metrics.get('eval_accuracy')
            precision = metrics.get('eval_precision')
            recall = metrics.get('eval_recall')
            f1 = metrics.get('eval_f1')
            log_message = f"Epoch: {epoch}, Eval Loss: {eval_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}"
            logger.info(log_message)  # Log metrics to the file

        return control




# Short exploration with pandas
dataframe = pd.read_csv("./processed_data/Data_Security.csv")

#rename column for huggingface API
dataframe.rename(columns={'Security Measure': 'labels'}, inplace=True)

# Encode labels and split data
train_df, eval_df = train_test_split(dataframe, test_size=0.2, random_state=42)

# Initialize a label encoder for each target column
encoder = LabelEncoder()

# encode training dataset
train_df['labels'] = encoder.fit_transform(train_df['labels'])

# Encode eval dataset
eval_df['labels'] = encoder.fit_transform(eval_df['labels'])

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

# # Prepare DataLoader
# small_train_dataset = train_tokenized_datasets.shuffle(seed=42).select(range(200))  # Small subset for example
# small_eval_dataset = eval_tokenized_datasets.shuffle(seed=42).select(range(100))

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
    train_dataset=train_tokenized_datasets, # MAKE SURE YOU ARE USING CORRECT DATASET
    eval_dataset=eval_tokenized_datasets,  # MAKE SURE YOU ARE USING CORRECT DATASET
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [LoggingCallback]
)

# Train the model
trainer.train()

# Evaluate the model

eval_results = trainer.evaluate()
print(eval_results)

# Save the model
model.save_pretrained(f"./data_security_model_{EPOCHS}_epochs")
tokenizer.save_pretrained("./data_security_model")
