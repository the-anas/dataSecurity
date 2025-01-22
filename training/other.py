import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback, DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import logging
import os

EPOCHS = 25
LEARNING_RATE = 5e-5   
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
logging_dir = "./training_metrics_logs"

# wandb set up
os.environ["WANDB_DIR"] = "/mnt/data/wandb_logs"  # Set the directory for WandB logs
wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="Annotating Privacy Policies", name= "Other model: expoloring why model won't generalize, 7 epochs",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "Batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    },
#    notes = "Lowered learning rate and batch size, increased epochs, and increased weight decay"
)

# set up logger
logging.basicConfig(
    filename=f"{logging_dir}/other_test_run2.txt",  # Log file location
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(message)s",  # Log format
    filemode='w'
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
dataframe = pd.read_csv("processed_data/Other.csv")


#rename column for huggingface API
dataframe.rename(columns={'Other Type': 'labels'}, inplace=True)

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
train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
eval_tokenized_dataset = eval_dataset.map(tokenize_function, batched=True)

# Prepare DataLoader
# small_train_dataset = train_tokenized_dataset.shuffle(seed=42).select(range(200))  # Small subset for example
# small_eval_dataset = eval_tokenized_dataset.shuffle(seed=42).select(range(100))

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)



training_args = TrainingArguments(
    output_dir='/mnt/data/other_results',  # Directory where models and logs will be saved
    evaluation_strategy="epoch",  # Perform evaluation at the end of each epoch
    save_strategy="epoch",        # Save checkpoints at the end of each epoch
    save_total_limit=1,           # Keep only the best checkpoint (based on accuracy)
    load_best_model_at_end=True,  # Load the best model when training is complete
    metric_for_best_model="eval_accuracy",  # Use accuracy to determine the best model
    greater_is_better=True,      # Higher accuracy means better model
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay= WEIGHT_DECAY,
    logging_steps=100,
    report_to="wandb",            # Log metrics to W&B
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
    train_dataset=train_tokenized_dataset, 
    eval_dataset=eval_tokenized_dataset,  
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [LoggingCallback]
)

# Train the model
trainer.train()


eval_results = trainer.evaluate()
print(eval_results)

# Save the model
model.save_pretrained("/mnt/data/other_model")
tokenizer.save_pretrained("/mnt/data/other_model")
