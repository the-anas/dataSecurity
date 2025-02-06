import wandb
import pickle
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
import numpy as np
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score, jaccard_score
import logging
import os

# HYPERPAREMETERS and constants
LEARNING_RATE = 2e-5
EPOCHS = 80 
BATCH_SIZE = 16
logging_dir = "./training_metrics_logs"


# wandb set up
os.environ["WANDB_DIR"] = "/mnt/data/wandb_logs"  # Set the directory for WandB logs
wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="Tracking DS Project", name= "80 Epochs Model",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "Batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    },
group="First model"
)

# set up logger
logging.basicConfig(
    filename=f"{logging_dir}/main_file_logs.txt",  # Log file location
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(message)s",  # Log format
    filemode = 'w'
)

logger = logging.getLogger()

# Custom callback to log metrics
class LoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            epoch = state.epoch
            eval_loss = metrics.get('eval_loss') # YOU CAN ADD    ,None as default here to avoid issues
            exact_match = metrics.get('eval_exact_match')
            multilabel_accuracy = metrics.get('eval_multilabel_accuracy')
            f1_macro = metrics.get('eval_f1_macro')
            f1_micro = metrics.get('eval_f1_micor')
            precision_macro = metrics.get('eval_precision_macro')
            recall_macro = metrics.get('eval_recall_macro')
            jaccard_macro = metrics.get('eval_jaccard_macro')
            precision_micro = metrics.get('eval_precision_micro')
            recall_micro = metrics.get('eval_recall_micro')
            jaccard_micro = metrics.get('eval_jaccard_micro')
            hamming_loss = metrics.get('eval_hamming_loss')

            log_message = f"""Epoch: {epoch}, Eval Loss: {eval_loss}, 
            Exact Match: {exact_match}, Multilabel Accuracy: {multilabel_accuracy}, F1 Macro: {f1_macro}, F1 Micro: {f1_micro}, Hamming Loss: {hamming_loss}, 
            Precision Macro: {precision_macro}, Recall Macro: {recall_macro}, Jaccard Macro: {jaccard_macro}, Precision Micro: {precision_micro}, 
            Recall Micro: {recall_micro}, Jaccard Micro: {jaccard_micro}"""
            logger.info(log_message)  # Log metrics to the file

        return control


# Loading and processing data
with open("./seperated_categories/agg_data.pkl",'rb') as dataframe_file:
  opened_dataframe = pickle.load(dataframe_file)

opened_dataframe['label'] = opened_dataframe['label'].apply(lambda x: [float(i) for i in x])


train_df, test_df = train_test_split(opened_dataframe, test_size=0.3)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the text data
def tokenize_function(examples):
    return tokenizer(examples['segment'], padding="max_length", truncation=True)

train_dataset_tokenized = train_dataset.map(tokenize_function, batched=True)
test_dataset_tokenized = test_dataset.map(tokenize_function, batched=True)

# Load DistilBERT model for sequence classification with 3 labels (multi-label classification)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                            num_labels=12,
                                                            problem_type="multi_label_classification",
                                                            )

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments

training_args = TrainingArguments(
    output_dir="/mnt/data/30_epochs_model",
    eval_strategy="epoch",               # Evaluate after every epoch
    report_to="wandb",
    num_train_epochs= EPOCHS,                 # Total number of epochs
    #save_strategy="epoch",               # Save the best model at the end of each epoch
    #load_best_model_at_end=True,         # Load the best model based on validation loss
    learning_rate= LEARNING_RATE,                  # Learning rate for the optimizer
    weight_decay=0.01,                   # Weight decay for regularization
    per_device_train_batch_size= BATCH_SIZE,      # Training batch size
    per_device_eval_batch_size= BATCH_SIZE,       # Evaluation batch size
    gradient_accumulation_steps=4,       # Accumulate gradients for fewer backward passes
    #logging_strategy="epoch",            # Log metrics at intervals of steps
    #log_level="info",                    # Log level (e.g., "info" or "error")
    #log_level_replica="warning",        # Adjust logs for distributed training replicas
    #logging_dir= logging_dir   ,            # Directory for storing logs
    #logging_steps = 100,
    #metric_for_best_model="eval_multilabel_accuracy",  # Metric to track the best model
    fp16=True,  # Enable mixed precision

    )



def compute_metrics(eval_pred):
    """
    Compute metrics for multilabel classification.
    :param eval_pred: Tuple (predictions, labels)
    :return: Dictionary with metric values
    """
    logits, labels = eval_pred
    
    # Apply sigmoid to logits for multilabel classification
    probs = 1 / (1 + np.exp(-logits))
    # Convert probabilities to binary predictions (0 or 1)
    preds = (probs > 0.5).astype(int)

    # Exact match ratio: Proportion of samples with all labels correct
    exact_match = np.all(preds == labels, axis=1).mean()

    # Multilabel accuracy: Average accuracy across all labels
    multilabel_accuracy = (preds == labels).mean()

    # F1 Score (macro and micro)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")

    # Precision, Recall, and Jaccard index (macro and micro)
    precision_macro = precision_score(labels, preds, average="macro")
    recall_macro = recall_score(labels, preds, average="macro")
    jaccard_macro = jaccard_score(labels, preds, average="macro")

    precision_micro = precision_score(labels, preds, average="micro")
    recall_micro = recall_score(labels, preds, average="micro")
    jaccard_micro = jaccard_score(labels, preds, average="micro")

    # Hamming loss
    hamming = hamming_loss(labels, preds)

    return {
        "exact_match": exact_match,
        "multilabel_accuracy": multilabel_accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "jaccard_macro": jaccard_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "jaccard_micro": jaccard_micro,
        "hamming_loss": hamming,
    }


# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tokenized,
    eval_dataset=test_dataset_tokenized,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks = [LoggingCallback]


)

# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()
#print(eval_results)

# Save the model
model.save_pretrained("/mnt/data/models/first_model_80")
tokenizer.save_pretrained("/mnt/data/models/first_model_80")


