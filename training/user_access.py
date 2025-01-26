from datasets import Dataset
import pandas as pd
from transformers import TrainerCallback, DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import wandb 
import logging
import torch
from transformers import DistilBertModel, PreTrainedModel, DistilBertConfig
import torch.nn as nn
from transformers import AdamW
import torch
import numpy as np

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
    filename=f"{logging_dir}/user_access_test_run.txt",  # Log file location
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(message)s",  # Log format
    filemode='w'
)

logger = logging.getLogger()

class LoggingCallback:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def on_evaluate(self, model, eval_dataloader, epoch, metrics):
        """
        Logs the evaluation metrics after each epoch.

        Args:
            model: The model being evaluated (optional, not used in this logger).
            eval_dataloader: The evaluation data loader (optional, not used in this logger).
            epoch: The current epoch number.
            metrics: A dictionary containing the evaluation metrics.
        """
        self.logger.info(f"--- Evaluation Metrics for Epoch {epoch+1} ---")
        for task, task_metrics in metrics.items():
            self.logger.info(f"Task: {task}")
            for metric_name, metric_value in task_metrics.items():
                self.logger.info(f"    {metric_name}: {metric_value:.4f}")
        self.logger.info("----------------------------------------")


# Short exploration with pandas
dataframe = pd.read_csv("User_Access_Edit_and_Deletion.csv")

# Preprocessing
# split data
train_df, eval_df = train_test_split(dataframe, test_size=0.2, random_state=42)


# Tokenize

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the texts in the DataFrame
inputs = tokenizer(list(train_df['segment']), padding=True, truncation=True, max_length=512, return_tensors="pt")
inputs_eval = tokenizer(list(eval_df['segment']), padding=True, truncation=True, max_length=512, return_tensors="pt")

# Convert labels to tensors
access_type_labels = torch.tensor(train_df['Access Type'].values)
access_scope_labels = torch.tensor(train_df['Access Scope'].values)

access_type_labels_eval = torch.tensor(eval_df['Access Type'].values)
access_scope_labels_eval = torch.tensor(eval_df['Access Scope'].values)

# Create a TensorDataset
train_dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], access_type_labels, access_scope_labels)
eval_dataset = TensorDataset(inputs_eval['input_ids'], inputs_eval['attention_mask'], access_type_labels_eval, access_scope_labels_eval)



# Create a DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True) # YOU CAN EDIT THIS ARGUMENT LATER AS YOU WANT
eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle = True )



class DistilBertForMultiTask(PreTrainedModel):
    def __init__(self, config, num_labels_task1, num_labels_task2):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)

        # Output heads for each task
        self.classifier_task1 = nn.Linear(config.dim, num_labels_task1)
        self.classifier_task2 = nn.Linear(config.dim, num_labels_task2)

    def forward(self, input_ids, attention_mask=None, labels_task1=None, labels_task2=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]  # Take <CLS> token hidden state

        # Classification heads
        logits_task1 = self.classifier_task1(pooled_output)
        logits_task2 = self.classifier_task2(pooled_output)

        # Add sigmoid for multi-label classification
        probs_task1 = torch.sigmoid(logits_task1)
        probs_task2 = torch.sigmoid(logits_task2)

        return probs_task1, probs_task2

# Initialize the configuration manually if needed
config = DistilBertConfig.from_pretrained('distilbert-base-uncased')

# Now initialize the model with the configuration and number of labels for each task
model = DistilBertForMultiTask(config, num_labels_task1=9, num_labels_task2=6)




# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Replace loss functions for multi-label classification
loss_fn_task1 = torch.nn.BCEWithLogitsLoss()
loss_fn_task2 = torch.nn.BCEWithLogitsLoss()


# Training loop with logs
# Move model to GPU or CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print(f"Training on device: {device}")


# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss_epoch = 0
    for batch_idx, batch in enumerate(train_dataloader):
        input_ids, attention_mask, labels_task1, labels_task2 = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_task1 = labels_task1.to(device).float()  # Convert labels to float for BCEWithLogitsLoss
        labels_task2 = labels_task2.to(device).float()

        logits_task1, logits_task2 = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute losses
        loss_task1 = loss_fn_task1(logits_task1, labels_task1)
        loss_task2 = loss_fn_task2(logits_task2, labels_task2)

        total_loss = loss_task1 + loss_task2

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss_epoch += total_loss.item()

    avg_loss_epoch = total_loss_epoch / len(train_dataloader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss_epoch:.4f}")

    # Evaluation


    model.eval()
    with torch.no_grad():
        all_preds_task1 = []
        all_preds_task2 = []
        all_labels_task1 = []
        all_labels_task2 = []

        for batch in eval_dataloader:
            input_ids, attention_mask, labels_task1, labels_task2 = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_task1 = labels_task1.to(device)
            labels_task2 = labels_task2.to(device)

            logits_task1, logits_task2 = model(input_ids=input_ids, attention_mask=attention_mask)

            # Apply sigmoid and thresholding for multi-label predictions
            preds_task1 = (logits_task1 > 0.5).int()  # Threshold at 0.5
            preds_task2 = (logits_task2 > 0.5).int()

            all_preds_task1.extend(preds_task1.cpu().numpy())
            all_preds_task2.extend(preds_task2.cpu().numpy())
            all_labels_task1.extend(labels_task1.cpu().numpy())
            all_labels_task2.extend(labels_task2.cpu().numpy())

         # Compute metrics for both tasks
        metrics_task1 = {
            'exact_match': np.mean(np.all(all_labels_task1 == all_preds_task1, axis=1)),
            'multilabel_accuracy': np.mean((all_labels_task1 == all_preds_task1).mean(axis=0)),
            'f1_macro': f1_score(all_labels_task1, all_preds_task1, average="macro"),
            'f1_micro': f1_score(all_labels_task1, all_preds_task1, average="micro"),
            'hamming_loss': hamming_loss(all_labels_task1, all_preds_task1),
        }
        metrics_task2 = {
            'exact_match':  np.mean(np.all(all_labels_task2, all_preds_task2, axis=1)),
            'multilabel_accuracy':  np.mean(np.all(all_labels_task2, all_preds_task2, axis=0)),
            'f1_macro': f1_score(all_labels_task2, all_preds_task2, average="macro"),
            'f1_micro': f1_score(all_labels_task2, all_preds_task2, average="micro"),
            'hamming_loss': hamming_loss(all_labels_task2, all_preds_task2),
        }

        print(f"Task 1 Metrics: {metrics_task1}")
        print(f"Task 2 Metrics: {metrics_task2}")


        # Call the logging callback
        callback = LoggingCallback()
        callback.on_evaluate(None, None, None, metrics={**metrics_task1, **metrics_task2})




# Save model after training and evaluation
# save model state
torch.save(model.state_dict(), 'user_access_model_state_dict.pth')

# save entire  model
torch.save(model, 'user_access_model_full.pth')

