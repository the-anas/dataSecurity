# Set up
from transformers import TrainerCallback, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, hamming_loss
import wandb
import logging
import os
import ast
from transformers import DistilBertModel, PreTrainedModel, DistilBertConfig
import torch.nn as nn
from transformers import AdamW
import torch
import numpy as np
import pandas as pd


EPOCHS =  70
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
logging_dir = "./training_metrics_logs"
THRESHOLD = 0.3

# wandb set up
os.environ["WANDB_DIR"] = "/mnt/data/wandb_logs"
wandb.login()
run = wandb.init(
# Set the project where this run will be logged
project="Tracking DS Project", name= "70 Epochs attempt",
# Track hyperparameters and run metadata
config={
    "learning_rate": LEARNING_RATE,
    "Batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "Threshold": THRESHOLD
},
group = "Policy Change",
)

# set up logger
logging.basicConfig(
    filename=f"{logging_dir}/policy_change_logs.txt",  # Log file location
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(message)s",  # Log format
    filemode='w'
)   

logger = logging.getLogger()

# Custom callback to log metrics
class LoggingCallback(TrainerCallback):
    def __init__(self):
            self.logger = logging.getLogger(__name__)

    def on_evaluate(self, epoch, metrics):
        """
        Logs the evaluation metrics after each epoch.

        Args:
            model: The model being evaluated (optional, not used in this logger).
            eval_dataloader: The evaluation data loader (optional, not used in this logger).
            epoch: The current epoch number.
            metrics: A dictionary containing the evaluation metrics.
        """
        self.logger.info(f"--- Evaluation Metrics for Epoch {epoch} ---")
        # print(f"metrics: {metrics}")
        for task, task_metrics in metrics.items():
            # print(f"task: {task}")
            # print(f"task metrics: {task_metrics}")
            self.logger.info(f"Task: {task}")
            for metric_name, metric_value in task_metrics.items():
                self.logger.info(f"    {metric_name}: {metric_value:.4f}")
        self.logger.info("----------------------------------------")


# Short exploration with pandas
dataframe = pd.read_csv("multilabel_data/Policy_Change.csv")

# Preprocessing

dataframe['User Choice'] = dataframe['User Choice'].apply(ast.literal_eval) # convert string to list
dataframe['User Choice'] = dataframe['User Choice'].apply(lambda x: [float(i) for i in x]) # convert elements in list to float

dataframe['Change Type'] = dataframe['Change Type'].apply(ast.literal_eval) # convert string to list
dataframe['Change Type'] = dataframe['Change Type'].apply(lambda x: [float(i) for i in x]) # convert elements in list to float

dataframe['Notification Type'] = dataframe['Notification Type'].apply(ast.literal_eval) # convert string to list
dataframe['Notification Type'] = dataframe['Notification Type'].apply(lambda x: [float(i) for i in x]) # convert elements in list to float


# split data
train_df, eval_df = train_test_split(dataframe, test_size=0.2, random_state=42)

# Tokenize
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the texts in the DataFrame
inputs = tokenizer(list(train_df['segment']), padding=True, truncation=True, max_length=512, return_tensors="pt")
inputs_eval = tokenizer(list(eval_df['segment']), padding=True, truncation=True, max_length=512, return_tensors="pt")

# Convert labels to tensors
change_type_labels = torch.tensor(np.array(train_df['Change Type'].tolist(), dtype=np.float32))
user_choice_labels = torch.tensor(np.array(train_df['User Choice'].tolist(), dtype=np.float32))
notif_type_labels = torch.tensor(np.array(train_df['Notification Type'].tolist(), dtype=np.float32))

change_type_labels_eval = torch.tensor(np.array(eval_df['Change Type'].tolist(), dtype=np.float32))
user_choice_labels_eval = torch.tensor(np.array(eval_df['User Choice'].tolist(), dtype=np.float32))
notif_type_labels_eval = torch.tensor(np.array(eval_df['Notification Type'].tolist(), dtype=np.float32))

# Create a TensorDataset
train_dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], change_type_labels, user_choice_labels, notif_type_labels)
eval_dataset = TensorDataset(inputs_eval['input_ids'], inputs_eval['attention_mask'], change_type_labels_eval, user_choice_labels_eval, notif_type_labels_eval)

# Create a DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True) # YOU CAN EDIT THIS ARGUMENT LATER AS YOU WANT
eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle = True )


# Adjust model for multitask case
class DistilBertForMultiTask(PreTrainedModel):
    def __init__(self, config, num_labels_task1, num_labels_task2, num_labels_task3):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(0.3)

        # Output heads for each task
        self.classifier_task1 = nn.Linear(config.dim, num_labels_task1)
        self.classifier_task2 = nn.Linear(config.dim, num_labels_task2)
        self.classifier_task3 = nn.Linear(config.dim, num_labels_task3)

    def forward(self, input_ids, attention_mask=None, labels_task1=None, labels_task2=None, labels_task3=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0, :]) 

        logits_task1 = self.classifier_task1(pooled_output)
        logits_task2 = self.classifier_task2(pooled_output)
        logits_task3 = self.classifier_task3(pooled_output)

        return logits_task1, logits_task2, logits_task3

# Initialize the configuration manually if needed
config = DistilBertConfig.from_pretrained('distilbert-base-uncased')

# Now initialize the model with the configuration and number of labels for each task
model = DistilBertForMultiTask(config, num_labels_task1=5, num_labels_task2=5, num_labels_task3=6)

# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Loss functions for each task
loss_fn_task1 = nn.BCEWithLogitsLoss()
loss_fn_task2 = nn.BCEWithLogitsLoss()
loss_fn_task3 = nn.BCEWithLogitsLoss()

# Training loop with logs
# Move model to GPU or CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print(f"Training on device: {device}")

for epoch in range(EPOCHS):  # Number of epochs
    model.train()
    total_loss_epoch = 0  # To accumulate the loss for the epoch
    for batch_idx, batch in enumerate(train_dataloader):
        # Assuming the batch is a list of tensors, unpack them
        input_ids, attention_mask, labels_task1, labels_task2, labels_task3 = batch

        # Move tensors to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_task1 = labels_task1.to(device)
        labels_task2 = labels_task2.to(device)
        labels_task3 = labels_task3.to(device)

        # Forward pass
        logits_task1, logits_task2, logits_task3 = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute the loss for each task
        loss_task1 = loss_fn_task1(logits_task1, labels_task1)
        loss_task2 = loss_fn_task2(logits_task2, labels_task2)
        loss_task3 = loss_fn_task3(logits_task3, labels_task3)

        # Total loss
        total_loss = loss_task1 + loss_task2 + loss_task3

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss_epoch += total_loss.item()  # Accumulate loss for the epoch

        # Log progress every 10 batches
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {total_loss.item():.4f}")

    # Average loss for the epoch
    avg_loss_epoch = total_loss_epoch / len(train_dataloader)
    print(f"Epoch {epoch+1} completed. Average Loss for this epoch: {avg_loss_epoch:.4f}")

      # Evaluation code


    model.eval()
    with torch.no_grad():
        all_preds_task1 = []
        all_preds_task2 = []
        all_preds_task3 = []
        all_labels_task1 = []
        all_labels_task2 = []
        all_labels_task3 = []

        for batch in eval_dataloader:
            # Unpack the batch directly (since it's a list of tensors, not a dictionary)
            input_ids, attention_mask, labels_task1, labels_task2, labels_task3 = batch

            # Move tensors to the device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_task1 = labels_task1.to(device)
            labels_task2 = labels_task2.to(device)
            labels_task3 = labels_task3.to(device)

            # Forward pass
            logits_task1, logits_task2, logits_task3 = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get predictions by taking the class with the highest logit value
            preds_task1 = (logits_task1 > THRESHOLD).int()
            preds_task2 = (logits_task2 > THRESHOLD).int()
            preds_task3 = (logits_task3 > THRESHOLD).int()

            # Collect predictions and true labels for metrics computation
            all_preds_task1.extend(preds_task1.cpu().numpy())
            all_preds_task2.extend(preds_task2.cpu().numpy())
            all_preds_task3.extend(preds_task3.cpu().numpy())

            all_labels_task1.extend(labels_task1.cpu().numpy())
            all_labels_task2.extend(labels_task2.cpu().numpy())
            all_labels_task3.extend(labels_task3.cpu().numpy())

        # Compute metrics
        metrics ={
        'Change Type' : {
            'exact_match': np.mean(np.all(np.array(all_labels_task1) == np.array(all_preds_task1), axis=1)),
            'multilabel_accuracy': np.mean(( np.array(all_labels_task1) == np.array(all_preds_task1)).mean(axis=0)),
            'f1_macro': f1_score(all_labels_task1, all_preds_task1, average="macro"),
            'f1_micro': f1_score(all_labels_task1, all_preds_task1, average="micro"),
            'hamming_loss': hamming_loss(all_labels_task1, all_preds_task1),
            },
        'User Choice' : {
            'exact_match':  np.mean(np.all(np.array(all_labels_task2)== np.array(all_preds_task2), axis=1)),
            'multilabel_accuracy':  np.mean(( np.array(all_labels_task2) == np.array(all_preds_task2)).mean(axis=0)),
            'f1_macro': f1_score(all_labels_task2, all_preds_task2, average="macro"),
            'f1_micro': f1_score(all_labels_task2, all_preds_task2, average="micro"),
            'hamming_loss': hamming_loss(all_labels_task2, all_preds_task2),
            }, 
        'Notification Type' : {
            'exact_match':  np.mean(np.all(np.array(all_labels_task3)== np.array(all_preds_task3), axis=1)),
            'multilabel_accuracy':  np.mean(( np.array(all_labels_task3) == np.array(all_preds_task3)).mean(axis=0)),
            'f1_macro': f1_score(all_labels_task3, all_preds_task3, average="macro"),
            'f1_micro': f1_score(all_labels_task3, all_preds_task3, average="micro"),
            'hamming_loss': hamming_loss(all_labels_task3, all_preds_task3),
            }
        
        }

        wandb.log(
        {
            'exact_match_CT': metrics['Change Type']['exact_match'],
            'multilabel_accuracy_CT': metrics['Change Type']['multilabel_accuracy'],
            'f1_macro_CT': metrics['Change Type']['f1_macro'],
            'f1_micro_CT': metrics['Change Type']['f1_micro'],
            'hamming_loss_CT': metrics['Change Type']['hamming_loss'],
       
            'exact_match_UC': metrics['User Choice']['exact_match'],
            'multilabel_accuracy_UC': metrics['User Choice']['multilabel_accuracy'],
            'f1_macro_UC': metrics['User Choice']['f1_macro'],
            'f1_micro_UC': metrics['User Choice']['f1_micro'],
            'hamming_loss_UC': metrics['User Choice']['hamming_loss'],

            
            'exact_match_NT': metrics['Notification Type']['exact_match'],
            'multilabel_accuracy_NT': metrics['Notification Type']['multilabel_accuracy'],
            'f1_macro_NT': metrics['Notification Type']['f1_macro'],
            'f1_micro_NT': metrics['Notification Type']['f1_micro'],
            'hamming_loss_NT': metrics['Notification Type']['hamming_loss'],


    })

        # Call the logging callback
        callback = LoggingCallback()
        callback.on_evaluate(metrics=metrics, epoch=epoch)


# save model state
torch.save(model.state_dict(), '/mnt/data/models/policy_change/policy_change_model_state_dict.pth')

# save entire  model
torch.save(model, '/mnt/data/models/policy_change/policy_change_model_full.pth')
