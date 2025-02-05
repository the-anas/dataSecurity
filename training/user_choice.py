import pandas as pd
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, hamming_loss
import wandb
import logging
from transformers import AdamW
import torch
import numpy as np
from transformers import DistilBertModel, PreTrainedModel, DistilBertConfig
import torch.nn as nn
import os
import ast

EPOCHS = 70
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
logging_dir = "./training_metrics_logs"
THRESHOLD = 0.3


# wandb set up
os.environ["WANDB_DIR"] = "/mnt/data/wandb_logs"  # Set the directory for WandB logs
wandb.login()
run = wandb.init(
# Set the project where this run will be logged
project="Tracking DS Project", name= "70 epoch attempt",
# Track hyperparameters and run metadata
config={
    "learning_rate": LEARNING_RATE,
    #"learning_rate_AS": LR_AS,
    "Batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "threshold": THRESHOLD
},
# notes="Training user_choice, lower training rate with lower threshold",
group = "User Choice"
)

# set up logger
logging.basicConfig(
    filename=f"{logging_dir}/user_choice_logs.txt",  # Log file location
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(message)s",  # Log format
    filemode='w'
)

logger = logging.getLogger()

class LoggingCallback:
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
dataframe = pd.read_csv("updated_multilabel_data/User_Choice2.csv")

# Preprocessing

dataframe['Choice Type'] = dataframe['Choice Type'].apply(ast.literal_eval) # convert string to list
dataframe['Choice Type'] = dataframe['Choice Type'].apply(lambda x: [float(i) for i in x]) # convert elements in list to float

dataframe['Choice Scope'] = dataframe['Choice Scope'].apply(ast.literal_eval) # convert string to list
dataframe['Choice Scope'] = dataframe['Choice Scope'].apply(lambda x: [float(i) for i in x]) # convert elements in list to float

# split data
train_df, eval_df = train_test_split(dataframe, test_size=0.2, random_state=42)

# Tokenize

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the texts in the DataFrame
inputs = tokenizer(list(train_df['segment']), padding=True, truncation=True, max_length=512, return_tensors="pt")
inputs_eval = tokenizer(list(eval_df['segment']), padding=True, truncation=True, max_length=512, return_tensors="pt")

# Convert labels to proper numpy array and then to tensors
choice_type_labels = torch.tensor(np.array(train_df['Choice Type'].tolist(), dtype=np.float32))
choice_scope_labels = torch.tensor(np.array(train_df['Choice Scope'].tolist(), dtype=np.float32))

choice_type_labels_eval = torch.tensor(np.array(eval_df['Choice Type'].tolist(), dtype=np.float32))
choice_scope_labels_eval = torch.tensor(np.array(eval_df['Choice Scope'].tolist(), dtype=np.float32))


# Create a TensorDataset
train_dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], choice_type_labels, choice_scope_labels)
eval_dataset = TensorDataset(inputs_eval['input_ids'], inputs_eval['attention_mask'], choice_type_labels_eval, choice_scope_labels_eval)



# Create a DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True) # YOU CAN EDIT THIS ARGUMENT LATER AS YOU WANT
eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle = True )

# Adjust model for multitask case
class DistilBertForMultiTask(PreTrainedModel):
    def __init__(self, config, num_labels_task1, num_labels_task2):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(0.3) # Dropout layer

        # Output heads for each task
        self.classifier_task1 = nn.Linear(config.dim, num_labels_task1)
        self.classifier_task2 = nn.Linear(config.dim, num_labels_task2)

    def forward(self, input_ids, attention_mask=None, labels_task1=None, labels_task2=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0, :]) 

        logits_task1 = self.classifier_task1(pooled_output)
        logits_task2 = self.classifier_task2(pooled_output)

        return logits_task1, logits_task2 

# Initialize the configuration manually if needed
config = DistilBertConfig.from_pretrained('distilbert-base-uncased')

# Now initialize the model with the configuration and number of labels for each task
model = DistilBertForMultiTask(config, num_labels_task1=9, num_labels_task2=5)


# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE) # 5e-5

# Loss functions for each task
loss_fn_task1 = torch.nn.BCEWithLogitsLoss()
loss_fn_task2 = torch.nn.BCEWithLogitsLoss()

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
        input_ids, attention_mask, labels_task1, labels_task2 = batch

        # Move tensors to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_task1 = labels_task1.to(device)
        labels_task2 = labels_task2.to(device)


        # Forward pass
        logits_task1, logits_task2 = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute the loss for each task
        loss_task1 = loss_fn_task1(logits_task1, labels_task1)
        loss_task2 = loss_fn_task2(logits_task2, labels_task2)


        # Total loss
        total_loss = loss_task1 + loss_task2

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss_epoch += total_loss.item()  # Accumulate loss for the epoch

        # Log progress every 100 batches
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
        all_labels_task1 = []
        all_labels_task2 = []

        for batch in eval_dataloader:
            # Unpack the batch directly (since it's a list of tensors, not a dictionary)
            input_ids, attention_mask, labels_task1, labels_task2 = batch

            # Move tensors to the device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_task1 = labels_task1.to(device)
            labels_task2 = labels_task2.to(device)

            # Forward pass
            logits_task1, logits_task2 = model(input_ids=input_ids, attention_mask=attention_mask)

            # Apply sigmoid and thresholding for multi-label predictions
            preds_task1 = (logits_task1 > THRESHOLD).int()  # Threshold at 0.5
            preds_task2 = (logits_task2 > THRESHOLD).int()

            # Collect predictions and true labels for metrics computation
            all_preds_task1.extend(preds_task1.cpu().numpy())
            all_preds_task2.extend(preds_task2.cpu().numpy())
            all_labels_task1.extend(labels_task1.cpu().numpy())
            all_labels_task2.extend(labels_task2.cpu().numpy())

         # Compute metrics
            metrics ={
        'Choice Type' : {
            'exact_match': np.mean(np.all(np.array(all_labels_task1) == np.array(all_preds_task1), axis=1)),
            'multilabel_accuracy': np.mean(( np.array(all_labels_task1) == np.array(all_preds_task1)).mean(axis=0)),
            'f1_macro': f1_score(all_labels_task1, all_preds_task1, average="macro"),
            'f1_micro': f1_score(all_labels_task1, all_preds_task1, average="micro"),
            'hamming_loss': hamming_loss(all_labels_task1, all_preds_task1),
            },
        'Choice Scope' : {
            'exact_match':  np.mean(np.all(np.array(all_labels_task2)== np.array(all_preds_task2), axis=1)),
            'multilabel_accuracy':  np.mean(( np.array(all_labels_task2) == np.array(all_preds_task2)).mean(axis=0)),
            'f1_macro': f1_score(all_labels_task2, all_preds_task2, average="macro"),
            'f1_micro': f1_score(all_labels_task2, all_preds_task2, average="micro"),
            'hamming_loss': hamming_loss(all_labels_task2, all_preds_task2),
            }
        }

        wandb.log(
        {
            'exact_match_CT': metrics['Choice Type']['exact_match'],
            'multilabel_accuracy_CT': metrics['Choice Type']['multilabel_accuracy'],
            'f1_macro_CT': metrics['Choice Type']['f1_macro'],
            'f1_micro_CT': metrics['Choice Type']['f1_micro'],
            'hamming_loss_CT': metrics['Choice Type']['hamming_loss'],
       
            'exact_match_CS': metrics['Choice Scope']['exact_match'],
            'multilabel_accuracy_CS': metrics['Choice Scope']['multilabel_accuracy'],
            'f1_macro_CS': metrics['Choice Scope']['f1_macro'],
            'f1_micro_CS': metrics['Choice Scope']['f1_micro'],
            'hamming_loss_CS': metrics['Choice Scope']['hamming_loss'],
    })

        # Call the logging callback
        callback = LoggingCallback()
        callback.on_evaluate(metrics=metrics, epoch=epoch)


# Save model after training and evaluation
# save model state
torch.save(model.state_dict(), '/mnt/data/models/user_choice/user_choice_model_state_dict.pth')

# save entire  model
torch.save(model, '/mnt/data/models/user_choice/user_choice_model_full.pth')
