from datasets import Dataset
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

# Short exploration with pandas
dataframe = pd.read_csv("User_Access_Edit_and_Deletion.csv")

# Preprocessing
# split data
train_df, eval_df = train_test_split(dataframe, test_size=0.2, random_state=42)

# Encode labels

# Initialize a label encoder for each target column
access_type_encoder = LabelEncoder()
access_scope_encoder = LabelEncoder()

# encode training dataset
train_df['Access Type'] = access_type_encoder.fit_transform(train_df['Access Type'])
train_df['Access Scope'] = access_scope_encoder.fit_transform(train_df['Access Scope'])

# Encode eval dataset
eval_df['Access Type'] = access_type_encoder.fit_transform(eval_df['Access Type'])
eval_df['Access Scope'] = access_scope_encoder.fit_transform(eval_df['Access Scope'])


# Tokenize
import torch

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

# Adjust model for multitask case
from transformers import DistilBertModel, PreTrainedModel, DistilBertConfig
import torch.nn as nn

class DistilBertForMultiTask(PreTrainedModel):
    def __init__(self, config, num_labels_task1, num_labels_task2):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)

        # Output heads for each task
        self.classifier_task1 = nn.Linear(config.dim, num_labels_task1)
        self.classifier_task2 = nn.Linear(config.dim, num_labels_task2)

    def forward(self, input_ids, attention_mask=None, labels_task1=None, labels_task2=None, labels_task3=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]  # Take <CLS> token hidden state

        logits_task1 = self.classifier_task1(pooled_output)
        logits_task2 = self.classifier_task2(pooled_output)

        return logits_task1, logits_task2

# Initialize the configuration manually if needed
config = DistilBertConfig.from_pretrained('distilbert-base-uncased')

# Now initialize the model with the configuration and number of labels for each task
model = DistilBertForMultiTask(config, num_labels_task1=9, num_labels_task2=6)


from transformers import AdamW
import torch

# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Loss functions for each task
loss_fn_task1 = torch.nn.CrossEntropyLoss()
loss_fn_task2 = torch.nn.CrossEntropyLoss()


# Training loop with logs
# Move model to GPU or CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print(f"Training on device: {device}")

EPOCHS = 10

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

        # Log progress every 10 batches
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{15}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {total_loss.item():.4f}")

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

        for batch in eval_subset_dataloader:
            # Unpack the batch directly (since it's a list of tensors, not a dictionary)
            input_ids, attention_mask, labels_task1, labels_task2 = batch

            # Move tensors to the device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_task1 = labels_task1.to(device)
            labels_task2 = labels_task2.to(device)

            # Forward pass
            logits_task1, logits_task2 = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get predictions by taking the class with the highest logit value
            preds_task1 = logits_task1.argmax(dim=-1)
            preds_task2 = logits_task2.argmax(dim=-1)

            # Collect predictions and true labels for metrics computation
            all_preds_task1.extend(preds_task1.cpu().numpy())
            all_preds_task2.extend(preds_task2.cpu().numpy())

            all_labels_task1.extend(labels_task1.cpu().numpy())
            all_labels_task2.extend(labels_task2.cpu().numpy())

        # Compute metrics for each task
        accuracy_task1 = accuracy_score(all_labels_task1, all_preds_task1)
        accuracy_task2 = accuracy_score(all_labels_task2, all_preds_task2)

        f1_task1 = f1_score(all_labels_task1, all_preds_task1, average='weighted')
        f1_task2 = f1_score(all_labels_task2, all_preds_task2, average='weighted')

        print(f"Accuracy Task 1: {accuracy_task1:.4f}, F1 Task 1: {f1_task1:.4f}")
        print(f"Accuracy Task 2: {accuracy_task2:.4f}, F1 Task 2: {f1_task2:.4f}")

# Save model after training and evaluation
# save model state
torch.save(model.state_dict(), 'user_access_model_state_dict.pth')

# save entire  model
torch.save(model, 'user_access_model_full.pth')

