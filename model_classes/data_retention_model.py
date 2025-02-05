import torch.nn as nn
from transformers import DistilBertModel

class DataRetention(nn.Module):
    def __init__(self, num_labels_task1, num_labels_task2, num_labels_task3):
        super(DataRetention, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        
        # Multi-label classification heads
        self.classifier_task1 = nn.Linear(self.distilbert.config.hidden_size, num_labels_task1)
        self.classifier_task2 = nn.Linear(self.distilbert.config.hidden_size, num_labels_task2)
        self.classifier_task3 = nn.Linear(self.distilbert.config.hidden_size, num_labels_task3)
        
    def forward(self, input_ids, attention_mask):
        # Get the hidden states from DistilBERT
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0, :]) 

        # Compute logits for each task
        logits_task1 = self.classifier_task1(pooled_output)
        logits_task2 = self.classifier_task2(pooled_output)
        logits_task3 = self.classifier_task3(pooled_output)

        return logits_task1, logits_task2, logits_task3  