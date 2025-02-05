from transformers import DistilBertModel, PreTrainedModel
import torch.nn as nn

class PolicyChange(PreTrainedModel):
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