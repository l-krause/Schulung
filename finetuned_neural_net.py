import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import torch

class FineTunedNeuralNet(nn.Module):
    
    def __init__(self, in_features=1, out_features=1, device="cpu"):
        super().__init__()
        self.pretrained_model =  AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, dtype=torch.float).to(device)

    def forward(self, input_ids, attention_mask, labels):
        pred = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        pred = self.fc(pred.logits)
        pred = torch.sigmoid(pred)
        return pred
