import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from finetuned_neural_net import FineTunedNeuralNet

# 1. Daten laden
dataset = load_dataset("imdb")

# 2. Tokenizer laden und anwenden
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

tokenized_datasets = dataset.map(preprocess, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# 3. DataLoader erstellen
train_loader = DataLoader(tokenized_datasets["train"].shuffle(seed=42).select(range(2000)), batch_size=8)
test_loader = DataLoader(tokenized_datasets["test"].shuffle(seed=42).select(range(1000)), batch_size=8)


# 4. Modell laden
model = FineTunedNeuralNet()

### Freeze parameters
for param in model.pretrained_model.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5. Optimierer
optimizer = Adam(model.parameters(), lr=2e-5)

criterion = torch.nn.BCELoss()

# 6. Training
model.train()
for epoch in range(2):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])
        loss = criterion(outputs.squeeze(), batch["label"].to(torch.float))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())

# 7. Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        correct += (preds == batch["label"]).sum().item()
        total += batch["label"].size(0)

print(f"Accuracy: {correct / total:.2%}")