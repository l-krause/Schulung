import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from my_neural_net import NeuralNet


config = {"features": (4, 16, 3),
          "batch_size": 16,
          "lr": 0.01,
          "num_epochs": 100}

# Gerät wählen (GPU falls vorhanden)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Iris-Datensatz laden
iris = load_iris()
X = iris.data  # 4 Merkmale
y = iris.target  # 3 Klassen

# Merkmale skalieren
scaler = StandardScaler()
X = scaler.fit_transform(X)

# In Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# In Torch-Tensors umwandeln
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Dataloader erstellen
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True) # könnt einen eigenen schreiben aber relativ kompliziert

# Modell initialisieren
model = NeuralNet(
    in_features=config["features"][0],
    hidden_features=config["features"][1],
    out_features=config["features"][2],
    device=device
).to(device)

# Loss und Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

# Training
for epoch in range(config["num_epochs"]):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor.to(device))
    _, predicted = torch.max(outputs, 1)
    for i, prediction in enumerate(predicted):
        print(iris.target_names[prediction])
    accuracy = (predicted.cpu() == y_test_tensor).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")
    from sklearn.metrics import classification_report
    print(classification_report(y_test_tensor, predicted.cpu(), target_names=iris.target_names))