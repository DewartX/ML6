import torch
import torch.nn as nn
import torch.optim as optim

#Данные
torch.manual_seed(15)

n_samples = 1000

# главный признак
main_values = torch.FloatTensor(n_samples, 1).uniform_(-100, 100)

#Шум
noise = torch.FloatTensor(n_samples, 4).uniform_(1, 1)

# вход
X = torch.cat([main_values, noise], dim=1)
y = (main_values > 0).float()


#Модель
model = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),

    nn.Linear(16, 12),
    nn.Tanh(),

    nn.Linear(12, 8),
    nn.LeakyReLU(),

    nn.Linear(8, 1)
)


criterion = nn.BCEWithLogitsLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1)


#Обучение
for epoch in range(200):

    optimizer.zero_grad()

    outputs = model(X)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()

    # accuracy
    with torch.no_grad():
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        acc = (preds == y).float().mean().item() * 100

    if epoch % 50 == 0 or epoch == 199:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Acc: {acc:.2f}%")


#Тест
print("\nТест:")

test_main = torch.FloatTensor(20, 1).uniform_(-100, 100)
test_noise = torch.randn(20, 4)
test_X = torch.cat([test_main, test_noise], dim=1)

with torch.no_grad():
    test_out = model(test_X)
    test_prob = torch.sigmoid(test_out)
    test_pred = (test_prob > 0.5).float()

    for i in range(20):
        val = test_main[i].item()
        prob = test_prob[i].item()
        label = "положительное" if test_pred[i].item() == 1 else "отрицательное"

        print(f"{val:8.2f} → {label:12} (prob {prob:.4f})")