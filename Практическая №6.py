import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# Настройка устройства (GPU если есть)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    #Для Google Colab
print(f'Используемое устройство: {device}')

## ПОДГОТОВКА ДАННЫХ ДЛЯ НЕЙРОСЕТИ
# Читаем базу данных
data = pd.read_csv('reviews_preprocessed.csv')
reviews = data.processed.values

# Выделяем все слова
all_words = ' '.join(data.processed.values).split()

# Создаем словарь
counter = Counter(all_words)
vocabulary = sorted(counter, key=counter.get, reverse=True)

# Задаем всем словам ID
int2word = dict(enumerate(vocabulary,1))
int2word[0] = 'PAD'
word2int = {word: id for id, word in int2word.items()}

# Отзывы после преобразования
reviews_enc = [[word2int[word] for word in review.split()] for review in reviews]

# Ячейки для отзывов
sequence_length = 256
# Создаём пустую матрицу нужного размера
reviews_padding = np.zeros((len(reviews_enc), sequence_length), dtype=int)
# Заполняем матрицу
for i, row in enumerate(reviews_enc):
    row = row[:sequence_length]
    reviews_padding[i, :len(row)] = row

label = data.label.to_numpy()

# Подготовим тестовые данные и данные для обучения
train_len = 0.6
test_len = 0.5

# Обучение
train_last_index = int(len(reviews_padding)*train_len)

train_x, remainder_x = reviews_padding[:train_last_index], reviews_padding[train_last_index:]
train_y, remainder_y = label[:train_last_index], label[train_last_index:]

# Тест
test_last_index = int(len(remainder_x)*test_len)

test_x = remainder_x[:test_last_index]
test_y = remainder_y[:test_last_index]

check_x = remainder_x[test_last_index:]
check_y = remainder_y[test_last_index:]

# Переводим в тензоры
train_dataset = TensorDataset(torch.from_numpy(train_x.copy()), torch.from_numpy(train_y.copy()))
test_dataset = TensorDataset(torch.from_numpy(test_x.copy()), torch.from_numpy(test_y.copy()))
check_dataset = TensorDataset(torch.from_numpy(check_x.copy()), torch.from_numpy(check_y.copy()))

# Data Loader
batch_size = 128
train_loader = DataLoader(train_dataset, shuffle=True,batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=True,batch_size=batch_size)
check_loader = DataLoader(check_dataset, shuffle=True,batch_size=batch_size)
#________________________________________________________________________________________________________

# Модель:
class TextModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, hidden_size, num_layers, dropout):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = dropout, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        out = inp.long()
        out = self.embedding(out)
        out = self.lstm(out)[0]
        out = out[:,-1,:]
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)

model = TextModel(len(word2int),256,128,4,0.25)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model_path = './best_model.pth'


def get_accuracy(out, target):
    predicted = torch.Tensor([1 if i else 0 for i in out > 0.5])
    equals = predicted == target
    return torch.mean(equals.type(torch.FloatTensor)).item()
'''
test_lost_min = torch.inf
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    train_accuracy = 0
    train_loss = 0

    for i, (current_reviews, target) in enumerate(train_loader):
        print('Trained (epoch %d): %d out of %d' %((epoch+1), i, len(train_loader)))
        optimizer.zero_grad()
        out = model(current_reviews)
        train_accuracy += get_accuracy(out.detach(), target)
        loss = criterion(out.squeeze(), target.float())
        train_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)        # Обрезаем градиент
        optimizer.step()

    print('Train accuracy: %f%%' % (train_accuracy * 100 / len(train_loader)))
    print('Train loss: %f' % (train_loss / len(train_loader)))

    model.eval()
    test_accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for i, (current_reviews, target) in enumerate(test_loader):
            print('Test (epoch %d): %d out of %d' %((epoch+1), i, len(test_loader)))
            out = model(current_reviews)
            test_accuracy += get_accuracy(out, target)
            loss = criterion(out.squeeze(), target.float())
            test_loss += loss.item()
        print('Validation accuracy: %f%%' % (test_accuracy * 100 / len(test_loader)))
        print('Validation loss: %f' % (test_loss / len(test_loader)))

    test_loss = test_loss / len(test_loader)
    if test_loss < test_lost_min:
        test_lost_min = test_loss
        torch.save(model.state_dict(), model_path)
'''

# Подгружаем модель из файла и проводим тест
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           # Так как обучал модель в Google Colab, веса нужно
model.load_state_dict(torch.load(model_path, map_location=device))              # веса нужно переоборудовать с GPU на CPU
model.to(device)
model.eval()

with torch.no_grad():
    check_accuracy = 0
    for current_reviews, target in check_loader:
        current_reviews = current_reviews.to(device)
        target = target.to(device)

        out = model(current_reviews)
        check_accuracy += get_accuracy(out, target)
    print('Accuracy: %f%%' % (check_accuracy *100 / len(check_loader)))
