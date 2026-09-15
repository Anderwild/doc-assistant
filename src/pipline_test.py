import torch
import torch.nn as nn
from vectorize import get_prepared_data
from model import TextClassifier

def main():
    print("Тестовый старт")

    x,y = get_prepared_data()

    in_features = x.shape[1]
    num_classes = int(y.max()) + 1

    model = TextClassifier(in_features=in_features, num_classes=num_classes)
    print("\nСтруктура нейросети:")
    print(model)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=le-3)


    logits = model(x)

    loss = loss_fn(logits, y)
    print("\nЗАМЕР, Значение ошибки (Loss) ДО этого шага обучения:", loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    new_logits = model(x)
    new_loss = loss_fn(new_logits, y)
    print("[ЗАМЕР] Значение ошибки (Loss) ПОСЛЕ шага обучения:", new_loss.item())

    if new_loss.item() < loss.item():
        print("Веса скорректированы, ошибка модели пошла вниз")

if __name__ == "__main__":
    main()