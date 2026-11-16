import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from vectorize import get_prepared_data
from model import TextClassifier

def main():
    print("Инициализация обучения:")

    # 1. Сначала разбиваем СЫРОЙ датасет
    df = pd.read_csv(DATA_PATH)  # добавьте в начало файла:
    # import pandas as pd
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # from sklearn.preprocessing import LabelEncoder
    # from pathlib import Path
    # BASE_DIR = Path(__file__).resolve().parent
    # DATA_PATH = BASE_DIR / ".." / "data" / "processed" / "clean.csv"

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        df["text"], df["topic"], test_size=0.2, random_state=42
    )

    # 2. TF-IDF обучаем ТОЛЬКО на train
    vectorizer = TfidfVectorizer(max_features=1000)
    X_tr = vectorizer.fit_transform(X_train_raw).toarray()
    X_te = vectorizer.transform(X_test_raw).toarray()

    # 3. LabelEncoder тоже только на train
    label_encoder = LabelEncoder()
    y_tr = label_encoder.fit_transform(y_train_raw)
    y_te = label_encoder.transform(y_test_raw)

    print(f"Обнаружено классов (тем): {len(label_encoder.classes_)}")

    X_train_tensor = torch.tensor(X_tr, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_tr, dtype=torch.long)
    X_test_tensor = torch.tensor(X_te, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_te, dtype=torch.long)

    in_features = X_train_tensor.shape[1]
    num_classes = len(label_encoder.classes_)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    # ... дальше цикл эпох остаётся без изменений


    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()


            total_train_loss += loss.item()
            _, predicted = torch.max(logits, dim = 1)
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)
        

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train



        model.eval()
        total_test_loss = 0 
        correct_test = 0
        total_test = 0 

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                logits = model(batch_X)
                loss = loss_fn(logits, batch_y)

                total_test_loss += loss.item()
                _, predicted = torch.max(logits, dim = 1)
                correct_test += (predicted == batch_y).sum().item()
                total_test += batch_y.size(0)
        

        avg_test_loss = total_test_loss / len(test_loader)
        test_acc = correct_test / total_test

        print(f"Эпоха [{epoch+1:02d} / {num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}")
              
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            os.makedirs('models', exist_ok=True)

            checkpoint = {
                'epoch' : epoch + 1,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'loss' : best_test_loss
            }
            torch.save(checkpoint, 'models/best_text_classifier.pth')
            print(f"--> Чекпоинт сохранен.")
    
    print("\n=== Процесс успешно завершен ===")


if __name__ == "__main__":
    main()