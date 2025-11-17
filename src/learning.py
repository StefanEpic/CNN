import torch
from src.metrics import print_model_metrics


# Код обучения одной эпохи
def train_one_epoch(epoch_index, model, optimizer, criterion, train_loader, device):
    running_loss = 0.
    total_samples = 0.
    for batch_index, data in enumerate(train_loader):
        # Извлечение батча
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # Обнуление градиентов
        optimizer.zero_grad()
        # Прямое распространение
        outputs = model(inputs)
        # Подсчёт ошибки
        loss = criterion(outputs, labels)
        # Обратное распространение
        loss.backward()
        # Обновление весов
        optimizer.step()
        # Суммирование ошибки
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
    # Возвращаем среднюю ошибку за всю эпоху
    avg_epoch_loss = running_loss / total_samples
    return avg_epoch_loss


# Цикл обучения
def run_train(model, criterion, optimizer, train_loader, val_loader, test_loader, class_names, epochs, device, best_vloss = 1e5):
    for epoch in range(epochs):
        print(f'Эпоха {epoch}')
        # Перевод модели в режим обучения
        model.train(True)
        # Эпоха обучения
        avg_loss = train_one_epoch(epoch, model, optimizer, criterion, train_loader, device)
        # Перевод модели в режим валидации
        model.eval()
        running_vloss = 0.0
        total_vsamples = 0
        # Валидация
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item() * vinputs.size(0)
                total_vsamples += vinputs.size(0)

        avg_vloss = running_vloss / total_vsamples
        # Сохранение лучшей модели
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'meds_classifier_{epoch}.pt'
            torch.save(model.state_dict(), model_path)

        print(f'В конце эпохи ошибка train {avg_loss:.4f}, ошибка val {avg_vloss:.4f}')
        print_model_metrics(model, test_loader, device, class_names)
