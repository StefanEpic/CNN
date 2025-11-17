import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def print_model_metrics(model, dataloader, device, class_names):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )

    if accuracy > 0.75:
        print(f"Общая точность (accuracy): {accuracy:.2%}")
        print(f"Точность (precision): {precision:.2%}")
        print(f"Полнота (recall): {recall:.2%}")
        print(f"Cреднее между точностью и полнотой (F1-Score): {f1:.2%}")
        
        cm = confusion_matrix(all_labels, all_predictions)
        errors_per_class = [cm[i].sum() - cm[i, i] for i in range(len(class_names))]

        # Топ-5 классов с ошибками
        error_indices = np.argsort(errors_per_class)[-5:][::-1]
        print("\nТоп-5 классов с ошибками:")
        for idx in error_indices:
            if errors_per_class[idx] > 0:
                print(f"  {class_names[idx]}: {errors_per_class[idx]} ошибок")

        # Классы без ошибок
        perfect_classes = [class_names[i] for i in range(len(class_names))
                           if errors_per_class[i] == 0 and cm[i].sum() > 0]
        if perfect_classes:
            print(f"\nКлассы без ошибок: {', '.join(perfect_classes)}")