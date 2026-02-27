from models import MultiLayerPerceptron, CNNModel, load_dataset, test_transform
import torch
import os

valid_path = input("Enter the path to the validation dataset: ")

if not os.path.exists(valid_path):
    print("The specified path does not exist.")
    exit(1)

valid_set, valid_loader = load_dataset(valid_path, test_transform)

loadable_models = []

mlp_model = MultiLayerPerceptron()
mlp_model.load('checkpoints/mlp-original.pt')

loadable_models.append(('Multi-Layer Perceptron', mlp_model))

cnn_model = CNNModel()
cnn_model.load('checkpoints/cnn-original.pt')

loadable_models.append(('Convolutional Neural Network', cnn_model))

for sparsity in [20, 50, 80]:
    cnn_sparse_model = CNNModel()
    cnn_sparse_model.load(f'checkpoints/cnn-pruned-{sparsity}%.pt')
    loadable_models.append((f'CNN Sparse {sparsity}%', cnn_sparse_model))

    cnn_finetuned_model = CNNModel()
    cnn_finetuned_model.load(f'checkpoints/cnn-fine-tuned-{sparsity}%.pt')
    loadable_models.append((f'CNN Fine-Tuned {sparsity}%', cnn_finetuned_model))

print("-----------------------------------")

while True:
    print("Available Models for Evaluation:")

    id = 0
    for model_name, model in loadable_models:
        print(f"[{id}] {model_name}")
        id += 1

    print("-----------------------------------")

    selected_id = int(input("Select a model by entering its ID: "))
    selected_model_name, selected_model = loadable_models[selected_id]

    print(f"Evaluating Model: {selected_model_name}")

    selected_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = selected_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f'Accuracy of the model on {valid_path}: {accuracy:.2f}%')
    print("-----------------------------------")