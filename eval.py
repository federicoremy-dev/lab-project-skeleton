import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.custom_net import CustomNet

def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / len(test_loader)
    test_accuracy = 100. * correct / total
    print(f'Test Loss: {test_loss:.6f} Acc: {test_accuracy:.2f}%')
    return test_accuracy

if __name__ == "__main__":
    DATA_DIR = "dataset/tiny-imagenet-200"
    CHECKPOINT = "checkpoints/model_epoch10.pth"

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(f"{DATA_DIR}/val", transform=transform)
    val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = CustomNet().cuda()
    model.load_state_dict(torch.load(CHECKPOINT))
    criterion = nn.CrossEntropyLoss()

    evaluate(model, val_loader, criterion)