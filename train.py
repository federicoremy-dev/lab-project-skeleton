import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.custom_net import CustomNet
import wandb

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

    # log su wandb
    wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "epoch": epoch})

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total
    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')

    # log su wandb
    wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})
    return val_accuracy

if __name__ == "__main__":
    # --- Configurazione ---
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    DATA_DIR = "dataset/tiny-imagenet-200"

    # --- Wandb init ---
    wandb.init(project="faimdl-lab3", config={
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE
    })

    # --- Transforms ---
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Dataset ---
    train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform)
    val_dataset   = datasets.ImageFolder(f"{DATA_DIR}/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- Modello ---
    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training loop ---
    for epoch in range(1, NUM_EPOCHS + 1):
        train(epoch, model, train_loader, criterion, optimizer)
        validate(model, val_loader, criterion)

        # salva checkpoint
        torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch}.pth")

    wandb.finish()