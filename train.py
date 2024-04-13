import argparse
import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
import json

def load_and_transform_data(data_dir):
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        "train": datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=data_transforms["valid"]),
        "test": datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms["test"])
    }

    train_data_loader = DataLoader(image_datasets["train"], batch_size=64, shuffle=True)
    valid_data_loader = DataLoader(image_datasets["valid"], batch_size=64)
    
    class_to_idx = image_datasets['train'].class_to_idx
    
    return train_data_loader, valid_data_loader, class_to_idx

def build_model(model_arch, hidden_units):
    model = getattr(models, model_arch)(pretrained=True)
    
    if 'vgg' in model_arch:
        in_features = 25088
    elif 'resnet' in model_arch:
        in_features = model.fc.in_features
    elif 'alexnet' in model_arch:
        in_features = model.classifier[1].in_features
    else:
        raise ValueError("Model architecture not supported")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    if 'vgg' in model_arch or 'alexnet' in model_arch:
        model.classifier = classifier
    elif 'resnet' in model_arch:
        model.fc = classifier
    else:
        raise ValueError("Model architecture not supported")

    return model

def train(model, train_data_loader, valid_data_loader, optimizer, criterion, device, epochs):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        else:
            validation_loss = 0.0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_data_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels)
                    validation_loss += batch_loss.item()

                    ps = torch.exp(outputs)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/len(train_data_loader):.3f}.. "
                  f"Validation loss: {validation_loss/len(valid_data_loader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(valid_data_loader):.3f}")

def save_checkpoint(model, optimizer, class_to_idx, model_arch, save_dir='.', filename='checkpoint.pth'):
    checkpoint = {
        'model_arch': model_arch,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx
    }
    if 'vgg' in model_arch or 'alexnet' in model_arch:
        checkpoint['classifier'] = model.classifier
    elif 'resnet' in model_arch:
        checkpoint['fc'] = model.fc
    else:
        raise ValueError("Model architecture not supported")

    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network on a flower dataset')
    parser.add_argument('data_dir', help='Directory containing the flower dataset')
    parser.add_argument('--save_dir', default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', default='vgg16', help='Model architecture (default: vgg16)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training (default: 0.001)')
    parser.add_argument('--hidden_units', type=int, default=2048, help='Number of hidden units in the classifier (default: 2048)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training (default: 5)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Training on device: {device}")

    train_loader, valid_loader, class_to_idx = load_and_transform_data(args.data_dir)
    model = build_model(args.arch, args.hidden_units)
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate) if 'vgg' in args.arch or 'alexnet' in args.arch else optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss()

    train(model, train_loader, valid_loader, optimizer, criterion, device, args.epochs)

    save_checkpoint(model, optimizer, class_to_idx, args.arch, save_dir=args.save_dir)
