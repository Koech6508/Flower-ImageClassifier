# Essential Libraries Import
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
import argparse

# Load Category Names from JSON File
def load_category_names(json_file='cat_to_name.json'):
    with open(json_file, 'r') as file:
        return json.load(file)

# Load the category names
category_names = load_category_names()

# Command Line Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a flower classification model.')

    parser.add_argument('--data_dir', type=str, default="./flowers/", help='Directory for dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoint.pth', help='Path to save the checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture to use')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=1024, help='Number of hidden units in the classifier')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout probability')

    return parser.parse_args()

# Parse arguments
args = parse_arguments()

# Prepare Dataset Paths
train_path = f"{args.data_dir}/train"
valid_path = f"{args.data_dir}/valid"
test_path = f"{args.data_dir}/test"

# Data Transformation
def get_transforms():
    training_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return training_transforms, validation_transforms

# Get transformations
train_transforms, val_transforms = get_transforms()

# Load Image Datasets
def load_datasets():
    train_data = datasets.ImageFolder(train_path, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_path, transform=val_transforms)
    test_data = datasets.ImageFolder(test_path, transform=val_transforms)
    
    return train_data, valid_data, test_data

train_data, valid_data, test_data = load_datasets()

# Data Loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

# Define the Classifier Model
def build_classifier(architecture, hidden_units, dropout_rate):
    if architecture == 'vgg16':
        base_model = models.vgg16(pretrained=True)
        input_size = 25088
    elif architecture == 'densenet121':
        base_model = models.densenet121(pretrained=True)
        input_size = 1024
    elif architecture == 'alexnet':
        base_model = models.alexnet(pretrained=True)
        input_size = 9216

    # Freeze parameters
    for param in base_model.parameters():
        param.requires_grad = False

    # Create a custom classifier
    classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(dropout_rate)),
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 256)),
        ('output', nn.Linear(256, 102)),
        ('softmax', nn.LogSoftmax(dim=1))
    ]))

    base_model.classifier = classifier
    return base_model

# Initialize Model
model = build_classifier(args.arch, args.hidden_units, args.dropout_rate)

# Define Loss Function and Optimizer
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Training Function
def train_network(model, train_loader, valid_loader, criterion, optimizer, epochs, use_gpu):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        print_interval = 40

        for images, labels in train_loader:
            steps += 1
            if use_gpu:
                images, labels = images.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Validate every few steps
            if steps % print_interval == 0:
                validate_model(model, valid_loader, criterion)

        print(f'Epoch: {epoch+1}/{epochs}, Training Loss: {total_loss/steps:.3f}')

# Validation Function
def validate_model(model, valid_loader, criterion):
    model.eval()
    validation_loss = 0
    accuracy = 0

    with torch.no_grad():
        for val_images, val_labels in valid_loader:
            if args.gpu:
                val_images, val_labels = val_images.to('cuda'), val_labels.to('cuda')

            val_outputs = model(val_images)
            validation_loss += criterion(val_outputs, val_labels).item()
            ps = torch.exp(val_outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == val_labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f'Validation Loss: {validation_loss/len(valid_loader):.3f}, Validation Accuracy: {accuracy/len(valid_loader):.3f}')
    model.train()

# Execute Training
train_network(model, train_loader, valid_loader, loss_function, optimizer, args.epochs, args.gpu)

# Testing Function
def evaluate_model(test_loader, model, use_gpu):
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in test_loader:
            if use_gpu:
                images, labels = images.to('cuda'), labels.to('cuda')

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = (correct_predictions / total_predictions) * 100
    print(f'Accuracy of the model on test dataset: {accuracy:.2f}%')

# Evaluate Model Performance
evaluate_model(test_loader, model, args.gpu)

# Save the Model Checkpoint
def save_checkpoint(model, save_path):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'architecture': args.arch,
        'learning_rate': args.learning_rate,
        'hidden_units': args.hidden_units,
        'gpu': args.gpu,
        'epochs': args.epochs,
        'dropout_rate': args.dropout_rate,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, save_path)

# Save the model checkpoint
save_checkpoint(model, args.save_dir)
