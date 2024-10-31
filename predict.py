# Importing essential libraries
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
import argparse

# Setting up command line argument parsing
parser = argparse.ArgumentParser(description="Predict flower species using a trained model.")

# Adding arguments to the parser for various inputs
parser.add_argument('--json_file', type=str, default='cat_to_name.json', help='Custom JSON file for category names.')
parser.add_argument('--test_file', type=str, default='flowers/train/43/image_02364.jpg', help='Image file for prediction.')
parser.add_argument('--checkpoint_file', type=str, default='checkpoint.pth', help='Checkpoint file for loading the model.')
parser.add_argument('--topk', type=int, default=5, help='Number of top predictions to return.')
parser.add_argument('--gpu', default='gpu', type=str, help='Specify model execution: CPU or GPU.')

# Parsing the command line inputs into variables
args = parser.parse_args()

json_file = args.json_file
test_file = args.test_file
checkpoint_file = args.checkpoint_file
topk = args.topk
gpu = args.gpu

# Loading the category names from the specified JSON file
with open(json_file, 'r') as f:
    category_mapping = json.load(f)

# Function to load the model from a checkpoint file
def load_model(checkpoint_file=checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    architecture = checkpoint['arch']
    learning_rate = checkpoint['lr']
    hidden_layer_units = checkpoint['hidden_layer']
    gpu_mode = checkpoint['gpu']
    epochs_trained = checkpoint['epochs']
    dropout_rate = checkpoint['dropout']
    classifier_structure = checkpoint['classifier']
    model_state = checkpoint['state_dict']
    class_index_mapping = checkpoint['class_to_idx']

    # Selecting the model architecture based on the checkpoint data
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained=True)

    model.classifier = classifier_structure
    model.class_to_idx = class_index_mapping
    model.load_state_dict(model_state)

    # Freezing model parameters
    for param in model.parameters():
        param.requires_grad = False

    return model

# Loading the model using the defined function
loaded_model = load_model()

# Function to preprocess the input image for the model
def process_image(image_path=test_file):
    """Resize, crop, and normalize the input image, returning it as a Numpy array."""
    img = Image.open(image_path)

    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    processed_image = transformation(img).float()
    return processed_image

# Function to predict the class of the input image using the trained model
def predict(image_path=test_file, model=loaded_model, topk=topk, gpu=gpu):
    """Predict the class of an image using a trained deep learning model."""
    
    # Processing the image
    image = process_image(image_path)
    image = image.unsqueeze_(0)

    # Move the model to GPU if specified
    if gpu == 'gpu':
        model.to('cuda:0')

    # Generate predictions
    with torch.no_grad():
        if gpu == 'gpu':
            image = image.to('cuda')
        output = model(image)

    # Calculate probabilities
    probabilities = F.softmax(output.data, dim=1)

    # Get the top k predictions
    top_probabilities, top_indices = probabilities.topk(topk)
    top_probabilities = top_probabilities.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    # Mapping index to class
    index_to_class = {value: key for key, value in model.class_to_idx.items()}
    predicted_classes = [index_to_class[idx] for idx in top_indices]

    return top_probabilities, predicted_classes

# Executing the prediction function and printing results
probs, classes = predict(test_file, loaded_model, topk)
print(probs)
print(classes)
