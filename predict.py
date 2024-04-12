import argparse
import json
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['model_arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img)
    return img_tensor

def predict(image_path, model, topk=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.exp(output)
        top_probabilities, top_indices = probabilities.topk(topk)
        top_probabilities = top_probabilities.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_classes = [idx_to_class[idx] for idx in top_indices]
        return top_probabilities, top_classes

def print_predictions(args):
    print("Image Path:", args.image_filepath)
    print("Model Path:", args.model_filepath)
    print("Category Names JSON Path:", args.category_names_json_filepath)
    print("Top K:", args.top_k)
    print("Using GPU:", args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")

    model = load_checkpoint(args.model_filepath)
    top_probabilities, top_classes = predict(args.image_filepath, model, args.top_k)

    if args.category_names_json_filepath:
        with open(args.category_names_json_filepath, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name[c] for c in top_classes]

    for i in range(min(args.top_k, len(top_classes))):
        print(f"{top_classes[i]}: {top_probabilities[i]:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(dest='image_filepath', help="Path of image (single)")
    parser.add_argument(dest='model_filepath', help="Checkpoint (Modal) file")
    
    parser.add_argument('--category_names_json_filepath', dest='category_names_json_filepath', help="Make use of a category to real name mapping", default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k', help="Top K", default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', help="GPU enable (CUDA if nvidia)", action='store_true', default=True)
    args = parser.parse_args()
    print_predictions(args)
