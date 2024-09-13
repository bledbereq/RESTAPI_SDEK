import clip
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
from collections import defaultdict
from category import *
import numpy as np
from scipy.spatial.distance import cosine


# Инициализация модели CLIP и устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)

def process_image_batch(images):
    all_labels = list(category_1.keys()) + list(category_2.keys()) + list(category_3.keys()) + list(category_4.keys())
    
    with torch.no_grad():
        # Подготовка изображений и текстов
        processed_images = torch.stack([preprocess(img) for img in images]).to(device)
        text = clip.tokenize(all_labels).to(device)
        
        # Обработка изображений
        logits_per_image, _ = model(processed_images, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    results = [
        sorted([(label, float(prob)) for label, prob in zip(all_labels, probs[i])], key=lambda x: x[1], reverse=True)
        for i in range(len(images))
    ]
    
    return results


def get_image_embeddings(images):
    with torch.no_grad():
        # Подготовка изображений
        processed_images = torch.stack([preprocess(img) for img in images]).to(device)
        
        # Получение эмбеддингов изображений
        image_features = model.encode_image(processed_images)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # Нормализация
        return image_features.cpu().numpy()


def get_text_embeddings(texts):
    with torch.no_grad():
        # Подготовка текста
        text_features = model.encode_text(clip.tokenize(texts).to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)  # Нормализация
        return text_features.cpu().numpy()

def compute_similarity(image_embeddings, text_embeddings):
    similarities = []
    for img_emb in image_embeddings:
        img_similarities = []
        for txt_emb in text_embeddings:
            sim = cosine(img_emb, txt_emb) 
            img_similarities.append(sim)
        similarities.append(img_similarities)
    return np.array(similarities)
