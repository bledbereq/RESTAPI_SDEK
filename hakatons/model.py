import clip
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
from collections import defaultdict
from category import *

# Инициализация модели CLIP и устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)

def process_image_batch(images):
    all_labels = list(category_1.values()) + list(category_2.values()) + list(category_3.values()) + list(category_4.values())
    
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
