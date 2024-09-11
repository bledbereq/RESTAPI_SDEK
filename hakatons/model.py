# import clip
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
from collections import defaultdict

# Инициализация модели CLIP и устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)

# Пакетная обработка изображений
def process_image_batch(images):
    labels = ['Обувь', 'Одежда, обувь', 'Одежда', 'Спорт и отдых', 'Товары с быстрой доставкой', 'Фанатская атрибутика']
    
    with torch.no_grad():
        # Подготовка изображений и текстов
        processed_images = torch.stack([preprocess(img) for img in images]).to(device)
        text = clip.tokenize(labels).to(device)
        
        # Обработка пакета изображений
        logits_per_image, _ = model(processed_images, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Преобразование результатов
    results = [sorted([(label, float(prob)) for label, prob in zip(labels, probs[i])], key=lambda x: x[1], reverse=True)
               for i in range(len(images))]
    return results
