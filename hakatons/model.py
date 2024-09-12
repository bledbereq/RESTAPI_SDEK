import clip
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO

# Инициализация модели CLIP и устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)

# Пакетная обработка изображений для получения эмбеддингов
def process_image_batch(images):
    with torch.no_grad():
        # Подготовка изображений
        processed_images = torch.stack([preprocess(img) for img in images]).to(device)
        
        # Получение эмбеддингов изображений
        image_embeddings = model.encode_image(processed_images)
        
        # Приведение эмбеддингов к CPU и преобразование в numpy
        image_embeddings = image_embeddings.cpu().numpy()
    
    return image_embeddings