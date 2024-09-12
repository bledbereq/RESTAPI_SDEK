from model import *
import json
from asyncs import *

def get_category(id="NaN", name="NaN", content="NaN", description="NaN", name_en="NaN", images="NaN"):
    # Получаем URL изображений
    urls = json.loads(images)
    
    # Асинхронная обработка изображений для получения эмбеддингов
    image_embeddings = model_threads(urls)

    # Преобразуем эмбеддинги из NumPy в список
    image_embeddings_list = [embedding.tolist() for embedding in image_embeddings]

    return {
        "id": id,
        "category_1": image_embeddings_list,  # Эмбеддинги в формате списка
        "category_2": 2,  
        "category_3": 3,  
        "category_4": 4,  
    }
