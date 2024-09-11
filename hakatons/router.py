from model import *
import json
from asyncs import *

def get_category(id="NaN", name="NaN", content="NaN", description="NaN", name_en="NaN", images="NaN"):
    # Получаем URL изображений
    urls = json.loads(images)
    
    # Асинхронная обработка изображений
    all_probabilities = model_threads(urls)

    # Вычисляем средние вероятности для каждой метки
    average_probabilities = calculate_average_probabilities(all_probabilities)

    # Возвращаем результат
    return {
        "id": id,
        "category_1": average_probabilities, 
        "category_2": 2,  
        "category_3": 3,  
        "category_4": 4,  
    }