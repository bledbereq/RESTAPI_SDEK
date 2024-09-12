from model import *
import json
from asyncs import *


def get_category(id="NaN", name="NaN", content="NaN", description="NaN", name_en="NaN", images="NaN"):
    # Получаем URL изображений
    urls = json.loads(images)
    
    # Обрабатываем изображения один раз
    all_probabilities = model_threads(urls)
    
    # Вычисляем средние вероятности для каждой метки в каждой категории
    average_probabilities_cat1 = calculate_average_probabilities(all_probabilities, category_1)
    average_probabilities_cat2 = calculate_average_probabilities(all_probabilities, category_2)
    average_probabilities_cat3 = calculate_average_probabilities(all_probabilities, category_3)
    average_probabilities_cat4 = calculate_average_probabilities(all_probabilities, category_4)

    # Возвращаем результат
    return {
        "id": id,
        "category_1": average_probabilities_cat1[0], 
        "category_2": average_probabilities_cat2[0],  
        "category_3": average_probabilities_cat3[0],  
        "category_4": average_probabilities_cat4[0],  
    }

