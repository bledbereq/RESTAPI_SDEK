from model import *
import json
from asyncs import *
from category import *


def get_category(id="NaN", name="NaN", content="NaN", description="NaN", name_en="NaN", images="NaN"):
    urls = json.loads(images)
    
    # Получаем изображения
    all_probabilities = model_threads(urls)
    
    # Вычисляем средние вероятности
    average_probabilities_cat1 = calculate_average_probabilities(all_probabilities, category_1)
    average_probabilities_cat2 = calculate_average_probabilities(all_probabilities, category_2)
    average_probabilities_cat3 = calculate_average_probabilities(all_probabilities, category_3)
    average_probabilities_cat4 = calculate_average_probabilities(all_probabilities, category_4)

    # #Получаем эмбеддинги для текста и изображений
    # images_list = model_threads_list(urls)
    # image_embeddings = get_image_embeddings(images_list)
    # text = name + content + description
    # text_embeddings = get_text_embeddings([content])

    # # Вычисляем косинусное сходство
    # similarity_scores = compute_similarity(image_embeddings, text_embeddings)

    # #Можно использовать среднее значение сходства или другое агрегирование
    # average_similarity = similarity_scores.mean()

    return {
        "id": id,
        "category_1": [category_1[average_probabilities_cat1[0][0]],average_probabilities_cat1[0][1]],
        "category_2": [category_2[average_probabilities_cat2[0][0]],average_probabilities_cat2[0][1]],
        "category_3": [category_3[average_probabilities_cat3[0][0]],average_probabilities_cat3[0][1]],
        "category_4": [category_4[average_probabilities_cat4[0][0]],average_probabilities_cat4[0][1]],
        "average_similarity": "95"
    }
