from model import *
import json
from asyncs import *
from category import *

from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import numpy as np

# Загрузка модели и токенизатора TinyBERT
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
text_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")

# Функция для получения эмбеддингов с помощью TinyBERT
def get_text_embeddings_tinybert(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = text_model(**inputs)
    # Используем [CLS] токен для получения эмбеддингов
    embeddings = outputs.last_hidden_state[:, 0, :]
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # Нормализация
    return embeddings.cpu().numpy()

# Обновленная версия get_category с использованием TinyBERT
def get_category(id="NaN", name="NaN", content="NaN", description="NaN", name_en="NaN", images="NaN"):
    urls = json.loads(images)
    
    # Получаем изображения
    all_probabilities = model_threads(urls)
    
    # Вычисляем средние вероятности
    average_probabilities_cat1 = calculate_average_probabilities(all_probabilities, category_1)
    average_probabilities_cat2 = calculate_average_probabilities(all_probabilities, category_2)
    average_probabilities_cat3 = calculate_average_probabilities(all_probabilities, category_3)
    average_probabilities_cat4 = calculate_average_probabilities(all_probabilities, category_4)

    # Получаем эмбеддинги для текста и изображений
    images_list = model_threads_list(urls)
    image_embeddings = get_image_embeddings(images_list)
    
    # Конкатенируем текстовые поля для сравнения
    text = name + " " + content + " " + description
    print("Текст для сравнения:", text)
    
    # Получаем эмбеддинги для текста с помощью TinyBERT
    text_embeddings = get_text_embeddings_tinybert([text])
    text_embeddings2 = get_text_embeddings_tinybert([average_probabilities_cat3[0][0]+ " " + average_probabilities_cat4[0][0]])
    # Вычисляем косинусное сходство между эмбеддингами изображений и текстов
    similarity_scores = compute_similarity(text_embeddings2, text_embeddings)

    # Используем среднее значение сходства для результата
    average_similarity = similarity_scores.mean()

    return {
        "id": id,
        "category_1": [category_1[average_probabilities_cat1[0][0]], average_probabilities_cat1[0][1]],
        "category_2": [category_2[average_probabilities_cat2[0][0]], average_probabilities_cat2[0][1]],
        "category_3": [category_3[average_probabilities_cat3[0][0]], average_probabilities_cat3[0][1]],
        "category_4": [category_4[average_probabilities_cat4[0][0]], average_probabilities_cat4[0][1]],
        "average_similarity": average_similarity
    }
