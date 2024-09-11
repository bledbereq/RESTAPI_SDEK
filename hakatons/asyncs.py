import aiohttp
import asyncio
from PIL import Image
from io import BytesIO
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from model import *

# Асинхронная функция для многопоточной загрузки изображений
async def fetch_images(urls):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        tasks = []
        semaphore = asyncio.Semaphore(10)  # Ограничение на количество одновременных запросов

        async def fetch_image(url):
            async with semaphore:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            img_data = await response.read()
                            return Image.open(BytesIO(img_data))
                        else:
                            print(f"Error fetching image {url}: {response.status}")
                            return None
                except Exception as e:
                    print(f"Failed to fetch image {url}: {e}")
                    return None

        tasks = [fetch_image(url) for url in urls]
        images = await asyncio.gather(*tasks)
        return [img for img in images if img is not None]

# Основная функция для обработки URL изображений
async def model_threads_async(json_mass_url_img):
    images = await fetch_images(json_mass_url_img)
    
    # Обработка изображений пакетами
    batch_size = 128
    batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
    
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, process_image_batch, batch) for batch in batches]
        results = await asyncio.gather(*tasks)
    
    # Объединение результатов
    probabilities = [item for sublist in results for item in sublist]
    return probabilities

# Синхронная обертка для запуска асинхронной функции
def model_threads(json_mass_url_img):
    return asyncio.run(model_threads_async(json_mass_url_img))

# Функция для объединения результатов и вычисления среднего значения для каждой метки
def calculate_average_probabilities(all_probabilities):
    sum_probs = defaultdict(float)
    count_probs = defaultdict(int)

    for prob_list in all_probabilities:
        for label, prob in prob_list:
            sum_probs[label] += prob
            count_probs[label] += 1

    average_probs = {label: sum_probs[label] / count_probs[label] for label in sum_probs}
    sorted_average_probs = sorted(average_probs.items(), key=lambda x: x[1], reverse=True)

    return sorted_average_probs
