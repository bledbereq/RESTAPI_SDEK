# Файл для получения эмбединга
# от модели CLIP  в размере 512
# для каждого изображения / массив изображений -> массив эмбедингов (DataFraime)
 
import clip
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json
i=1
# Инициализация модели CLIP и устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)
i=0
# Функция для извлечения эмбеддинга изображения через модель CLIP
def fetch_image_embedding(img, output_dim=512):
    with torch.no_grad():
        image = preprocess(img).unsqueeze(0).to(device)
        embedding = model.encode_image(image).cpu().squeeze(0).tolist()
    
    # Урезаем или заполняем эмбеддинг до нужного размера
    if len(embedding) > output_dim:
        embedding = embedding[:output_dim]
    elif len(embedding) < output_dim:
        embedding += [0] * (output_dim - len(embedding))
    global i
    i = i+1
    print(i)
    return embedding


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

# Асинхронная обработка изображений и вычисление эмбеддингов
async def model_threads_async(json_mass_url_img):
    images = await fetch_images(json_mass_url_img)
    
    # Обработка изображений пакетами
    batch_size = 128
    batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
    
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, lambda batch: [fetch_image_embedding(img) for img in batch], batch)
                 for batch in batches]
        embeddings = await asyncio.gather(*tasks)
    
    # Объединение эмбеддингов
    embeddings = [item for sublist in embeddings for item in sublist]
    return embeddings

# Синхронная обертка для запуска асинхронной функции
def model_threads(json_mass_url_img):
    return asyncio.run(model_threads_async(json_mass_url_img))

def add_embeddings_to_dataframe(df):
    df = df.copy()  # Создаем копию, чтобы избежать предупреждения
    df['embedding'] = df['images'].apply(lambda urls: model_threads(json.loads(urls)))
    df.drop(columns=['images'], inplace=True)
    return df


# Загрузка данных
data = pd.read_excel('./Script-19_2024-09-11 10-34.xlsx')

print(data.info())
# Добавление эмбеддингов к датасету
data_with_embeddings = add_embeddings_to_dataframe(data)

# Сохранение данных с эмбеддингами
data_with_embeddings.to_csv('data_with_embeddings.csv', index=False)
data = pd.read_csv('data_with_embeddings.csv')
print(data.info())