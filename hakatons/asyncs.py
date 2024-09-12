import aiohttp
from collections import defaultdict
import asyncio
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from model import process_image_batch
from PIL import Image
from io import BytesIO
import aiohttp
import asyncio

async def fetch_images(urls, max_semaphore=120, max_size=(200, 200)):
    session = await get_session()
    semaphore = asyncio.Semaphore(max_semaphore)

    async def fetch_image(url):
        async with semaphore:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        img_data = await response.read()
                        img = Image.open(BytesIO(img_data))
                        img.thumbnail(max_size)  # Уменьшение размера
                        return img
                    else:
                        print(f"Error fetching image {url}: {response.status}")
                        return None
            except Exception as e:
                print(f"Failed to fetch image {url}: {e}")
                return None

    tasks = [fetch_image(url) for url in urls]
    images = await asyncio.gather(*tasks)
    await session.close()
    return [img for img in images if img is not None]

# Инициализация aiohttp.ClientSession один раз
async def get_session():
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))


async def model_threads_async(json_mass_url_img, batch_size=4096):
    images = await fetch_images(json_mass_url_img)
    
    # Разбиваем изображения на батчи
    batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
    
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, process_image_batch, batch) for batch in batches]
        results = await asyncio.gather(*tasks)
    
    embeddings = [item for sublist in results for item in sublist]
    return embeddings

def model_threads(json_mass_url_img):
    return asyncio.run(model_threads_async(json_mass_url_img))

def model_threads_list(json_mass_url_img):
    return asyncio.run(fetch_images(json_mass_url_img))


def calculate_average_probabilities(all_probabilities, category_labels):
    sum_probs = defaultdict(float)
    count_probs = defaultdict(int)

    for prob_list in all_probabilities:
        for label, prob in prob_list:
            if label in category_labels.keys():
                sum_probs[label] += prob
                count_probs[label] += 1

    average_probs = {label: sum_probs[label] / count_probs[label] for label in sum_probs}
    sorted_average_probs = sorted(average_probs.items(), key=lambda x: x[1], reverse=True)
    return sorted_average_probs
