# Используем базовый образ с Python 3.9
FROM python:3.9-slim

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы приложения
COPY . .

# Устанавливаем зависимости
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Запускаем Flask сервер
CMD ["python", "app.py"]
