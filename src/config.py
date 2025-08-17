import os
import pandas as pd
from tqdm import tqdm

# Конфигурация
os.makedirs('../artifacts', exist_ok=True)                         # Создание папки для весов моделей и нормализации
pd.set_option('display.max_columns', None)                              # Вывод всех колонок в DataFrame
tqdm.pandas()                                                           # Отображение обучения

GPT_MODEL_NAME = 'all-MiniLM-L6-v2'                                     # Модель для эмбедингов диалогов
SENTIMENT_MODEL_TASK = 'sentiment-analysis'                             # Модель для извлечения тональности
SENTIMENT_MODEL = 'seara/rubert-tiny2-russian-sentiment'
STYLE_MODEL = 'ru_core_news_sm'                                         # Модель для извлечения стиля
QUESTIONS_DATA_BASE = 'kuznetsoffandrey/sberquad'                       # База данных для вопросов
