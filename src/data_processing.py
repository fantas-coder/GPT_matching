# Стандартные библиотеки
import numpy as np
import random
import pickle
from typing import Tuple, List

# Загрузчик датасетов
from datasets import load_dataset

# Модели для векторизации бд
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from transformers import pipeline
import spacy

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Многопоточность
from concurrent.futures import ThreadPoolExecutor

# Конфигурация
from src.config import (os, pd, tqdm,
                        GPT_MODEL_NAME, SENTIMENT_MODEL_TASK, SENTIMENT_MODEL, STYLE_MODEL, QUESTIONS_DATA_BASE)


class DataProcessor:
    def __init__(self):
        self.gpt_model = SentenceTransformer(GPT_MODEL_NAME)            # Модель для эмбедингов текстов
        self.job_model = None                                           # Word2Vec для кодирования признаков
        self.sentiment_model = pipeline(                                # Модель для извлечения тональности
            SENTIMENT_MODEL_TASK,
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            return_all_scores=True
        )
        self.topic_model = BERTopic(language='russian', verbose=True)   # Модель для извлечения тематики
        self.nlp = spacy.load(STYLE_MODEL)                              # Модель для извлечения стиля
        self.scalers = {                                                # Параметры для нормализации
            'salary': None,
            'age': None,
            'features': None,
            'topic': None
        }

    def load_artifacts(self) -> None:
        """
        Функция загружает сохраненные модели и нормализаторы из 'artifacts'
        """
        try:
            os.makedirs('../artifacts', exist_ok=True)
            if os.path.exists('../artifacts/job_word2vec.model'):
                self.job_model = Word2Vec.load('../artifacts/job_word2vec.model')
            for name in self.scalers:
                scaler_path = f'../artifacts/{name}_scaler.pkl'
                if os.path.exists(scaler_path):
                    self.scalers[name] = pickle.load(open(scaler_path, 'rb'))
            if os.path.exists('../artifacts/topic_model'):
                self.topic_model = BERTopic.load('../artifacts/topic_model')
        except Exception as e:
            print(f"Ошибка загрузки артефактов: {e}")
            self.job_model = None
            self.scalers = {'salary': None, 'age': None, 'features': None, 'topic': None}

    def load_and_preprocess(
            self,
            filepath: str
    ) -> pd.DataFrame:
        """
        Функция загружает данные из filepath и удаляет лишние столбцы

        :param filepath: Путь к базе данных в формате .csv
        :return: База данных в формате DataFrame
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            df = pd.read_csv(filepath)
            return df.reset_index().rename(columns={'index': 'user_id'}).drop(['name', 'ethnic.origin'], axis=1)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {filepath} не найден")
        except KeyError as e:
            raise KeyError(f"Отсутствуют ожидаемые столбцы: {e}")

    def load_questions(
            self,
            num_records: int
    ) -> List[str]:
        """
        Функция загружает вопросы из SberQuAD, объединяя их по контексту

        :param num_records: Длина списка вопросов для совмещения с БД профилей
        :return: Возвращает список вопросов длиной num_records
        """
        try:
            dataset = load_dataset(QUESTIONS_DATA_BASE)

            # Создаем словарь {контекст: [список вопросов]}
            context_questions = {}
            for item in dataset['train']:
                context = item['context']
                question = item['question']

                if context not in context_questions:
                    context_questions[context] = []
                # Добавляем только уникальные вопросы для контекста
                if question not in context_questions[context]:
                    context_questions[context].append(question)

            # Объединяем вопросы для каждого контекста
            combined_questions = []
            for questions in context_questions.values():
                if len(questions) > 1:  # Берем только контексты с несколькими вопросами
                    combined = " ".join(questions)  # Разделитель для ясности
                    combined_questions.append(combined)

            random.shuffle(combined_questions)
            return combined_questions[:num_records]
        except Exception as e:
            print(f"Ошибка загрузки SberQuAD: {e}")
            # Генерация простых вопросов с нумерацией по контекстам
            return [f"Вопросы о теме {i}: Что такое X? Как работает Y?"
                    for i in range(num_records)]

    def encode_features(
            self,
            df: pd.DataFrame,
            train_mode: bool = False
    ) -> pd.DataFrame:
        """
        Функция кодирует признаки и добавляет тематику, тональность и стиль

        :param df: Исходная БД (Профиль + Текст)
        :param train_mode: Флаг, говорящий обучать ли модели или использовать уже обученные
        :return: Закодированная БД с выявленными признаками (тематика, тональность и стиль)
        """
        required_columns = ['sex', 'job.title', 'question']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют столбцы: {missing_columns}")

        # Кодирование пола
        df['gender_enc'] = df['sex'].map({'Male': 1, 'Female': 0})

        # Кодирование должностей
        df['job_clean'] = df['job.title'].str.lower().str.strip().fillna('')
        if train_mode or self.job_model is None:
            sentences = [[title] for title in tqdm(df['job_clean'], desc='Обработка должностей')]
            self.job_model = Word2Vec(
                sentences=sentences,
                vector_size=8,
                min_count=1,
                workers=4
            )
            os.makedirs('../artifacts', exist_ok=True)
            self.job_model.save('../artifacts/job_word2vec.model')
        avg_vector = np.mean(self.job_model.wv.vectors, axis=0) if self.job_model.wv.vectors.size > 0 else np.zeros(8)
        df['job_vector'] = df['job_clean'].apply(
            lambda x: self.job_model.wv[x] if x in self.job_model.wv else avg_vector
        )

        # Векторизация вопросов
        print("\nВекторизация вопросов...")
        questions = df['question'].fillna('').tolist()

        with ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(
                    lambda x: self.gpt_model.encode(x, normalize_embeddings=True) if pd.notnull(x) else np.zeros(384),
                    questions
                ),
                total=len(questions),
                desc='Обработка вопросов'
            ))
        df['question_vector'] = results

        # Извлечение тематики
        print("\nИзвлечение тематики...")
        valid_questions = [q if q else "Пустой вопрос" for q in questions]
        if train_mode:
            try:
                topics, _ = self.topic_model.fit_transform(valid_questions)
                os.makedirs('../artifacts', exist_ok=True)
                self.topic_model.save('../artifacts/topic_model', serialization='safetensors')
                print("Модель BERTopic сохранена с использованием safetensors")
            except Exception as e:
                print(f"Ошибка сохранения модели BERTopic: {e}. Продолжаем без сохранения модели.")
                topics = [0] * len(valid_questions)  # Фallback: нулевые темы
        else:
            try:
                topics = self.topic_model.transform(valid_questions)[0]
            except Exception as e:
                print(f"Ошибка преобразования тем: {e}. Используем нулевые темы.")
                topics = [0] * len(valid_questions)
        df['topic'] = topics

        # Обработка неизвестных тем
        if not train_mode and self.scalers['topic'] is not None:
            max_topic = self.scalers['topic'].data_max_[0]
            df['topic'] = df['topic'].apply(lambda x: x if x <= max_topic else max_topic)

        # Проверка тем
        print(f"Уникальные темы: {np.unique(topics)}")

        # Извлечение тональности
        print("\nИзвлечение тональности...")

        def get_sentiment_score(question):
            if pd.notnull(question) and question:
                result = self.sentiment_model(question)
                scores = {item['label']: item['score'] for item in result[0]}
                return [scores.get('positive', 0), scores.get('neutral', 0), scores.get('negative', 0)]
            return [0, 0, 0]

        df[['sentiment_pos', 'sentiment_neu', 'sentiment_neg']] = df['question'].progress_apply(
            lambda x: pd.Series(get_sentiment_score(x))
        )

        # Извлечение признаков стиля
        print("\nИзвлечение признаков стиля...")

        def extract_style_features(question):
            if pd.notnull(question) and question:
                doc = self.nlp(question)
                length = len(question.split())
                question_words = len(
                    [token for token in doc if token.text.lower() in ['что', 'как', 'почему', 'где', 'когда', 'кто']])
                formality_score = len([token for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB']]) / (
                            len(doc) + 1)
                return [length, question_words, formality_score]
            return [0, 0, 0]

        df[['question_length', 'question_words', 'formality_score']] = df['question'].progress_apply(
            lambda x: pd.Series(extract_style_features(x))
        )

        return df.drop(['job.title', 'sex', 'question', 'job_clean'], axis=1, errors='ignore')

    def normalize_features(
            self,
            df: pd.DataFrame,
            train_mode: bool = False
    ) -> pd.DataFrame:
        """
        Функция нормализует все признаки

        :param df: Закодированная БД
        :param train_mode: Флаг, говорящий обучать ли нормализаторы или использовать уже обученные
        :return: Нормализованная база данных
        """
        required_columns = ['annual.salary', 'age', 'question_vector', 'job_vector',
                            'sentiment_pos', 'sentiment_neu', 'sentiment_neg',
                            'question_length', 'question_words', 'formality_score', 'topic']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют столбцы: {missing_columns}")

        # Инициализация нормализаторов
        if train_mode or not all(self.scalers.values()):
            self.scalers = {
                'salary': StandardScaler(),
                'age': MinMaxScaler(),
                'features': StandardScaler(),
                'topic': MinMaxScaler()
            }

        df['salary_norm'] = self.scalers['salary'].fit_transform(df[['annual.salary']].fillna(0)) if train_mode else (
                            self.scalers['salary'].transform(df[['annual.salary']].fillna(0)))
        df['age_norm'] = self.scalers['age'].fit_transform(df[['age']].fillna(0)) if train_mode else (
                         self.scalers['age'].transform(df[['age']].fillna(0)))
        df['topic_norm'] = self.scalers['topic'].fit_transform(df[['topic']].fillna(0)) if train_mode else (
                           self.scalers['topic'].transform(df[['topic']].fillna(0)))

        # Нормализация векторов
        for col in ['question_vector', 'job_vector']:
            df[col] = df[col].apply(lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else x)

        # Нормализация числовых признаков
        df[['sentiment_pos', 'sentiment_neu', 'sentiment_neg',
            'question_length', 'question_words', 'formality_score']] = (
            self.scalers['features'].fit_transform(
                df[['sentiment_pos', 'sentiment_neu', 'sentiment_neg',
                    'question_length', 'question_words', 'formality_score']].fillna(0)
            ) if train_mode else
            self.scalers['features'].transform(
                df[['sentiment_pos', 'sentiment_neu', 'sentiment_neg',
                    'question_length', 'question_words', 'formality_score']].fillna(0)
            )
        )

        # Сохранение артефактов в режиме обучения
        if train_mode:
            os.makedirs('../artifacts', exist_ok=True)
            for name, scaler in self.scalers.items():
                pickle.dump(scaler, open(f'../artifacts/{name}_scaler.pkl', 'wb'))

        return df.drop(['annual.salary', 'age', 'organization', 'topic'], axis=1, errors='ignore')

    def create_vectors(
            self,
            df: pd.DataFrame
    ) -> np.ndarray:
        """
        Функция создаёт финальных векторов np.array из нормализованной БД

        :param df: Нормализованная БД
        :return: Финальный np.array
        """
        features = [
            df[['age_norm', 'salary_norm', 'topic_norm', 'sentiment_pos',
                'sentiment_neu', 'sentiment_neg', 'question_length',
                'question_words', 'formality_score']].values,   # 9 значений
            df['gender_enc'].values.reshape(-1, 1),             # 1 значение
            np.vstack(df['job_vector'].values),                 # 8 значений
            np.vstack(df['question_vector'].values)             # 384 значения
        ]
        vectors = np.hstack(features).astype('float32')         # Итого: 9+1+8+384=402
        print(f"Созданы векторы размерности: {vectors.shape}")
        return vectors

    def visualize_vectors(
            self,
            vectors: np.ndarray,
            df: pd.DataFrame,
            output_path: str = '../results/vectors_visualization.png'
    ) -> None:
        """
        Визуализация векторов с помощью t-SNE

        :param vectors: Векторы пользователей (numpy array)
        :param df: DataFrame с метаданными (topic, gender_enc)
        :param output_path: Путь для сохранения графика
        """
        print("\nВизуализация векторов с помощью t-SNE...")
        if len(vectors) != len(df):
            raise ValueError(
                f"Количество векторов ({len(vectors)}) не соответствует количеству записей в DataFrame ({len(df)})")

        # Применение t-SNE для снижения размерности до 2D
        tsne = TSNE(n_components=2, random_state=42, n_jobs=-1, perplexity=min(30, len(vectors) - 1))
        vectors_2d = tsne.fit_transform(vectors)

        # Создание DataFrame для визуализации
        viz_df = pd.DataFrame({
            'x': vectors_2d[:, 0],
            'y': vectors_2d[:, 1],
            'topic': df['topic_norm'].astype(str),
            'gender': df['gender_enc'].map({1: 'Male', 0: 'Female'}),
            'user_id': df['user_id'].astype(str)
        })

        # Построение scatter-графика
        plt.figure(figsize=(10, 8))
        scatter = sns.scatterplot(
            data=viz_df,
            x='x',
            y='y',
            hue='topic',
            style='gender',
            palette='tab20',
            alpha=0.6,
            legend=False
        )
        plt.title('t-SNE Визуализация векторизованной базы данных')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')

        # Сохранение графика
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"График сохранен в {output_path}")
        plt.close()

    def visualize_saved_vectors(
            self,
            vectors_path: str = '../artifacts/user_vectors.npy',
            metadata_path: str = '../data/processed_profiles.csv',
            output_path: str = '../results/vectors_visualization.png'
    ) -> None:
        """
        Визуализация сохраненных векторов из artifacts/user_vectors.npy

        :param vectors_path: Путь до сохраненных векторов
        :param metadata_path: Путь до CSV с метаданными (processed_profiles.csv)
        :param output_path: Путь для сохранения графика
        """
        if not os.path.exists(vectors_path):
            raise FileNotFoundError(f"Файл векторов {vectors_path} не найден")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Файл метаданных {metadata_path} не найден")

        # Загрузка векторов
        vectors = np.load(vectors_path, allow_pickle=True).astype('float32')
        print(f"Загружены векторы размерности: {vectors.shape}")

        # Загрузка метаданных
        df = pd.read_csv(metadata_path)
        required_columns = ['user_id', 'topic_norm', 'gender_enc']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют столбцы в {metadata_path}: {missing_columns}")

        # Проверка соответствия размеров
        if len(vectors) != len(df):
            raise ValueError(
                f"Количество векторов ({len(vectors)}) не соответствует количеству записей в {metadata_path} ({len(df)})")

        # Визуализация
        self.visualize_vectors(vectors, df, output_path=output_path)

    def process_data(
            self,
            input_path: str = '../data/atlanta_salary_data_2015_full.csv',
            output_path: str = '../data/processed_profiles.csv',
            intermediate_path: str = '../data/intermediate_dataset.csv'
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Функция выполняет основной пайплайн обработки БД
        (Загрузка профилей + текстов -> Кодирование + Нормализация -> Создание векторов)

        :param input_path: Путь до БД профилей в .csv
        :param output_path: Путь для сохранения векторизованной БД в .csv
        :param intermediate_path: Путь для сохранения начальной объединённой БД (Профиль + Текст) в .csv
        :return: Возвращает векторизованную БД и финальный вектор
        """
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)

        # Загрузка и объединение данных
        df = self.load_and_preprocess(input_path)   # Загрузка БД профилей
        questions = self.load_questions(len(df))    # Загрузка тестов
        df['question'] = questions[:len(df)]        # Объединение в одну БД

        # Сохранение промежуточной БД с вопросами
        print("Сохранение промежуточной БД с вопросами...")
        df.to_csv(intermediate_path, index=False, encoding='utf-8')
        print(f"Промежуточная БД сохранена в {intermediate_path}")
        print("Исходная БД")
        print(df.head())

        # Кодирование признаков
        df = self.encode_features(df, train_mode=True)               # Кодируем признаки
        df = self.normalize_features(df, train_mode=True)            # Нормализация бд
        print("Нормализованная БД")
        print(df.head())

        # Визуализация, создание и сохранение векторов
        vectors = self.create_vectors(df)                                                       # Создание
        self.visualize_vectors(vectors, df, output_path='../results/vectors_visualization.png')    # Визуализация
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)                                                     # Сохранение
        os.makedirs('../artifacts', exist_ok=True)
        np.save('../artifacts/user_vectors.npy', vectors)

        return df, vectors
