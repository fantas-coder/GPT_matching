# Стандартные библиотеки
import random
import pickle

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

# Конфигурация
from src.config import (os, pd, np, tqdm, logging, Tuple, List,
                        GPT_MODEL_NAME, SENTIMENT_MODEL_TASK, SENTIMENT_MODEL, STYLE_MODEL, QUESTIONS_DATA_BASE)

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Создаем обработчик для вывода логов через tqdm
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception as e:
            logger.error(f"Ошибка настройки tqdm: {e}")
            self.handleError(record)


# Удаляем существующие обработчики, чтобы избежать дублирования
logger.handlers = []
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
tqdm_handler = TqdmLoggingHandler()
tqdm_handler.setFormatter(formatter)
logger.addHandler(tqdm_handler)


class DataProcessor:
    def __init__(self):
        self.gpt_model = SentenceTransformer(GPT_MODEL_NAME)  # Модель для эмбедингов текстов
        self.job_model = None  # Word2Vec для кодирования признаков
        self.sentiment_model = pipeline(  # Модель для извлечения тональности
            SENTIMENT_MODEL_TASK,
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            return_all_scores=True
        )
        self.topic_model = BERTopic(language='russian', verbose=True)  # Модель для извлечения тематики
        self.nlp = spacy.load(STYLE_MODEL)  # Модель для извлечения стиля
        self.scalers = {  # Параметры для нормализации
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
                logger.info("Загружена модель Word2Vec: ../artifacts/job_word2vec.model")
            for name in self.scalers:
                scaler_path = f'../artifacts/{name}_scaler.pkl'
                if os.path.exists(scaler_path):
                    self.scalers[name] = pickle.load(open(scaler_path, 'rb'))
                    logger.info(f"Загружен нормализатор: {scaler_path}")
            if os.path.exists('../artifacts/topic_model'):
                self.topic_model = BERTopic.load('../artifacts/topic_model')
                logger.info("Загружена модель BERTopic: ../artifacts/topic_model")
        except Exception as e:
            logger.error(f"Ошибка загрузки артефактов: {e}")
            self.job_model = None
            self.scalers = {'salary': None, 'age': None, 'features': None, 'topic': None}

    def load_and_preprocess(
            self,
            input_filepath: str = '../data/atlanta_salary_data_2015_full.csv',
            output_filepath: str = '../data/prepared_data.csv',
    ) -> pd.DataFrame:
        """
        Загружает данные из input_filepath, удаляет лишние столбцы, добавляет X, Y, Z и сохраняет в output_filepath

        :param input_filepath: Путь к начальной базе данных в формате .csv
        :param output_filepath: Путь к предобработанной базе данных в формате .csv
        :return: База данных в формате DataFrame
        """
        os.makedirs(os.path.dirname(input_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        try:
            df = pd.read_csv(input_filepath)
            if 'user_id' in df.columns:
                logger.info("Обнаружен существующий user_id в start_filepath, используем его")
                df['user_id'] = df['user_id'].astype(int)
            else:
                df = df.reset_index().rename(columns={'index': 'user_id'})
                df['user_id'] = df['user_id'].astype(int)
            df = df.drop(['name', 'ethnic.origin'], axis=1, errors='ignore')
            # Добавление колонок X, Y, Z со случайными значениями от 0 до 1000
            df['X'] = np.random.randint(0, 1001, size=len(df))
            df['Y'] = np.random.randint(0, 1001, size=len(df))
            df['Z'] = np.random.randint(0, 1001, size=len(df))
            # Сохранение обновленного датасета
            df.to_csv(output_filepath, index=False, encoding='utf-8')
            logger.info(f"Загружен и обработан файл: {output_filepath}, строк: {len(df)}, добавлены колонки X, Y, Z")
            logger.info(f"Тип user_id: {df['user_id'].dtype}, форма: {df['user_id'].shape}")
            return df
        except FileNotFoundError:
            logger.error(f"Файл {output_filepath} не найден")
            raise FileNotFoundError(f"Файл {output_filepath} не найден")
        except KeyError as e:
            logger.error(f"Отсутствуют ожидаемые столбцы: {e}")
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
            questions = combined_questions[:num_records]
            logger.info(f"Загружено {len(questions)} вопросов из SberQuAD")
            return questions
        except Exception as e:
            logger.error(f"Ошибка загрузки SberQuAD: {e}")
            questions = [f"Вопросы о теме {i}: Что такое X? Как работает Y?" for i in range(num_records)]
            logger.warning(f"Сгенерированы заглушки для {num_records} вопросов")
            return questions

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
            logger.error(f"Отсутствуют столбцы: {missing_columns}")
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
            logger.info("Сохранена модель Word2Vec: ../artifacts/job_word2vec.model")
        avg_vector = np.mean(self.job_model.wv.vectors, axis=0) if self.job_model.wv.vectors.size > 0 else np.zeros(8)
        df['job_vector'] = df['job_clean'].apply(
            lambda x: self.job_model.wv[x] if x in self.job_model.wv else avg_vector
        )

        # Векторизация вопросов
        logger.info("Векторизация вопросов...")
        questions = df['question'].fillna('').tolist()
        # Пакетная обработка
        results = self.gpt_model.encode(
            questions,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        df['question_vector'] = [results[i] if questions[i] else np.zeros(384) for i in range(len(questions))]
        logger.info("Векторизация вопросов завершена")

        # Извлечение тематики
        logger.info("Извлечение тематики...")
        valid_questions = [q if q else "Пустой вопрос" for q in questions]
        if train_mode:
            try:
                topics, _ = self.topic_model.fit_transform(valid_questions)
                os.makedirs('../artifacts', exist_ok=True)
                self.topic_model.save('../artifacts/topic_model', serialization='safetensors')
                logger.info("Модель BERTopic сохранена: ../artifacts/topic_model")
            except Exception as e:
                logger.error(f"Ошибка сохранения модели BERTopic: {e}")
                topics = [0] * len(valid_questions)  # Фallback: нулевые темы
        else:
            try:
                topics = self.topic_model.transform(valid_questions)[0]
            except Exception as e:
                logger.error(f"Ошибка преобразования тем: {e}")
                topics = [0] * len(valid_questions)
        df['topic'] = topics

        # Обработка неизвестных тем
        if not train_mode and self.scalers['topic'] is not None:
            max_topic = self.scalers['topic'].data_max_[0]
            df['topic'] = df['topic'].apply(lambda x: x if x <= max_topic else max_topic)

        # Проверка тем
        logger.info(f"Уникальные темы: {np.unique(topics)}")

        # Извлечение тональности
        logger.info("Извлечение тональности...")

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
        logger.info("Извлечение признаков стиля...")

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
                            'question_length', 'question_words', 'formality_score',
                            'topic', 'X', 'Y', 'Z']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Отсутствуют столбцы: {missing_columns}")
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
        feature_columns = ['sentiment_pos', 'sentiment_neu', 'sentiment_neg',
                           'question_length', 'question_words', 'formality_score',
                           'X', 'Y', 'Z']
        df[feature_columns] = (
            self.scalers['features'].fit_transform(df[feature_columns].fillna(0)
                                                   ) if train_mode else
            self.scalers['features'].transform(df[feature_columns].fillna(0)))

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
                'question_words', 'formality_score']].values,  # 9 значений
            df['gender_enc'].values.reshape(-1, 1),  # 1 значение
            np.vstack(df['job_vector'].values),  # 8 значений
            np.vstack(df['question_vector'].values),  # 384 значения
            df[['X', 'Y', 'Z']].values  # 3 значения (без нормализации)
        ]

        vectors = np.hstack(features).astype('float32')  # Итого: 9+1+8+384+3=405
        logger.info(f"Созданы векторы размерности: {vectors.shape}")
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
        logger.info("Визуализация векторов с помощью t-SNE...")
        if len(vectors) != len(df):
            logger.error(
                f"Количество векторов ({len(vectors)}) не соответствует количеству записей в DataFrame ({len(df)})")
            raise ValueError(
                f"Количество векторов ({len(vectors)}) не соответствует количеству записей в DataFrame ({len(df)})")

        # Проверка входных данных
        required_columns = ['topic_norm', 'gender_enc', 'user_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Отсутствуют столбцы в DataFrame: {missing_columns}")
            raise ValueError(f"Отсутствуют столбцы в DataFrame: {missing_columns}")

        # Проверка типов данных и пропусков
        logger.info(f"Проверка DataFrame: {df[required_columns].dtypes}")
        logger.info(f"Количество пропусков: topic_norm={df['topic_norm'].isna().sum()}, "
                    f"gender_enc={df['gender_enc'].isna().sum()}, user_id={df['user_id'].isna().sum()}")
        # Заполнение пропусков
        df = df.copy()  # Создаем копию, чтобы не изменять оригинальный DataFrame
        df['topic_norm'] = df['topic_norm'].fillna(0).astype(float)
        df['gender_enc'] = df['gender_enc'].fillna(0).astype(int)
        df['user_id'] = df['user_id'].fillna(-1).astype(int)

        # Применение t-SNE для снижения размерности до 2D
        tsne = TSNE(n_components=2, random_state=42, n_jobs=-1, perplexity=min(30, len(vectors) - 1))
        vectors_2d = tsne.fit_transform(vectors)
        logger.info(f"Форма vectors_2d после TSNE: {vectors_2d.shape}")

        if vectors_2d.shape[1] != 2:
            logger.error(f"Ожидается vectors_2d с 2 компонентами, получено: {vectors_2d.shape[1]}")
            raise ValueError(f"Ожидается vectors_2d с 2 компонентами, получено: {vectors_2d.shape[1]}")

        # Явное преобразование в одномерные массивы
        x = vectors_2d[:, 0].flatten().astype(float)
        y = vectors_2d[:, 1].flatten().astype(float)
        logger.info(f"Форма x: {x.shape}, y: {y.shape}")

        # Проверка типов данных
        topic_norm = df['topic_norm'].astype(str).values
        gender = df['gender_enc'].map({1: 'Male', 0: 'Female'}).fillna('Unknown').values
        user_id = df['user_id'].astype(str).values
        logger.info(f"Форма topic_norm: {topic_norm.shape}, gender: {gender.shape}, user_id: {user_id.shape}")

        # Создание DataFrame для визуализации
        viz_df = pd.DataFrame({
            'x': x,
            'y': y,
            'topic': topic_norm,
            'gender': gender,
            'user_id': user_id
        })

        # Построение scatter-графика
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
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
        logger.info(f"График сохранен в {output_path}")
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
            logger.error(f"Файл векторов {vectors_path} не найден")
            raise FileNotFoundError(f"Файл векторов {vectors_path} не найден")
        if not os.path.exists(metadata_path):
            logger.error(f"Файл метаданных {metadata_path} не найден")
            raise FileNotFoundError(f"Файл метаданных {metadata_path} не найден")

        # Загрузка векторов
        vectors = np.load(vectors_path, allow_pickle=True).astype('float32')
        logger.info(f"Загружены векторы размерности: {vectors.shape}")

        # Загрузка метаданных
        df = pd.read_csv(metadata_path)
        required_columns = ['user_id', 'topic_norm', 'gender_enc']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Отсутствуют столбцы в {metadata_path}: {missing_columns}")
            raise ValueError(f"Отсутствуют столбцы в {metadata_path}: {missing_columns}")

        # Проверка соответствия размеров
        if len(vectors) != len(df):
            logger.error(
                f"Количество векторов ({len(vectors)}) не соответствует количеству записей в {metadata_path} ({len(df)})")
            raise ValueError(
                f"Количество векторов ({len(vectors)}) не соответствует количеству записей в {metadata_path} ({len(df)})")

        # Визуализация
        self.visualize_vectors(vectors, df, output_path=output_path)

    def process_data(
            self,
            input_path: str = '../data/atlanta_salary_data_2015_full.csv',
            prepared_path: str = '../data/prepared_data.csv',
            intermediate_path: str = '../data/intermediate_dataset.csv',
            output_path: str = '../data/processed_profiles.csv'
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Функция выполняет основной пайплайн обработки БД
        (Загрузка профилей + текстов -> Кодирование + Нормализация -> Создание векторов)

        :param input_path: Путь до начальной БД профилей в .csv
        :param prepared_path: Путь до БД профилей (предобработанная версия) в .csv
        :param intermediate_path: Путь до объединённой БД (Профиль + Текст) в .csv
        :param output_path: Путь до векторизованной БД в .csv
        :return: Возвращает векторизованную БД и финальный вектор
        """
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.makedirs(os.path.dirname(prepared_path), exist_ok=True)
        os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Загрузка и объединение данных
        df = self.load_and_preprocess(input_path, prepared_path)        # Загрузка БД профилей
        questions = self.load_questions(len(df))                        # Загрузка тестов
        df['question'] = questions[:len(df)]                            # Объединение в одну БД

        # Сохранение промежуточной БД с вопросами
        df.to_csv(intermediate_path, index=False, encoding='utf-8')
        logger.info(f"Промежуточная БД сохранена: {intermediate_path}")
        logger.info(f"Исходная БД (первые 5 строк):\n{df.head()}")

        # Кодирование признаков
        df = self.encode_features(df, train_mode=True)                  # Кодируем признаки
        df = self.normalize_features(df, train_mode=True)               # Нормализация бд
        logger.info(f"Нормализованная БД (первые 5 строк):\n{df.head()}")

        # Визуализация, создание и сохранение векторов
        vectors = self.create_vectors(df)                               # Создание
        self.visualize_vectors(vectors, df)                             # Визуализация
        df.to_csv(output_path, index=False)                             # Сохранение
        os.makedirs('../artifacts', exist_ok=True)
        np.save('../artifacts/user_vectors.npy', vectors)
        logger.info(f"Сохранены векторы: ../artifacts/user_vectors.npy")
        logger.info(f"Сохранена векторизованная БД: {output_path}")

        logger.info(f"Созданы векторы размерности: {vectors.shape}")
        logger.info(f"Пример вектора: {vectors[0]}")

        return df, vectors

    def retrain_word2vec(
            self,
            intermediate_path: str = '../data/intermediate_dataset.csv',
            output_path: str = '../data/processed_profiles.csv'
    ) -> None:
        """
        Переобучает модель Word2Vec и обновляет job_vector

        :param intermediate_path: Путь до объединённой БД (Профиль + Текст) в .csv
        :param output_path: Путь до векторизованной БД в .csv
        """
        # Загрузка данных из intermediate_dataset.csv
        try:
            intermediate_df = pd.read_csv(intermediate_path)
            if 'job.title' not in intermediate_df.columns:
                logger.error(f"Столбец 'job.title' отсутствует в {intermediate_path}")
                raise ValueError(f"Столбец 'job.title' отсутствует в {intermediate_path}")
        except FileNotFoundError:
            logger.error(f"Файл {intermediate_path} не найден")
            raise FileNotFoundError(f"Файл {intermediate_path} не найден")

        # Подготовка данных для обучения
        job_titles = intermediate_df['job.title'].str.lower().str.strip().fillna('').tolist()
        job_titles = [title for title in job_titles if title and isinstance(title, str)]
        if not job_titles:
            logger.error(f"Нет валидных job.title в {intermediate_path}")
            raise ValueError(f"Нет валидных job.title в {intermediate_path}")

        new_sentences = [[title] for title in job_titles]
        logger.info(f"Количество job.title для обучения Word2Vec: {len(new_sentences)}")
        logger.info(f"Примеры job.title: {new_sentences[:5]}")

        # Переобучение Word2Vec
        self.job_model = Word2Vec(
            sentences=new_sentences,
            vector_size=8,
            min_count=1,
            workers=4
        )
        os.makedirs('../artifacts', exist_ok=True)
        self.job_model.save('../artifacts/job_word2vec.model')
        logger.info("Сохранена модель Word2Vec: ../artifacts/job_word2vec.model")

        # Обновление финальных векторов в processed_profiles.csv
        try:
            df = pd.read_csv(output_path)

            # Обновление job_vector
            df['job_vector'] = intermediate_df['job.title'].str.lower().str.strip().fillna('')
            avg_vector = np.mean(self.job_model.wv.vectors, axis=0) if self.job_model.wv.vectors.size > 0 else np.zeros(
                8)
            df['job_vector'] = df['job_vector'].apply(
                lambda x: self.job_model.wv[x] if x in self.job_model.wv else avg_vector
            )

            # Обработка question_vector
            def parse_question_vector(x):
                """Преобразует строку с вектором в numpy array"""
                try:
                    clean_str = x.replace('/n', ' ').replace('[', '').replace(']', '')
                    return np.fromstring(clean_str, sep=' ')
                except Exception as ex:
                    logger.info(f"Ошибка преобразования df['question_vector']: {ex}")

            df['question_vector'] = df['question_vector'].apply(parse_question_vector)

            # Пересчет финальных векторов
            vectors = self.create_vectors(df)

            # Сохранение обновленных данных
            df.to_csv(output_path, index=False, encoding='utf-8')
            np.save('../artifacts/user_vectors.npy', vectors)
            logger.info(
                f"Обновлены {output_path} и ../artifacts/user_vectors.npy после переобучения Word2Vec")
        except Exception as e:
            logger.error(f"Ошибка обновления векторов после переобучения Word2Vec: {e}")
            raise

    def retrain_all_models(
            self,
            intermediate_path: str = '../data/intermediate_dataset.csv',
            output_path: str = '../data/processed_profiles.csv'
    ) -> None:
        """
        Переобучает все модели (Word2Vec, BERTopic, sentiment, style) и обновляет вектора

        :param intermediate_path: Путь до объединённой БД (Профиль + Текст)
        :param output_path: Путь до векторизованной БД для сохранения
        """
        try:
            intermediate_df = pd.read_csv(intermediate_path)
            required_columns = ['job.title', 'question', 'user_id', 'sex', 'age', 'annual.salary']
            missing_columns = [col for col in required_columns if col not in intermediate_df.columns]
            if missing_columns:
                logger.error(f"Отсутствуют столбцы в {intermediate_path}: {missing_columns}")
                raise ValueError(f"Отсутствуют столбцы в {intermediate_path}: {missing_columns}")
        except FileNotFoundError:
            logger.error(f"Файл {intermediate_path} не найден")
            raise FileNotFoundError(f"Файл {intermediate_path} не найден")

        # Обновляем вектора в processed_profiles.csv и user_vectors.npy
        try:
            # Кодирование признаков
            df = self.encode_features(intermediate_df, train_mode=True)
            df = self.normalize_features(df, train_mode=True)
            logger.info(f"Нормализованная БД (первые 5 строк):\n{df.head()}")

            # Создание и сохранение векторов
            vectors = self.create_vectors(df)  # Создание

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)  # Сохранение

            os.makedirs('../artifacts', exist_ok=True)
            np.save('../artifacts/user_vectors.npy', vectors)
            logger.info(
                f"Обновлены {output_path} и ../artifacts/user_vectors.npy после переобучения всех моделей")
        except Exception as e:
            logger.error(f"Ошибка обновления векторов после переобучения всех моделей: {e}")
            raise
