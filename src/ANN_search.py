# Стандартные библиотеки
import json
import pickle

# Модели и их инструменты
import torch
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from cachetools import LRUCache
from scipy.spatial.distance import cosine

# Конфигурация
from config import os, pd, np, logging, Tuple, List, Dict


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FaissIndexManager:
    def __init__(self,
                 vectors_path: str = '../artifacts/user_vectors.npy',
                 index_path: str = '../artifacts/faiss_index_ivf.index',
                 intermediate_csv: str = '../data/intermediate_dataset.csv',
                 processed_csv: str = '../data/processed_profiles.csv',
                 nprobe_path: str = '../artifacts/nprobe.json',
                 weights_path: str = '../artifacts/weights.json'):
        self.vectors_path = vectors_path
        self.index_path = index_path
        self.intermediate_csv = intermediate_csv
        self.processed_csv = processed_csv
        self.nprobe_path = nprobe_path
        self.weights_path = weights_path
        self.cache = LRUCache(maxsize=1000)                # Кеш в памяти
        self.cache_file = '../artifacts/search_cache.pkl'  # Файл для сохранения кеша
        self.gpu_available = torch.cuda.is_available()
        self.gpu_id = 0 if self.gpu_available else -1
        logger.info(
            f"GPU available for FAISS: {self.gpu_available}, Device: {torch.cuda.get_device_name(0) if self.gpu_available else 'CPU'}")

        os.makedirs(os.path.dirname(self.intermediate_csv), exist_ok=True)
        os.makedirs(os.path.dirname(self.processed_csv), exist_ok=True)
        os.makedirs('../results', exist_ok=True)
        os.makedirs(os.path.dirname(self.nprobe_path), exist_ok=True)

        self.load_cache()
        self.load_nprobe()

    def build_faiss_ivf_index(
            self,
            nlist: int = 100,
            nprobe: int = 10
    ) -> faiss.IndexIVFFlat:
        """
        Функция обучает FAISS IndexIVFFlat для ANN-поиска

        :param nlist: Количество кластеров в индексе FAISS
        :param nprobe: Количество кластеров для приближённого (ANN) поиска
        :return: Возвращает обученный индекс FAISS
        """
        if not os.path.exists(self.vectors_path):
            logger.error(f"Файл векторов {self.vectors_path} не найден")
            raise FileNotFoundError(f"Файл векторов {self.vectors_path} не найден")

        # Загрузка векторов
        vectors = np.load(self.vectors_path, allow_pickle=True).astype('float32')
        if len(vectors.shape) != 2:
            logger.error(f"Ожидается двумерный массив, получена форма: {vectors.shape}")
            raise ValueError(f"Ожидается двумерный массив, получена форма: {vectors.shape}")

        dimension = vectors.shape[1]
        num_elements = vectors.shape[0]

        # Создание индекса IVFFlat
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

        # Перенос индекса на GPU, если доступен
        if self.gpu_available:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)
            logger.info(f"FAISS index transferred to GPU (device {self.gpu_id})")

        # Обучение индекса
        logger.info(f"Обучение FAISS IndexIVFFlat на {num_elements} векторах...")
        if not index.is_trained:
            index.train(vectors)
        if not index.is_trained:
            logger.error("Не удалось обучить индекс FAISS")
            raise ValueError("Не удалось обучить индекс FAISS")

        # Добавление векторов
        logger.info(f"Добавление {num_elements} векторов в индекс...")
        index.add(vectors)

        # Установка nprobe по умолчанию
        self.save_nprobe(nprobe)
        logger.info(f"Значение nprobe установлено по умолчанию {nprobe}")

        # Сохранение индекса
        cpu_index = faiss.index_gpu_to_cpu(index) if self.gpu_available else index
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(cpu_index, self.index_path)
        logger.info(
            f"FAISS IndexIVFFlat обучен и сохранен в {self.index_path}. Размерность: {dimension}, пользователей: {num_elements}, кластеров: {nlist}")
        self.cache = LRUCache(maxsize=1000)
        self.save_cache()
        logger.info("Кеш очищен создания индекса")
        return index

    def generate_explanation(
            self,
            query_data: pd.Series,
            match_data: pd.Series,
            query_processed: pd.Series,
            match_processed: pd.Series
    ) -> str:
        """
        Генерирует объяснение для матча, сравнивая запрашиваемого пользователя с матчом.

        :param query_data: Данные запрашиваемого пользователя из intermediate_df (pd.Series)
        :param match_data: Данные матча из intermediate_df (pd.Series)
        :param query_processed: Векторизованные данные запрашиваемого пользователя из processed_df (pd.Series)
        :param match_processed: Векторизованные данные матча из processed_df (pd.Series)
        :return: Текстовое объяснение
        """
        explanations = []

        # Сравнение возраста
        if abs(query_data['age'] - match_data['age']) <= 5:
            explanations.append(f"похожий возраст ({match_data['age']} лет)")

        # Сравнение пола
        if query_data['sex'] == match_data['sex']:
            explanations.append(f"одинаковый пол ({match_data['sex']})")

        # Сравнение должности (косинусное сходство job_vector)
        query_job_vector = np.fromstring(
            query_processed['job_vector'].replace('\n', ' ').replace('[', '').replace(']', ''), sep=' ')
        match_job_vector = np.fromstring(
            match_processed['job_vector'].replace('\n', ' ').replace('[', '').replace(']', ''), sep=' ')
        job_similarity = 1 - cosine(query_job_vector, match_job_vector)
        if job_similarity >= 0.8:
            explanations.append(f"похожая должность ({match_data['job.title']})")

        # Сравнение организации
        if query_data['organization'] == match_data['organization']:
            explanations.append(f"одинаковая организация ({match_data['organization']})")

        # Сравнение зарплаты
        if abs(query_data['annual.salary'] - match_data['annual.salary']) <= 0.2 * query_data['annual.salary']:
            explanations.append(f"похожая зарплата ({match_data['annual.salary']})")

        # Сравнение диалога (косинусное сходство question_vector)
        query_question_vector = np.fromstring(
            query_processed['question_vector'].replace('\n', ' ').replace('[', '').replace(']', ''), sep=' ')
        match_question_vector = np.fromstring(
            match_processed['question_vector'].replace('\n', ' ').replace('[', '').replace(']', ''), sep=' ')
        question_similarity = 1 - cosine(query_question_vector, match_question_vector)
        if question_similarity >= 0.8:
            keywords_str = match_data.get('question_keywords', 'не определены')
            explanations.append(
                f"похожий диалог (сходство: {question_similarity:.2f}, ключевые слова: {keywords_str})")

        # Сравнение темы диалога
        if abs(query_processed['topic_norm'] - match_processed['topic_norm']) <= 0.1:
            topic_keywords = match_data.get('topic_keywords', f"значение: {match_processed['topic_norm']:.2f}")
            explanations.append(f"похожая тема диалога (тема: {topic_keywords})")

        # Сравнение координат X, Y, Z отдельно
        if abs(query_data['X'] - match_data['X']) <= 100:
            explanations.append(f"похожая координата X ({match_data['X']})")
        if abs(query_data['Y'] - match_data['Y']) <= 100:
            explanations.append(f"похожая координата Y ({match_data['Y']})")
        if abs(query_data['Z'] - match_data['Z']) <= 100:
            explanations.append(f"похожая координата Z ({match_data['Z']})")

        if explanations:
            return f"Этот пользователь подходит, потому что у вас {', '.join(explanations)}."
        return "Этот пользователь подходит по общим характеристикам профиля и диалога."

    def search_faiss_ivf(
            self,
            query_vector: np.ndarray,
            query_user_id: int = None,
            k: int = 50,
            nprobe: int = None,
            save_nprobe_flag: bool = True
    ) -> Tuple[List[Dict], List[int], List[float]]:
        """
        Функция ищет топ-K метчей в FAISS IndexIVFFlat

        :param query_vector: Вектор объекта, для которого выполняется поиск
        :param query_user_id: id запрашиваемого пользователя
        :param k: Количество метчей
        :param nprobe: Количество кластеров для приближённого (ANN) поиска
        :param save_nprobe_flag: Флаг, отвечающий за сохранения nprobe при загрузке
        :return: Возвращает кортеж: список индексов метчей и список расстояния до соответсвующих метчей
        """
        if not os.path.exists(self.index_path):
            logger.error(f"Индекс {self.index_path} не найден")
            raise FileNotFoundError(f"Индекс {self.index_path} не найден")
        if not os.path.exists(self.intermediate_csv):
            logger.error(f"Файл {self.intermediate_csv} не найден")
            raise FileNotFoundError(f"Файл {self.intermediate_csv} не найден")
        if not os.path.exists(self.processed_csv):
            logger.error(f"Файл {self.processed_csv} не найден")
            raise FileNotFoundError(f"Файл {self.processed_csv} не найден")

        # Загрузка индекса
        index = faiss.read_index(self.index_path)
        if self.gpu_available:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)
            logger.info(f"FAISS index transferred to GPU (device {self.gpu_id}) for search")

        # Проверка формы и типа query_vector
        query_vector = np.array(query_vector).astype('float32')
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape[1] != index.d:
            logger.error(f"Ожидаемая размерность query_vector: {index.d}, получено: {query_vector.shape[1]}")
            raise ValueError(f"Ожидаемая размерность query_vector: {index.d}, получено: {query_vector.shape[1]}")
        if query_vector.shape[0] != 1:
            logger.error(f"Ожидается один вектор запроса, получено: {query_vector.shape[0]}")
            raise ValueError(f"Ожидается один вектор запроса, получено: {query_vector.shape[0]}")

        # Установка количества кластеров для поиска (баланс скорости/точности)
        if not nprobe:
            nprobe = self.load_nprobe()
        if save_nprobe_flag:
            self.save_nprobe(nprobe)
        index.nprobe = nprobe

        # Поиск
        distances, indices = index.search(query_vector, k)

        # Загрузка баз данных
        intermediate_df = pd.read_csv(self.intermediate_csv)
        processed_df = pd.read_csv(self.processed_csv)

        matches = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(processed_df):
                proc_user_id = processed_df.iloc[idx]['user_id']
                if query_user_id is not None and proc_user_id == query_user_id:
                    continue  # Пропускаем запись с тем же user_id, что и запрос
                user_data = intermediate_df[intermediate_df['user_id'] == proc_user_id]
                if not user_data.empty:
                    user_data = user_data.iloc[0]
                    matches.append({
                        'user_id': int(user_data['user_id']),
                        'sex': str(user_data.get('sex', 'N/A')),
                        'job.title': str(user_data.get('job.title', 'N/A')),
                        'organization': str(user_data.get('organization', 'N/A')),
                        'annual.salary': float(user_data.get('annual.salary', 0)),
                        'age': int(user_data.get('age', 0)),
                        'question': str(user_data.get('question', 'N/A')),
                        'question_keywords': str(user_data.get('question_keywords', 'не определены')),
                        'topic_keywords': str(user_data.get('topic_keywords', 'не определены')),
                        'X': int(user_data.get('X', 0)),
                        'Y': int(user_data.get('Y', 0)),
                        'Z': int(user_data.get('Z', 0)),
                        'distance': float(dist)
                    })
                else:
                    matches.append({
                        'user_id': int(proc_user_id),
                        'sex': 'N/A',
                        'job.title': 'N/A',
                        'organization': 'N/A',
                        'annual.salary': 0.0,
                        'age': 0,
                        'question': 'N/A',
                        'question_keywords': 'не определены',
                        'topic_keywords': 'не определены',
                        'X': 0,
                        'Y': 0,
                        'Z': 0,
                        'distance': float(dist)
                    })

        # Ранжирование по тематике, стилю и тональности
        ranked_matches = self.rank_matches(matches, intermediate_df, processed_df, query_user_id, top_k=10)

        with open(f'../results/search_results_user_{query_user_id}.json', 'w', encoding='utf-8') as f:
            json.dump(ranked_matches, f, ensure_ascii=False, indent=4)
        logger.info(f"Результаты поиска для user_id {query_user_id} сохранены в ../results/search_results_user_{query_user_id}.json")

        return ranked_matches, indices[0].tolist(), distances[0].tolist()

    def rank_matches(
            self,
            matches: List[Dict],
            intermediate_df: pd.DataFrame,
            processed_df: pd.DataFrame,
            query_user_id: int,
            top_k: int = 10,
            weights: Dict[str, float] = None,
            save_weights_flag: bool = True
    ) -> List[Dict]:
        """
        Ранжирование метчей на основе признаков (тональность, стиль, тематика)

        :param matches: Список словарей, где каждый словарь — это матч от FAISS
        :param intermediate_df: Pandas DataFrame из файла intermediate_dataset.csv
        :param processed_df: Pandas DataFrame из файла processed_profiles.csv
        :param query_user_id: ID запрашиваемого пользователя
        :param top_k: Количество возвращаемых метчей после ранжирования
        :param weights: Словарь с весами для признаков (sentiment, style, topic)
        :param save_weights_flag: Флаг, отвечающий за сохранения используемых весов в файл
        :return: Список из top_k словарей с метаданными матчей, дополненных полем relevance_score и explanation
        """
        # Настройка весов
        if weights is None:
            weights = self.load_weights()
        if save_weights_flag:
            self.save_weights(weights)

        if not all(k in weights for k in ['sentiment', 'style', 'topic']):
            logger.error("Веса должны содержать ключи: 'sentiment', 'style', 'topic'")
            raise ValueError("Веса должны содержать ключи: 'sentiment', 'style', 'topic'")
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            logger.error("Сумма весов должна быть равна 1")
            raise ValueError("Сумма весов должна быть равна 1")

        # Проверка входных данных
        feature_columns = ['sentiment_pos', 'sentiment_neu', 'sentiment_neg',
                           'question_length', 'question_words', 'formality_score', 'topic_norm']
        missing_columns = [col for col in feature_columns if col not in processed_df.columns]
        if missing_columns:
            logger.error(f"Отсутствуют столбцы в processed_df: {missing_columns}")
            raise ValueError(f"Отсутствуют столбцы в processed_df: {missing_columns}")

        # Получение данных запрашиваемого пользователя
        query_row = processed_df[processed_df['user_id'] == query_user_id]
        if query_row.empty:
            logger.error(f"Пользователь с user_id {query_user_id} не найден в processed_df")
            raise ValueError(f"Пользователь с user_id {query_user_id} не найден в processed_df")
        query_processed = query_row.iloc[0]
        query_data = intermediate_df[intermediate_df['user_id'] == query_user_id]
        if query_data.empty:
            logger.error(f"Пользователь с user_id {query_user_id} не найден в intermediate_df")
            raise ValueError(f"Пользователь с user_id {query_user_id} не найден в intermediate_df")
        query_data = query_data.iloc[0]

        # Ранжирование метчей
        ranked_matches = []
        for match in matches:
            user_id = match['user_id']
            match_row = processed_df[processed_df['user_id'] == user_id]
            if match_row.empty:
                continue
            match_processed = match_row.iloc[0]

            # Вычисление скора релевантности
            # Тональность первые три признака (sentiment_pos, sentiment_neu, sentiment_neg)
            sentiment_query = query_processed[['sentiment_pos', 'sentiment_neu', 'sentiment_neg']].values.reshape(1, -1)
            sentiment_match = match_processed[['sentiment_pos', 'sentiment_neu', 'sentiment_neg']].values.reshape(1, -1)
            sentiment_score = cosine_similarity(sentiment_query, sentiment_match)[0][0]
            # Стиль - следующие три признака (question_length, question_words, formality_score)
            style_query = query_processed[['question_length', 'question_words', 'formality_score']].values.reshape(1, -1)
            style_match = match_processed[['question_length', 'question_words', 'formality_score']].values.reshape(1, -1)
            style_score = cosine_similarity(style_query, style_match)[0][0]
            # Тематика - нормализованное значение темы (topic_norm)
            topic_query = query_processed['topic_norm']
            topic_match = match_processed['topic_norm']
            topic_score = 1.0 if abs(topic_query - topic_match) < 0.1 else 0.5

            # Взвешенная сумма
            relevance_score = (
                    weights['sentiment'] * sentiment_score +
                    weights['style'] * style_score +
                    weights['topic'] * topic_score
            )

            # Формирование результата без объяснения
            ranked_matches.append({
                'user_id': match['user_id'],
                'sex': match['sex'],
                'job.title': match['job.title'],
                'organization': match['organization'],
                'annual.salary': float(match['annual.salary']),
                'age': int(match['age']),
                'question': match['question'],
                'question_keywords': match['question_keywords'],
                'topic_keywords': match['topic_keywords'],
                'X': int(match['X']),
                'Y': int(match['Y']),
                'Z': int(match['Z']),
                'distance': float(match['distance']),
                'relevance_score': float(relevance_score),
                '_match_processed': match_processed  # Временное хранение для генерации объяснений
            })

        # Сортировка и выбор топ-k
        ranked_matches = sorted(ranked_matches, key=lambda x: x['relevance_score'], reverse=True)[:top_k]

        # Генерация объяснений только для топ-k матчей
        final_matches = []
        for r_match in ranked_matches:
            match_processed = r_match.pop('_match_processed')  # Извлекаем и удаляем временное поле
            match_data = intermediate_df[intermediate_df['user_id'] == r_match['user_id']]
            if match_data.empty:
                continue
            match_data = match_data.iloc[0]
            explanation = self.generate_explanation(query_data, match_data, query_processed, match_processed)
            r_match['explanation'] = explanation
            final_matches.append(r_match)

        # Вывод и сохранение результатов
        logger.info(f"\nТоп-{len(final_matches)} матчей после ранжирования:")
        for i, match in enumerate(final_matches, 1):
            logger.info(f"{i}.{'\n'}user_id: {match['user_id']}{'\n'}Пол: {match['sex']}{'\n'}"
                        f"Должность: {match['job.title']}{'\n'}Организация: {match['organization']}{'\n'}"
                        f"Зарплата: {match['annual.salary']}{'\n'}Возраст: {match['age']}{'\n'}"
                        f"Вопрос: {match['question']}{'\n'}Ключевые слова вопроса: {match['question_keywords']}{'\n'}"
                        f"Ключевые слова темы: {match['topic_keywords']}{'\n'}"
                        f"X: {match['X']}, Y: {match['Y']}{'\n'}Z: {match['Z']}{'\n'}"
                        f"Дистанция: {match['distance']:.4f}{'\n'}Релевантность: {match['relevance_score']:.4f}{'\n'}"
                        f"Объяснение: {match['explanation']}{'\n'}")

        return final_matches

    def search_by_user_id(
            self,
            user_id: int,
            k: int = 50,
            nprobe: int = None,
            save_nprobe_flag: bool = True
    ) -> Tuple[List[Dict], List[int], List[float]]:
        """
        Функция ищет топ-K метчей в FAISS IndexIVFFlat по user_id

        :param user_id: id пользователя, для которого выполняется поиск
        :param k: Количество соседних до ранжирования
        :param nprobe: Количество кластеров для приближённого (ANN) поиска
        :param save_nprobe_flag: Флаг, отвечающий за сохранения nprobe при загрузке
        :return: Возвращает кортеж: список индексов метчей и список расстояния до соответсвующих метчей
        """
        if not nprobe:
            nprobe = self.load_nprobe()
        if save_nprobe_flag:
            self.save_nprobe(nprobe)

        # Проверка кеша
        cache_key = (user_id, k, nprobe)
        if cache_key in self.cache:
            logger.info(f"Результаты для user_id={user_id}, k={k}, nprobe={nprobe} найдены в кеше")
            return self.cache[cache_key]

        logger.info(f"Поиск ближайших соседей для user_id={user_id}, k={k}, nprobe={nprobe}")

        if not os.path.exists(self.index_path):
            logger.error(f"Индекс {self.index_path} не найден")
            raise FileNotFoundError(f"Индекс {self.index_path} не найден")
        if not os.path.exists(self.intermediate_csv):
            logger.error(f"Файл {self.intermediate_csv} не найден")
            raise FileNotFoundError(f"Файл {self.intermediate_csv} не найден")
        if not os.path.exists(self.processed_csv):
            logger.error(f"Файл {self.processed_csv} не найден")
            raise FileNotFoundError(f"Файл {self.processed_csv} не найден")
        if not os.path.exists(self.vectors_path):
            logger.error(f"Файл {self.vectors_path} не найден")
            raise FileNotFoundError(f"Файл {self.vectors_path} не найден")

        processed_df = pd.read_csv(self.processed_csv)
        if 'user_id' not in processed_df.columns:
            logger.error("Столбец 'user_id' не найден в processed_csv")
            raise ValueError("Столбец 'user_id' не найден в processed_csv")
        if user_id not in processed_df['user_id'].values:
            logger.error(f"user_id {user_id} не найден в processed_csv")
            raise ValueError(f"user_id {user_id} не найден в processed_csv")

        vectors = np.load(self.vectors_path, allow_pickle=True).astype('float32')
        idx = processed_df[processed_df['user_id'] == user_id].index
        if len(idx) != 1:
            logger.error(f"Ожидается ровно один индекс для user_id {user_id}, найдено: {len(idx)}")
            raise ValueError(f"Ожидается ровно один индекс для user_id {user_id}, найдено: {len(idx)}")
        idx = idx[0]
        if idx >= len(vectors):
            logger.error(f"Индекс {idx} выходит за пределы массива векторов (длина: {len(vectors)})")
            raise ValueError(f"Индекс {idx} выходит за пределы массива векторов (длина: {len(vectors)})")

        query_vector = vectors[idx]
        ranked_matches, indices, distances = self.search_faiss_ivf(
            query_vector=query_vector,
            query_user_id=user_id,
            k=k,
            nprobe=nprobe
        )

        # Сохранение результатов в кеш
        self.cache[cache_key] = (ranked_matches, indices, distances)
        self.save_cache()
        logger.info(f"Результаты для user_id={user_id}, k={k}, nprobe={nprobe} добавлены в кеш")

        return ranked_matches, indices, distances

    def load_cache(self) -> None:
        """
        Загружает кеш из файла, если он существует
        """
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache_dict = pickle.load(f)
                self.cache = LRUCache(maxsize=1000)
                for key, value in cache_dict.items():
                    self.cache[key] = value
                logger.info(f"Загружен кеш из {self.cache_file}, размер: {len(self.cache)} записей")
            else:
                logger.info(f"Файл кеша {self.cache_file} не найден, создается пустой кеш")
        except Exception as e:
            logger.error(f"Ошибка загрузки кеша: {e}")
            self.cache = LRUCache(maxsize=1000)

    def save_cache(self) -> None:
        """
        Сохраняет кеш в файл
        """
        try:
            cache_dict = dict(self.cache)  # Преобразуем LRUCache в обычный словарь
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_dict, f)
            logger.info(f"Кеш сохранен в {self.cache_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения кеша: {e}")

    def load_nprobe(self) -> int:
        """
        Загружает значение nprobe из файла, если он существует.
        """
        try:
            if os.path.exists(self.nprobe_path):
                with open(self.nprobe_path, 'r') as f:
                    data = json.load(f)
                    nprobe = int(data.get('nprobe', 10))
                    logger.info(f"Загружено nprobe={nprobe} из {self.nprobe_path}")
                    return nprobe
            else:
                logger.info(f"Файл {self.nprobe_path} не найден, используется nprobe=10 по умолчанию")
                return 10
        except Exception as e:
            logger.error(f"Ошибка загрузки nprobe из {self.nprobe_path}: {e}")
            return 10

    def save_nprobe(self, nprobe: int) -> None:
        """
        Сохраняет значение nprobe в файл.
        """
        try:
            with open(self.nprobe_path, 'w') as f:
                json.dump({'nprobe': nprobe}, f)
            logger.info(f"Сохранено nprobe={nprobe} в {self.nprobe_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения nprobe в {self.nprobe_path}: {e}")

    def load_weights(self) -> Dict[str, float]:
        """
        Загружает веса из файла, если он существует.
        """
        default_weights = {'sentiment': 0.4, 'style': 0.4, 'topic': 0.2}
        try:
            if os.path.exists(self.weights_path):
                with open(self.weights_path, 'r') as f:
                    data = json.load(f)
                    if all(k in data for k in ['sentiment', 'style', 'topic']):
                        weights = {k: float(data[k]) for k in ['sentiment', 'style', 'topic']}
                        logger.info(f"Загружены веса из {self.weights_path}: {weights}")
                        if abs(sum(weights.values()) - 1.0) > 1e-6:
                            logger.warning(
                                f"Переданные веса не нормализованы (сумма: {sum(weights.values())}), используются веса по умолчанию: {default_weights}")
                            return default_weights
                        return weights
                    else:
                        logger.warning(
                            f"Некорректный формат весов в {self.weights_path}, используются веса по умолчанию")
                        return default_weights
            else:
                logger.info(f"Файл {self.weights_path} не найден, используются веса по умолчанию: {default_weights}")
                return default_weights
        except Exception as e:
            logger.error(f"Ошибка загрузки весов из {self.weights_path}: {e}")
            return default_weights

    def save_weights(self, weights: Dict[str, float]) -> None:
        """
        Сохраняет веса в файл.
        """
        try:
            with open(self.weights_path, 'w') as f:
                json.dump(weights, f)
            logger.info(f"Сохранены веса в {self.weights_path}: {weights}")
        except Exception as e:
            logger.error(f"Ошибка сохранения весов в {self.weights_path}: {e}")
