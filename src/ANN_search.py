# Стандартные библиотеки
import json
import pickle

# Модели и их инструменты
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from cachetools import LRUCache

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
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(index, self.index_path)
        logger.info(
            f"FAISS IndexIVFFlat обучен и сохранен в {self.index_path}. Размерность: {dimension}, пользователей: {num_elements}, кластеров: {nlist}")
        self.cache = LRUCache(maxsize=1000)
        self.save_cache()
        logger.info("Кеш очищен создания индекса")
        return index

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

        index = faiss.read_index(self.index_path)

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
                    # Проверяем, есть ли user_id в исходной базе
                    user_data = user_data.iloc[0]
                    matches.append({
                        'user_id': int(user_data['user_id']),
                        'sex': str(user_data.get('sex', 'N/A')),
                        'job.title': str(user_data.get('job.title', 'N/A')),
                        'organization': str(user_data.get('organization', 'N/A')),
                        'annual.salary': float(user_data.get('annual.salary', 0)),
                        'age': int(user_data.get('age', 0)),
                        'question': str(user_data.get('question', 'N/A')),
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
                        'X': 0,
                        'Y': 0,
                        'Z': 0,
                        'distance': float(dist)
                    })

        # Формирование читаемого вывода
        logger.info(f"\nТоп-{min(k, len(matches))} ближайших пользователей (до ранжирования):")
        for i, res in enumerate(matches, 1):
            logger.info(f"{i}. user_id: {res['user_id']}, Пол: {res['sex']}, "
                        f"Должность: {res['job.title']}, Организация: {res['organization']}, "
                        f"Зарплата: {res['annual.salary']}, Возраст: {res['age']}, "
                        f"Вопрос: {res['question']}, X: {res['X']}, Y: {res['Y']}, Z: {res['Z']}, "
                        f"Дистанция: {res['distance']:.4f}")

        # Ранжирование по тематике, стилю и тональности
        query_features = processed_df[processed_df['user_id'] == matches[0]['user_id']][
            ['sentiment_pos', 'sentiment_neu', 'sentiment_neg', 'question_length', 'question_words', 'formality_score', 'topic_norm']
        ].values[0]

        feature_columns = ['sentiment_pos', 'sentiment_neu', 'sentiment_neg', 'question_length', 'question_words',
                           'formality_score', 'topic_norm']
        if query_features.shape != (len(feature_columns),):
            logger.error(
                f"Ожидается query_features размерности ({(len(feature_columns),)}), получено: {query_features.shape}")
            raise ValueError(
                f"Ожидается query_features размерности ({(len(feature_columns),)}), получено: {query_features.shape}")

        ranked_matches = self.rank_matches(matches, query_features, processed_df, top_k=10)

        with open(f'../results/search_results_user_{query_user_id}.json', 'w', encoding='utf-8') as f:
            json.dump(ranked_matches, f, ensure_ascii=False, indent=4)
        logger.info(f"Результаты поиска для user_id {query_user_id} сохранены в ../results/search_results_user_{query_user_id}.json")

        return ranked_matches, indices[0].tolist(), distances[0].tolist()

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

    def rank_matches(
            self,
            matches: List[Dict],
            query_features: np.ndarray,
            processed_df: pd.DataFrame,
            top_k: int = 10,
            weights: Dict[str, float] = None,
            save_weights_flag: bool = True
    ) -> List[Dict]:
        """
        Ранжирование метчей на основе признаков (тональность, стиль, тематика)

        :param matches: Список словарей, где каждый словарь — это матч от FAISS
        :param query_features: NumPy-массив размером (7,) с нормализованными признаками запроса
        :param processed_df: Pandas DataFrame из файла processed_profiles.csv, содержащий нормализованные признаки для всех пользователей
        :param top_k: Количество возвращаемых метчей после ранжирования
        :param weights: Словарь с весами для признаков (sentiment, style, topic)
        :param save_weights_flag: Флаг, отвечающий за сохранения используемых весов в файл
        :return: Список из top_k словарей с метаданными матчей, дополненных полем relevance_score (скор релевантности)
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
        if query_features.shape != (len(feature_columns),):
            logger.error(
                f"Ожидается query_features размерности ({(len(feature_columns),)}), получено: {query_features.shape}")
            raise ValueError(f"Ожидается query_features размерности ({(len(feature_columns),)}), получено: {query_features.shape}")

        # Ранжирование метчей
        ranked_matches = []
        for match in matches:
            user_id = match['user_id']
            match_row = processed_df[processed_df['user_id'] == user_id]
            if match_row.empty:
                continue
            # Извлекаются признаки матча
            match_features = match_row[feature_columns].values[0]

            # Вычисление скора релевантности
            # Тональность первые три признака (sentiment_pos, sentiment_neu, sentiment_neg)
            sentiment_query = query_features[:3].reshape(1, -1)
            sentiment_match = match_features[:3].reshape(1, -1)
            sentiment_score = cosine_similarity(sentiment_query, sentiment_match)[0][0]
            # Стиль - следующие три признака (question_length, question_words, formality_score)
            style_query = query_features[3:6].reshape(1, -1)
            style_match = match_features[3:6].reshape(1, -1)
            style_score = cosine_similarity(style_query, style_match)[0][0]
            # Тематика - нормализованное значение темы (topic_norm)
            topic_query = query_features[6]
            topic_match = match_features[6]
            topic_score = 1.0 if abs(topic_query - topic_match) < 0.1 else 0.5

            # Взвешенная сумма
            relevance_score = (
                weights['sentiment'] * sentiment_score +
                weights['style'] * style_score +
                weights['topic'] * topic_score
            )

            # Формирование результата
            ranked_matches.append({
                'user_id': match['user_id'],
                'sex': match['sex'],
                'job.title': match['job.title'],
                'organization': match['organization'],
                'annual.salary': match['annual.salary'],
                'age': match['age'],
                'question': match['question'],
                'X': match['X'],
                'Y': match['Y'],
                'Z': match['Z'],
                'distance': match['distance'],
                'relevance_score': relevance_score
            })

        # Сортировка и выбор топ-k
        ranked_matches = sorted(ranked_matches, key=lambda x: x['relevance_score'], reverse=True)[:top_k]

        # Вывод и сохранение результатов
        logger.info(f"\nТоп-{top_k} метчей после ранжирования:")
        for i, match in enumerate(ranked_matches, 1):
            logger.info(f"{i}. user_id: {match['user_id']}, Пол: {match['sex']}, "
                        f"Должность: {match['job.title']}, Организация: {match['organization']}, "
                        f"Зарплата: {match['annual.salary']}, Возраст: {match['age']}, "
                        f"Вопрос: {match['question']}, X: {match['X']}, Y: {match['Y']}, Z: {match['Z']}, "
                        f"Дистанция: {match['distance']:.4f}, Релевантность: {match['relevance_score']:.4f}")

        return ranked_matches

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
