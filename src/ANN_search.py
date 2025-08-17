# Стандартные библиотеки
import numpy as np
import json
from typing import Tuple, List, Dict

# Модели и их инструменты
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Конфигурация
from config import os, pd


class FaissIndexManager:
    def __init__(self,
                 vectors_path: str = '../artifacts/user_vectors.npy',
                 index_path: str = '../artifacts/faiss_index_ivf.index',
                 intermediate_csv: str = '../data/intermediate_dataset.csv',
                 processed_csv: str = '../data/processed_profiles.csv'):
        self.vectors_path = vectors_path
        self.index_path = index_path
        self.intermediate_csv = intermediate_csv
        self.processed_csv = processed_csv

        os.makedirs(os.path.dirname(self.intermediate_csv), exist_ok=True)
        os.makedirs(os.path.dirname(self.processed_csv), exist_ok=True)
        os.makedirs('../results', exist_ok=True)

    def build_faiss_ivf_index(
            self,
            nlist: int = 100
    ) -> faiss.IndexIVFFlat:
        """
        Функция обучает FAISS IndexIVFFlat для ANN-поиска

        :param nlist: Количество кластеров для приближённого (ANN) поиска
        :return: Возвращает обученный индекс FAISS
        """
        if not os.path.exists(self.vectors_path):
            raise FileNotFoundError(f"Файл векторов {self.vectors_path} не найден")

        # Загрузка векторов
        vectors = np.load(self.vectors_path, allow_pickle=True).astype('float32')
        if len(vectors.shape) != 2:
            raise ValueError(f"Ожидается двумерный массив, получена форма: {vectors.shape}")

        dimension = vectors.shape[1]
        num_elements = vectors.shape[0]

        # Создание индекса IVFFlat
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

        # Обучение индекса
        print(f"Обучение FAISS IndexIVFFlat на {num_elements} векторах...")
        if not index.is_trained:
            index.train(vectors)
        if not index.is_trained:
            raise ValueError("Не удалось обучить индекс FAISS")

        # Добавление векторов
        print(f"Добавление {num_elements} векторов в индекс...")
        index.add(vectors)

        # Сохранение индекса
        faiss.write_index(index, self.index_path)
        print(
            f"FAISS IndexIVFFlat обучен и сохранен в {self.index_path}. Размерность: {dimension}, пользователей: {num_elements}, кластеров: {nlist}")

        return index

    def search_faiss_ivf(
            self,
            query_vector: np.ndarray,
            query_user_id: int = None,
            k: int = 50,
            nprobe: int = 10
    ) -> Tuple[List[Dict], List[int], List[float]]:
        """
        Функция ищет топ-K метчей в FAISS IndexIVFFlat

        :param query_vector: Вектор объекта, для которого выполняется поиск
        :param query_user_id: id запрашиваемого пользователя
        :param k: Количество метчей
        :param nprobe: Количество кластеров для приближённого (ANN) поиска
        :return: Возвращает кортеж: список индексов метчей и список расстояния до соответсвующих метчей
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Индекс {self.index_path} не найден")
        if not os.path.exists(self.intermediate_csv):
            raise FileNotFoundError(f"Файл {self.intermediate_csv} не найден")
        if not os.path.exists(self.processed_csv):
            raise FileNotFoundError(f"Файл {self.processed_csv} не найден")

        index = faiss.read_index(self.index_path)

        # Проверка формы и типа query_vector
        query_vector = np.array(query_vector).astype('float32')
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape[1] != index.d:
            raise ValueError(f"Ожидаемая размерность query_vector: {index.d}, получено: {query_vector.shape[1]}")
        if query_vector.shape[0] != 1:
            raise ValueError(f"Ожидается один вектор запроса, получено: {query_vector.shape[0]}")

        # Установка количества кластеров для поиска (баланс скорости/точности)
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
                        'distance': float(dist)
                    })

        # Формирование читаемого вывода
        print(f"\nТоп-{min(k, len(matches))} ближайших пользователей (до ранжирования):")
        for i, res in enumerate(matches, 1):
            print(f"{i}. user_id: {res['user_id']}, Пол: {res['sex']}, "
                  f"Должность: {res['job.title']}, Организация: {res['organization']}, "
                  f"Зарплата: {res['annual.salary']}, Возраст: {res['age']}, "
                  f"Вопрос: {res['question']}, Дистанция: {res['distance']:.4f}")

        query_features = processed_df[processed_df['user_id'] == matches[0]['user_id']][
            ['sentiment_pos', 'sentiment_neu', 'sentiment_neg', 'question_length', 'question_words', 'formality_score', 'topic_norm']
        ].values[0]

        feature_columns = ['sentiment_pos', 'sentiment_neu', 'sentiment_neg', 'question_length', 'question_words',
                           'formality_score', 'topic_norm']
        if query_features.shape != (len(feature_columns),):
            raise ValueError(
                f"Ожидается query_features размерности ({(len(feature_columns),)}), получено: {query_features.shape}")

        ranked_matches = self.rank_matches(matches, query_features, processed_df, top_k=10)

        with open(f'../results/search_results_user_{query_user_id}.json', 'w', encoding='utf-8') as f:
            json.dump(ranked_matches, f, ensure_ascii=False, indent=4)
        print(f"Результаты поиска для user_id {query_user_id} сохранены в ../results/search_results_user_{query_user_id}.json")

        return ranked_matches, indices[0].tolist(), distances[0].tolist()

    def search_by_user_id(
            self,
            user_id: int,
            k: int = 50,
            nprobe: int = 10
    ) -> Tuple[List[Dict], List[int], List[float]]:
        """
        Функция ищет топ-K метчей в FAISS IndexIVFFlat по user_id

        :param user_id: id пользователя, для которого выполняется поиск
        :param k: Количество соседних до ранжирования
        :param nprobe: Количество кластеров для приближённого (ANN) поиска
        :return: Возвращает кортеж: список индексов метчей и список расстояния до соответсвующих метчей
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Индекс {self.index_path} не найден")
        if not os.path.exists(self.intermediate_csv):
            raise FileNotFoundError(f"Файл {self.intermediate_csv} не найден")
        if not os.path.exists(self.processed_csv):
            raise FileNotFoundError(f"Файл {self.processed_csv} не найден")
        if not os.path.exists(self.vectors_path):
            raise FileNotFoundError(f"Файл {self.vectors_path} не найден")

        processed_df = pd.read_csv(self.processed_csv)
        if 'user_id' not in processed_df.columns:
            raise ValueError("Столбец 'user_id' не найден в processed_csv")
        if user_id not in processed_df['user_id'].values:
            raise ValueError(f"user_id {user_id} не найден в processed_csv")

        vectors = np.load(self.vectors_path, allow_pickle=True).astype('float32')
        idx = processed_df[processed_df['user_id'] == user_id].index
        if len(idx) != 1:
            raise ValueError(f"Ожидается ровно один индекс для user_id {user_id}, найдено: {len(idx)}")
        idx = idx[0]
        if idx >= len(vectors):
            raise ValueError(f"Индекс {idx} выходит за пределы массива векторов (длина: {len(vectors)})")

        query_vector = vectors[idx]
        ranked_matches, indices, distances = self.search_faiss_ivf(
            query_vector=query_vector,
            query_user_id=user_id,
            k=k,
            nprobe=nprobe
        )

        return ranked_matches, indices, distances

    def rank_matches(
            self,
            matches: List[Dict],
            query_features: np.ndarray,
            processed_df: pd.DataFrame,
            top_k: int = 10,
            weights: Dict[str, float] = None
    ) -> List[Dict]:
        """
        Ранжирование метчей на основе признаков (тональность, стиль, тематика)

        :param matches: Список словарей, где каждый словарь — это матч от FAISS
        :param query_features: NumPy-массив размером (7,) с нормализованными признаками запроса
        :param processed_df: Pandas DataFrame из файла processed_profiles.csv, содержащий нормализованные признаки для всех пользователей
        :param top_k: Количество возвращаемых метчей после ранжирования
        :param weights: Словарь с весами для признаков (sentiment, style, topic)
        :return: Список из top_k словарей с метаданными матчей, дополненных полем relevance_score (скор релевантности)
        """
        # Настройка весов
        if weights is None:
            weights = {'sentiment': 0.4, 'style': 0.4, 'topic': 0.2}

        if not all(k in weights for k in ['sentiment', 'style', 'topic']):
            raise ValueError("Веса должны содержать ключи: 'sentiment', 'style', 'topic'")
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("Сумма весов должна быть равна 1")

        # Проверка входных данных
        feature_columns = ['sentiment_pos', 'sentiment_neu', 'sentiment_neg',
                           'question_length', 'question_words', 'formality_score', 'topic_norm']
        missing_columns = [col for col in feature_columns if col not in processed_df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют столбцы в processed_df: {missing_columns}")
        if query_features.shape != (len(feature_columns),):
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
                'distance': match['distance'],
                'relevance_score': relevance_score
            })

        # Сортировка и выбор топ-k
        ranked_matches = sorted(ranked_matches, key=lambda x: x['relevance_score'], reverse=True)[:top_k]

        # Вывод и сохранение результатов
        print(f"\nТоп-{top_k} метчей после ранжирования:")
        for i, match in enumerate(ranked_matches, 1):
            print(f"{i}. user_id: {match['user_id']}, Пол: {match['sex']}, "
                  f"Должность: {match['job.title']}, Организация: {match['organization']}, "
                  f"Зарплата: {match['annual.salary']}, Возраст: {match['age']}, "
                  f"Вопрос: {match['question']}, Дистанция: {match['distance']:.4f}, "
                  f"Релевантность: {match['relevance_score']:.4f}")

        return ranked_matches
