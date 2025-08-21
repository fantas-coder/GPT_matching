# Конфигурация
from src.config import os, pd, np, logging, Tuple
from src.data_processing import DataProcessor
from src.ANN_search import FaissIndexManager
from src.feedback import FeedbackManager

import faiss
from cachetools import LRUCache


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_new_record(
        record: dict,
        processor: DataProcessor = None,
        faiss_manager: FaissIndexManager = None,
        intermediate_path: str = '../data/intermediate_dataset.csv',
        vectors_csv: str = '../data/processed_profiles.csv',
        vectors_path: str = '../artifacts/user_vectors.npy',
        index_path: str = '../artifacts/faiss_index_ivf.index',
        nlist: int = 100
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Функция добавление новую записи в базу данных

    :param record: Данные пользователя (Профиль + Текст)
    :param processor: Экземпляр класса обработки БД
    :param faiss_manager: Экземпляр класса обучения FAISS
    :param intermediate_path: Объединённая БД (Профиль + Текст)
    :param vectors_csv: Путь до текущей векторизованной БД
    :param vectors_path: Путь до текущих финальных векторов
    :param index_path: Путь до обученного индекса FAISS
    :param nlist: Количество кластеров для обучения FAISS
    :return: Кортеж новой векторизованной БД и новых финальных векторов
    """
    required_keys = ['age', 'sex', 'job.title', 'organization', 'annual.salary', 'question']
    missing_keys = [key for key in required_keys if key not in record]
    if missing_keys:
        logger.error(f"Отсутствуют обязательные поля: {missing_keys}")
        raise ValueError(f"Отсутствуют обязательные поля: {missing_keys}")

    # Валидация данных
    if record['age'] < 0:
        logger.error("Возраст не может быть отрицательным")
        raise ValueError("Возраст не может быть отрицательным")
    if record['annual.salary'] < 0:
        logger.error("Зарплата не может быть отрицательной")
        raise ValueError("Зарплата не может быть отрицательной")
    if record['sex'] not in ['Male', 'Female']:
        logger.error("Пол должен быть 'Male' или 'Female'")
        raise ValueError("Пол должен быть 'Male' или 'Female'")
    if not record['question'] or not isinstance(record['question'], str):
        logger.error("Вопрос должен быть непустой строкой")
        raise ValueError("Вопрос должен быть непустой строкой")
    if not record['job.title'] or not isinstance(record['job.title'], str):
        logger.error("Должность должна быть непустой строкой")
        raise ValueError("Должность должна быть непустой строкой")
    if not (0 <= record['X'] <= 1000 and 0 <= record['Y'] <= 1000 and 0 <= record['Z'] <= 1000):
        logger.error("Значения X, Y, Z должны быть в диапазоне [0, 1000]")
        raise ValueError("Значения X, Y, Z должны быть в диапазоне [0, 1000]")

    # Инициализация процессора, если он не создан
    if processor is None:
        processor = DataProcessor()
        processor.load_artifacts()

    # Проверка наличия моделей и нормализаторов
    if processor.job_model is None or any(scaler is None for scaler in processor.scalers.values()):
        logger.error(
            "Не удалось загрузить необходимые модели или нормализаторы. Сначала выполните обучение с помощью process_data.")
        raise ValueError(
            "Не удалось загрузить необходимые модели или нормализаторы. Сначала выполните обучение с помощью process_data.")

    # Создание DataFrame для новой записи
    new_df = pd.DataFrame([record])
    new_df['user_id'] = new_df.index + (pd.read_csv(vectors_csv)['user_id'].max() + 1 if os.path.exists(vectors_csv) else 0)

    # Кодирование и нормализация
    new_df = processor.encode_features(new_df, train_mode=False)
    new_df = processor.normalize_features(new_df, train_mode=False)

    # Создание вектора для новой записи
    new_vector = processor.create_vectors(new_df)

    # Проверка формы и типа new_vector
    if len(new_vector.shape) != 2 or new_vector.shape[0] != 1:
        logger.error(f"Ожидается new_vector формы (1, d), получено: {new_vector.shape}")
        raise ValueError(f"Ожидается new_vector формы (1, d), получено: {new_vector.shape}")
    if new_vector.dtype != np.float32:
        new_vector = new_vector.astype('float32')

    # Загрузка существующих данных
    if os.path.exists(vectors_csv) and os.path.exists(vectors_path):
        existing_df = pd.read_csv(vectors_csv)
        existing_vectors_array = np.load(vectors_path, allow_pickle=False)
        if existing_vectors_array.dtype != np.float32:
            existing_vectors_array = existing_vectors_array.astype('float32')
        if new_vector.shape[1] != existing_vectors_array.shape[1]:
            logger.error(
                f"Размерность нового вектора ({new_vector.shape[1]}) не соответствует существующим данным ({existing_vectors_array.shape[1]})")
            raise ValueError(
                f"Размерность нового вектора ({new_vector.shape[1]}) не соответствует существующим данным ({existing_vectors_array.shape[1]})")
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_vectors = np.vstack([existing_vectors_array, new_vector])
    else:
        updated_df = new_df
        updated_vectors = new_vector

    # Сохранение промежуточной БД
    os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
    if os.path.exists(intermediate_path):
        intermediate_df = pd.read_csv(intermediate_path)
        new_intermediate = pd.DataFrame([record])
        new_intermediate['user_id'] = new_df['user_id']
        updated_intermediate = pd.concat([intermediate_df, new_intermediate], ignore_index=True)
        updated_intermediate.to_csv(intermediate_path, index=False, encoding='utf-8')
        logger.info(f"Промежуточная БД обновлена в {intermediate_path}")

    # Сохранение обновленных данных
    os.makedirs(os.path.dirname(vectors_csv), exist_ok=True)
    updated_df.to_csv(vectors_csv, index=False)

    os.makedirs(os.path.dirname(vectors_path), exist_ok=True)
    np.save(vectors_path, updated_vectors)

    # Инициализация Faiss, если он не создан
    if faiss_manager is None:
        faiss_manager = FaissIndexManager()

    # Добавление вектора в FAISS-индекс
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        index.add(new_vector)
        faiss.write_index(index, index_path)
        logger.info(f"Вектор добавлен в существующий FAISS-индекс: {index_path}")
    else:
        logger.info("Индекс не существует, создается новый...")
        if faiss_manager is None:
            faiss_manager = FaissIndexManager()
        faiss_manager.build_faiss_ivf_index(nlist=nlist)

    # Очистка кеша
    faiss_manager.cache = LRUCache(maxsize=1000)
    faiss_manager.save_cache()
    logger.info("Кеш очищен после добавления новой записи")

    logger.info(
        f"Новая запись добавлена. Обновлены {vectors_csv}, {vectors_path}, {intermediate_path}, {index_path}")
    logger.info(f"Обновленная БД:\n{updated_df.tail(1)}")
    logger.info(f"Обновленные векторы размерностью: {updated_vectors.shape}")

    return updated_df, updated_vectors


def main(train_flag=False):
    processor = DataProcessor()
    processor.load_artifacts()
    faiss_manager = FaissIndexManager()
    feedback_manager = FeedbackManager(processor=processor, faiss_manager=faiss_manager)

    if train_flag:
        df, vectors = processor.process_data(
            input_path='../data/atlanta_salary_data_2015_full.csv',
            prepared_path='../data/prepared_data.csv',
            intermediate_path='../data/intermediate_dataset.csv',
            output_path='../data/processed_profiles.csv'
        )

        faiss_manager.build_faiss_ivf_index()
    else:
        # Пример добавления новой записи
        new_record = {
            'age': 22,
            'sex': 'Male',
            'job.title': 'ml data scientist',
            'organization': 'yandex',
            'annual.salary': 300000,
            'question': 'Привет! Хочу найти топ 10 метчей для меня',
            'X': np.random.randint(0, 1001),
            'Y': np.random.randint(0, 1001),
            'Z': np.random.randint(0, 1001)
        }
        df, vectors = add_new_record(
            record=new_record,
            processor=processor,
            faiss_manager=faiss_manager
        )

    # Пример поиска по user_id
    # df = pd.read_csv("../data/processed_profiles.csv")
    user_id = df['user_id'].iloc[-1]  # Берем user_id последней записи
    ranked_matches, indices, distances = faiss_manager.search_by_user_id(user_id=user_id, k=50)
    logger.info(f"Ранжированные результаты поиска для user_id {user_id}:")
    for i, res in enumerate(ranked_matches, 1):
        logger.info(f"{i}. user_id: {res['user_id']}, Пол: {res['sex']}, "
                    f"Должность: {res['job.title']}, Организация: {res['organization']}, "
                    f"Зарплата: {res['annual.salary']}, Возраст: {res['age']}, "
                    f"Вопрос: {res['question']}, X: {res['X']}, Y: {res['Y']}, Z: {res['Z']}, "
                    f"Дистанция: {res['distance']:.4f}, Релевантность: {res['relevance_score']:.4f}")

    # Визуализация сохраненных векторов
    # processor.visualize_saved_vectors()

    # Собираем обратную связь
    feedback_manager.collect_feedback(ranked_matches, user_id)


if __name__ == "__main__":
    main(train_flag=False)
