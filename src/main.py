# Стандартные библиотеки
import numpy as np
from typing import Tuple

# Конфигурация
from src.config import os, pd
from src.data_processing import DataProcessor
from src.ANN_search import FaissIndexManager


def add_new_record(
        record: dict,
        processor: DataProcessor = None,
        faiss_manager: FaissIndexManager = None,
        intermediate_path: str = '../data/intermediate_dataset.csv',
        existing_csv: str = '../data/processed_profiles.csv',
        existing_vectors: str = '../artifacts/user_vectors.npy',
        index_path: str = '../artifacts/faiss_index_ivf.index',
        nlist: int = 100
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Функция добавление новую записи в базу данных

    :param record: Данные пользователя (Профиль + Текст)
    :param processor: Экземпляр класса обработки БД
    :param faiss_manager: Экземпляр класса обучения FAISS
    :param intermediate_path: Объединённая БД (Профиль + Текст)
    :param existing_csv: Путь до текущей векторизованной БД
    :param existing_vectors: Путь до текущих финальных векторов
    :param index_path: Путь до обученного индекса FAISS
    :param nlist: Количество кластеров для обучения FAISS
    :return: Кортеж новой векторизованной БД и новых финальных векторов
    """
    required_keys = ['age', 'sex', 'job.title', 'organization', 'annual.salary', 'question']
    missing_keys = [key for key in required_keys if key not in record]
    if missing_keys:
        raise ValueError(f"Отсутствуют обязательные поля: {missing_keys}")

    # Валидация данных
    if record['age'] < 0:
        raise ValueError("Возраст не может быть отрицательным")
    if record['annual.salary'] < 0:
        raise ValueError("Зарплата не может быть отрицательной")
    if record['sex'] not in ['Male', 'Female']:
        raise ValueError("Пол должен быть 'Male' или 'Female'")
    if not record['question'] or not isinstance(record['question'], str):
        raise ValueError("Вопрос должен быть непустой строкой")
    if not record['job.title'] or not isinstance(record['job.title'], str):
        raise ValueError("Должность должна быть непустой строкой")

    # Инициализация процессора, если он не создан
    if processor is None:
        processor = DataProcessor()
        processor.load_artifacts()

    # Проверка наличия моделей и нормализаторов
    if processor.job_model is None or any(scaler is None for scaler in processor.scalers.values()):
        raise ValueError(
            "Не удалось загрузить необходимые модели или нормализаторы. Сначала выполните обучение с помощью process_data.")

    # Создание DataFrame для новой записи
    new_df = pd.DataFrame([record])
    new_df['user_id'] = new_df.index + (pd.read_csv(existing_csv)['user_id'].max() + 1 if os.path.exists(existing_csv) else 0)

    # Кодирование и нормализация
    new_df = processor.encode_features(new_df, train_mode=False)
    new_df = processor.normalize_features(new_df, train_mode=False)

    # Создание вектора для новой записи
    new_vector = processor.create_vectors(new_df)

    # Проверка формы и типа new_vector
    if len(new_vector.shape) != 2 or new_vector.shape[0] != 1:
        raise ValueError(f"Ожидается new_vector формы (1, d), получено: {new_vector.shape}")
    if new_vector.dtype != np.float32:
        new_vector = new_vector.astype('float32')

    # Загрузка существующих данных
    if os.path.exists(existing_csv) and os.path.exists(existing_vectors):
        existing_df = pd.read_csv(existing_csv)
        existing_vectors_array = np.load(existing_vectors, allow_pickle=False)
        if existing_vectors_array.dtype != np.float32:
            existing_vectors_array = existing_vectors_array.astype('float32')
        if new_vector.shape[1] != existing_vectors_array.shape[1]:
            raise ValueError(
                f"Размерность нового вектора ({new_vector.shape[1]}) не соответствует существующим данным ({existing_vectors_array.shape[1]})")
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_vectors = np.vstack([existing_vectors_array, new_vector])
    else:
        updated_df = new_df
        updated_vectors = new_vector

    # Сохранение промежуточной БД

    if os.path.exists(intermediate_path):
        intermediate_df = pd.read_csv(intermediate_path)
        new_intermediate = pd.DataFrame([record])
        new_intermediate['user_id'] = new_df['user_id']
        updated_intermediate = pd.concat([intermediate_df, new_intermediate], ignore_index=True)
        updated_intermediate.to_csv(intermediate_path, index=False, encoding='utf-8')
        print(f"Промежуточная БД обновлена в {intermediate_path}")

    # Сохранение обновленных данных
    updated_df.to_csv(existing_csv, index=False)
    np.save(existing_vectors, updated_vectors)

    # Инициализация процессора, если он не создан
    if faiss_manager is None:
        faiss_manager = FaissIndexManager()

    # Переобучение FAISS-индекса
    print(f"Переобучение FAISS-индекса с учетом всех векторов ({updated_vectors.shape[0]} записей)...")
    faiss_manager.build_faiss_ivf_index(nlist=nlist)

    print(f"Новая запись добавлена. Обновлены {existing_csv}, {existing_vectors}, {intermediate_path}, {index_path}")
    return updated_df, updated_vectors


def main(train_flag=False):
    processor = DataProcessor()
    processor.load_artifacts()
    faiss_manager = FaissIndexManager()

    if train_flag:
        df, vectors = processor.process_data(
            input_path='../data/atlanta_salary_data_2015_full.csv',
            output_path='../data/processed_profiles.csv',
            intermediate_path='../data/intermediate_dataset.csv'
        )
        print(f"Созданы векторы размерности: {vectors.shape}")
        print(f"Пример вектора: {vectors[0]}")
        faiss_manager.build_faiss_ivf_index()
    else:
        # Пример добавления новой записи
        new_record = {
            'age': 26,
            'sex': 'Male',
            'job.title': 'data analytic',
            'organization': 'sber',
            'annual.salary': 500000,
            'question': 'Ку! Тоже хочу 10 метчей!!'
        }
        df, vectors = add_new_record(record=new_record, processor=processor, faiss_manager=faiss_manager)
        print(f"Обновленная БД:\n{df.tail(1)}")
        print(f"Обновленные векторы размерностью: {vectors.shape}")

    # Пример поиска по user_id
    # df = pd.read_csv("../data/processed_profiles.csv")
    user_id = df['user_id'].iloc[-1]  # Берем user_id последней записи
    ranked_matches, indices, distances = faiss_manager.search_by_user_id(user_id=user_id, k=50)
    print(f"Ранжированные результаты поиска для user_id {user_id}:")
    for i, res in enumerate(ranked_matches, 1):
        print(f"{i}. user_id: {res['user_id']}, Пол: {res['sex']}, "
              f"Должность: {res['job.title']}, Организация: {res['organization']}, "
              f"Зарплата: {res['annual.salary']}, Возраст: {res['age']}, "
              f"Вопрос: {res['question']}, Дистанция: {res['distance']:.4f}, "
              f"Релевантность: {res['relevance_score']:.4f}")

    # Визуализация сохраненных векторов
    processor.visualize_saved_vectors()


if __name__ == "__main__":
    main(train_flag=False)
