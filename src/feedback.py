from src.config import os, pd, logging, List, Dict


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeedbackManager:
    def __init__(self, processor, faiss_manager, weights: Dict[str, float] = None):
        self.processor = processor    # Экземпляр DataProcessor для переобучения моделей
        self.faiss_manager = faiss_manager      # Экземпляр FaissIndexManager
        self.weights = weights if weights else {'sentiment': 0.4, 'style': 0.4, 'topic': 0.2}
        self.best_matches = []
        self.good_matches = []
        self.medium_matches = []
        self.bad_matches = []
        self.very_bad_matches = []
        self.output_dir = '../data'
        os.makedirs(self.output_dir, exist_ok=True)

    def collect_feedback(
            self,
            ranked_matches: List[Dict],
            user_id_query: int
    ) -> None:
        """
        Собирает оценку через консоль для всех ранжированных матчей и выполняет действия

        :param ranked_matches: Список ранжированных матчей из search_by_user_id
        :param user_id_query: ID пользователя запроса
        """
        if not ranked_matches:
            logger.warning("Нет матчей для оценки")
            return

        logger.info("\nТоп-10 релевантных матчей:")
        for i, match in enumerate(ranked_matches, 1):
            logger.info(f"{i}. user_id: {match['user_id']}, Пол: {match['sex']}, "
                        f"Должность: {match['job.title']}, Организация: {match['organization']}, "
                        f"Зарплата: {match['annual.salary']}, Возраст: {match['age']}, "
                        f"Вопрос: {match['question']}, Релевантность: {match['relevance_score']:.4f}")

        while True:
            try:
                rating = int(input(f"Введите одну оценку для всех матчей (1-5): "))
                if 0 < rating < 6:
                    break
                logger.error("Оценка должна быть от 1 до 5")
            except ValueError:
                logger.error("Введите целое число от 1 до 5")

        self.process_feedback(user_id_query, ranked_matches, rating)
        logger.info(f"Оценка {rating} применена ко всем матчам")

    def process_feedback(
            self,
            user_id_query: int,
            matches: List[Dict],
            rating: int
    ) -> None:
        """
        Обрабатывает оценку для списка матчей и выполняет соответствующие действия:
        5- ничего не делаем
        4- изменяем веса ранжирования
        3- изменяем кол-во кластеров поиска
        2- переобучаем word2vec + п.3
        1- переобучаем все модели + п.3

        :param user_id_query: ID пользователя запроса
        :param matches: Список словарей с данными матчей
        :param rating: Оценка (1-5)
        """
        match_data_list = [
            {
                'user_id_query': user_id_query,
                'user_id_match': match['user_id'],
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
                'relevance_score': match['relevance_score']
            } for match in matches
        ]

        if rating == 5:
            self.best_matches.extend(match_data_list)
            self.save_matches(self.best_matches, 'best_matches.csv')
            logger.info(f"Добавлено {len(match_data_list)} матчей в лучшие примеры")
        elif rating == 4:
            self.good_matches.extend(match_data_list)
            self.weights['style'] = min(self.weights['style'] + 0.05, 1.0)
            total = self.weights['style'] + self.weights['sentiment'] + self.weights['topic']
            self.weights = {k: v / total for k, v in self.weights.items()}
            self.save_matches(self.good_matches, 'good_matches.csv')
            logger.info(
                f"Увеличен вес стиля: {self.weights['style']:.2f}, сохранено {len(match_data_list)} матчей в good_matches.csv")
        elif rating == 3:
            self.medium_matches.extend(match_data_list)
            self.faiss_manager.nprobe = min(getattr(self.faiss_manager, 'nprobe', 10) + 5, 50)
            self.save_matches(self.medium_matches, 'medium_matches.csv')
            logger.info(
                f"Увеличен nprobe до {self.faiss_manager.nprobe}, сохранено {len(match_data_list)} матчей в medium_matches.csv")
        elif rating == 2:
            self.bad_matches.extend(match_data_list)
            self.save_matches(self.bad_matches, 'bad_matches.csv')
            self.faiss_manager.nprobe = min(getattr(self.faiss_manager, 'nprobe', 10) + 5, 50)
            try:
                self.processor.retrain_word2vec()
                logger.info("Word2Vec переобучен на плохих примерах")
            except Exception as e:
                logger.error(f"Ошибка переобучения Word2Vec: {e}")
        elif rating == 1:
            self.very_bad_matches.extend(match_data_list)
            self.save_matches(self.very_bad_matches, 'very_bad_matches.csv')
            try:
                self.processor.retrain_all_models()
                self.faiss_manager.build_faiss_ivf_index()
                self.faiss_manager.nprobe = min(getattr(self.faiss_manager, 'nprobe', 10) + 5, 50)
                logger.info(f"Все модели и FAISS переобучены, nprobe увеличен до {self.faiss_manager.nprobe}")
            except Exception as e:
                logger.error(f"Ошибка переобучения моделей и FAISS: {e}")

    def save_matches(
            self,
            matches: List[Dict],
            filename: str
    ) -> None:
        """
        Сохраняет матчи в CSV-файл

        :param matches: Список матчей
        :param filename: Имя файла (e.g., best_matches.csv)
        """
        try:
            df = pd.DataFrame(matches)
            df.to_csv(os.path.join(self.output_dir, filename), index=False, encoding='utf-8')
            logger.info(f"Сохранено {len(matches)} записей в {filename}")
        except Exception as e:
            logger.error(f"Ошибка сохранения {filename}: {e}")
