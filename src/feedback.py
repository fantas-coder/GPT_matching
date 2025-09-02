from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

from config import (os, pd, logging, List, Dict,
                    FEEDBACK_MODEL, SENTIMENT_MODEL_TASK, SENTIMENT_MODEL)


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeedbackManager:
    def __init__(self, processor, faiss_manager, weights: Dict[str, float] = None):
        self.processor = processor
        self.faiss_manager = faiss_manager
        self.weights = weights if weights else {'sentiment': 0.4, 'style': 0.4, 'topic': 0.2}
        self.best_matches = []
        self.good_matches = []
        self.medium_matches = []
        self.bad_matches = []
        self.very_bad_matches = []
        self.output_dir = '../data/saved_matches'
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model = SentenceTransformer(FEEDBACK_MODEL)
        self.sentiment_model = pipeline(  # Модель для извлечения тональности
            task=SENTIMENT_MODEL_TASK,
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            device=self.device
        )
        self.reference_phrases = [
            # Рейтинг 5: Позитивные слова и фразы
            "отлично", "круто", "супер", "всё отлично", "просто класс", "замечательно", "шикарно", "превосходно",
            "всё супер",
            # Рейтинг 4: Хорошие слова и фразы
            "хорошо", "нормально", "всё хорошо", "неплохо", "окей", "прилично", "достойно",
            # Рейтинг 3: Нейтральные слова и фразы
            "средне", "норм", "так себе", "всё норм", "сойдёт", "посредственно", "на троечку",
            # Рейтинг 2: Негативные слова и фразы
            "плохо", "не очень", "не то", "такое себе", "не особо", "не айс",
            # Рейтинг 1: Сильно негативные слова и фразы
            "ужасно", "отстой", "просто ужас", "кошмар", "полный отстой", "фу", "ужас"
        ]
        self.reference_embeddings = self.embedding_model.encode(self.reference_phrases)

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

        # Вывод и сохранение результатов
        logger.info(f"\nТоп-{len(ranked_matches)} матчей после ранжирования:")
        for i, match in enumerate(ranked_matches, 1):
            logger.info(f"{i}.{'\n'}user_id: {match['user_id']}{'\n'}Пол: {match['sex']}{'\n'}"
                        f"Должность: {match['job.title']}{'\n'}Организация: {match['organization']}{'\n'}"
                        f"Зарплата: {match['annual.salary']}{'\n'}Возраст: {match['age']}{'\n'}"
                        f"Вопрос: {match['question']}{'\n'}Ключевые слова вопроса: {match['question_keywords']}{'\n'}"
                        f"Ключевые слова темы: {match['topic_keywords']}{'\n'}"
                        f"X: {match['X']}, Y: {match['Y']}{'\n'}Z: {match['Z']}{'\n'}"
                        f"Дистанция: {match['distance']:.4f}{'\n'}Релевантность: {match['relevance_score']:.4f}{'\n'}"
                        f"Объяснение: {match['explanation']}{'\n'}")

        while True:
            user_input = input(
                "Введите оценку (1-5 или в свободной форме, например 'отлично' или 'Это было круто!'): ").strip().lower()
            rating, score_100 = self._parse_rating(user_input)
            if rating is not None:
                logger.info(f"Распознанная оценка: {rating} (score: {score_100:.1f}/100)")
                break
            logger.warning("Не удалось распознать оценку. Попробуйте снова (например, '4' или 'хорошо').")

        self.process_feedback(user_id_query, ranked_matches, rating)
        logger.info(f"Оценка {rating} применена ко всем матчам")

    def _parse_rating(self, user_input: str) -> tuple[int | None, float | None]:
        """Парсит ввод: число → напрямую, текст → эмбеддинги + sentiment."""
        # Пустая строка
        if not user_input:
            logger.warning("Пустой ввод")
            return None, None

        # Проверка на число
        try:
            rating = int(user_input)
            if 1 <= rating <= 5:
                logger.info(f"Введено число: {rating}")
                return rating, rating * 20  # 1→20, 5→100
            logger.warning(f"Число {user_input} вне диапазона 1-5")
        except ValueError:
            pass

        # Эмбеддинги
        input_embedding = self.embedding_model.encode(user_input)
        similarities = util.cos_sim(input_embedding, self.reference_embeddings)[0]
        max_sim = similarities.max().item()
        best_idx = torch.argmax(similarities).item()
        logger.info(f"Максимальное косинусное сходство: {max_sim:.2f} (фраза: {self.reference_phrases[best_idx]})")

        # Sentiment-анализ
        try:
            sentiment_result = self.sentiment_model(user_input)[0]
            label = sentiment_result['label'].upper()
            sentiment_score = sentiment_result['score']
            logger.info(f"Sentiment: {label}, score: {sentiment_score:.2f}")
            if label == 'NEGATIVE':
                score_100 = (1 - sentiment_score) * 50  # 0-50
            elif label == 'POSITIVE':
                score_100 = 50 + sentiment_score * 50  # 50-100
            else:
                score_100 = 50  # Нейтрал
        except Exception as e:
            logger.error(f"Ошибка sentiment-анализа: {e}")
            return None, None

        # Определяем рейтинг по индексам reference_phrases
        rating_ranges = [
            (0, 8, 5),  # индексы 0-8 → рейтинг 5
            (9, 15, 4),  # индексы 9-15 → рейтинг 4
            (16, 22, 3),  # индексы 16-22 → рейтинг 3
            (23, 28, 2),  # индексы 23-28 → рейтинг 2
            (29, len(self.reference_phrases) - 1, 1)  # индексы 29+ → рейтинг 1
        ]

        # Логика рейтинга
        if max_sim >= 0.5 and max_sim >= sentiment_score:
            # Высокое сходство и эмбеддинги увереннее
            for start, end, rating in rating_ranges:
                if start <= best_idx <= end:
                    logger.info(
                        f"Использую рейтинг по эмбеддингам: {rating} (ближайшая фраза: {self.reference_phrases[best_idx]}, max_sim: {max_sim:.2f})")
                    return rating, score_100
            # Если индекс не попал в диапазон (крайний случай)
            logger.warning(f"Индекс {best_idx} вне диапазонов, использую sentiment")

        # Низкое сходство: рейтинг по sentiment
        logger.warning(f"Низкое сходство ({max_sim:.2f}), использую sentiment")
        if score_100 > 80:
            rating = 5
        elif score_100 > 60:
            rating = 4
        elif score_100 > 40:
            rating = 3
        elif score_100 > 20:
            rating = 2
        else:
            rating = 1

        return rating, score_100

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
        1- переобучаем word2vec + п.2

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
            self.save_matches(self.good_matches, 'good_matches.csv')
            weights = self.faiss_manager.load_weights()
            weights['style'] = min(weights['style'] + 0.05, 1.0)
            total = weights['style'] + weights['sentiment'] + weights['topic']
            weights = {k: v / total for k, v in weights.items()}
            self.faiss_manager.save_weights(weights)
            logger.info(
                f"Увеличен вес стиля: {weights['style']:.2f}, сохранено {len(match_data_list)} матчей в good_matches.csv")
        elif rating == 3:
            self.medium_matches.extend(match_data_list)
            self.save_matches(self.medium_matches, 'medium_matches.csv')
            nprobe = min(self.faiss_manager.load_nprobe() + 5, 50)
            self.faiss_manager.save_nprobe(nprobe)
            logger.info(
                f"Увеличен nprobe до {nprobe}, сохранено {len(match_data_list)} матчей в medium_matches.csv")
        elif rating == 2:
            self.bad_matches.extend(match_data_list)
            self.save_matches(self.bad_matches, 'bad_matches.csv')
            nprobe = min(self.faiss_manager.load_nprobe() + 5, 50)
            self.faiss_manager.save_nprobe(nprobe)
            logger.info(
                f"Увеличен nprobe до {nprobe}, сохранено {len(match_data_list)} матчей в medium_matches.csv")
            try:
                self.processor.retrain_word2vec()
                logger.info("Word2Vec переобучен")
            except Exception as e:
                logger.error(f"Ошибка переобучения Word2Vec: {e}")
        elif rating == 1:
            self.very_bad_matches.extend(match_data_list)
            self.save_matches(self.very_bad_matches, 'very_bad_matches.csv')
            try:
                self.processor.retrain_word2vec()
                logger.info("Word2Vec переобучен")
            except Exception as e:
                logger.error(f"Ошибка переобучения Word2Vec: {e}")
            try:
                self.faiss_manager.build_faiss_ivf_index()
                nprobe = min(self.faiss_manager.load_nprobe() + 5, 50)
                self.faiss_manager.save_nprobe(nprobe)
                logger.info(f"FAISS переобучен, nprobe увеличен до {nprobe}")
            except Exception as e:
                logger.error(f"Ошибка переобучения FAISS: {e}")

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
