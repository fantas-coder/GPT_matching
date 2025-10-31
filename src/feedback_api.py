from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from data_processing import DataProcessor
from ANN_search import FaissIndexManager
from feedback import FeedbackManager

import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Pydantic-схемы /feedback
class FeedbackRequest(BaseModel):
    user_id: int
    match_id: int
    feedback: str
    lang: str = "ru"


class FeedbackResponse(BaseModel):
    success: bool
    rating: int | None
    score: float | None
    message: str

# Pydantic-схемы для /best-match
class BestMatchRequest(BaseModel):
    user_id: int

class BestMatchResponse(BaseModel):
    success: bool
    user_id: int | None
    match_id: int | None
    sex: str | None
    job_title: str | None
    organization: str | None
    annual_salary: float | None
    age: int | None
    question: str | None
    X: float | None
    Y: float | None
    Z: float | None
    distance: float | None
    relevance_score: float | None
    explanation: str | None
    message: str


# Инициализация объектов
processor = DataProcessor()
faiss_manager = FaissIndexManager()
feedback_manager = FeedbackManager(processor=processor, faiss_manager=faiss_manager)


# Функции зависимости
def get_feedback_manager():
    return feedback_manager


def get_faiss_manager():
    return faiss_manager


# Инициализация FastAPI
app = FastAPI(title="Feedback API")


# Эндпоинт для фидбека
@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest, manager: FeedbackManager = Depends(get_feedback_manager)):
    rating, score_100, message = manager.collect_feedback_api(
        user_id=request.user_id,
        match_id=request.match_id,
        feedback=request.feedback,
        lang=request.lang
    )
    if rating is None:
        raise HTTPException(status_code=400, detail=message)
    return FeedbackResponse(success=True, rating=rating, score=score_100, message=message)


# Новый эндпоинт для получения лучшего матча
@app.post("/best-match", response_model=BestMatchResponse)
async def get_best_match(request: BestMatchRequest, faiss_manager: FaissIndexManager = Depends(get_faiss_manager)):
    logger.info(f"Поиск лучшего матча для user_id {request.user_id}...")
    try:
        ranked_matches, indices, distances = faiss_manager.search_by_user_id(user_id=request.user_id, k=50)
        if not ranked_matches:
            logger.warning(f"Матчи для user_id {request.user_id} не найдены")
            return JSONResponse(
                content=BestMatchResponse(
                    success=False,
                    user_id=request.user_id,
                    match_id=None,
                    sex=None,
                    job_title=None,
                    organization=None,
                    annual_salary=None,
                    age=None,
                    question=None,
                    X=None,
                    Y=None,
                    Z=None,
                    distance=None,
                    relevance_score=None,
                    explanation=None,
                    message="Матчи для указанного user_id не найдены"
                ).dict(),
                headers={"Content-Type": "application/json; charset=utf-8"}
            )

        # Берём первый (лучший) матч
        best_match = ranked_matches[0]

        return JSONResponse(
            content=BestMatchResponse(
                success=True,
                user_id=request.user_id,
                match_id=best_match['user_id'],
                sex=best_match['sex'],
                job_title=best_match['job.title'],
                organization=best_match['organization'],
                annual_salary=best_match['annual.salary'],
                age=best_match['age'],
                question=best_match['question'],
                X=best_match['X'],
                Y=best_match['Y'],
                Z=best_match['Z'],
                distance=best_match['distance'],
                relevance_score=best_match['relevance_score'],
                explanation=best_match['explanation'],
                message="Лучший матч успешно найден"
            ).dict(),  headers={"Content-Type": "application/json; charset=utf-8"}
        )
    except Exception as e:
        logger.error(f"Ошибка при поиске матча для user_id {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")