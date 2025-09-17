from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from data_processing import DataProcessor
from ANN_search import FaissIndexManager
from feedback import FeedbackManager


# Pydantic-схемы
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

processor = DataProcessor()
faiss_manager = FaissIndexManager()
feedback_manager = FeedbackManager(processor=processor, faiss_manager=faiss_manager)


# Функция зависимости
def get_feedback_manager():
    return feedback_manager


# Инициализация FastAPI
app = FastAPI(title="Feedback API")


# Эндпоинт
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
