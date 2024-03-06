from fastapi import APIRouter
from .catdog_cls_router import router as catdog_cls_router
from .redirect_router import router as redirect_router

router = APIRouter()
router.include_router(catdog_cls_router, prefix="/catdog_classification")
router.include_router(redirect_router, tags=["redirect"])
