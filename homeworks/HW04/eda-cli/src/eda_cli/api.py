from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from .core import (
    compute_health_metrics,
    get_missing_data_report,
    extract_dataset_info,
)

# логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eda_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Service is starting up...")
    # Здесь можно загрузить ML-модели или подключиться к БД
    yield
    logger.info("Service is shutting down...")

# FastAPI
app = FastAPI(
    title="Dataset Quality Service",
    description=(
        "HTTP-микросервис для оценки пригодности датасетов к машинному обучению. "
        "Использует эвристический анализ (EDA) вместо тяжелых моделей."
    ),
    version="0.2.1",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

#Pydantic 

class DatasetFeatures(BaseModel):
    """Входные данные: агрегированные характеристики датасета для эвристики."""
    n_rows: int = Field(..., ge=0, description="Общее количество строк")
    n_cols: int = Field(..., ge=0, description="Общее количество колонок")
    max_missing_share: float = Field(..., ge=0.0, le=1.0, description="Максимальная доля пропусков (0.0 - 1.0)")
    numeric_cols: int = Field(..., ge=0, description="Количество числовых признаков")
    categorical_cols: int = Field(..., ge=0, description="Количество категориальных признаков")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "n_rows": 10000,
                "n_cols": 15,
                "max_missing_share": 0.05,
                "numeric_cols": 10,
                "categorical_cols": 5
            }
        }
    )

class ServiceStatus(BaseModel):
    status: str
    service: str
    version: str
    uptime_seconds: float

class QualityResult(BaseModel):
    """Результат проверки качества."""
    ok_for_model: bool = Field(..., description="Годен ли датасет для базового ML")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Оценка качества от 0 до 1")
    message: str = Field(..., description="Текстовое резюме")
    latency_ms: float = Field(..., description="Время обработки запроса (мс)")
    
    #  флаг
    flags: Dict[str, bool] | None = Field(
        default=None, 
        description="Активные флаги проблем (например, has_id_duplicates=True)"
    )
    
    # для проверки файла
    dataset_shape: Dict[str, int] | None = Field(
        default=None, 
        description="Размерность прочитанного датасета"
    )


START_TIME = perf_counter()

def _calculate_score_stub(features: DatasetFeatures) -> float:
    """Логика расчета скора на основе 'сухих' цифр (заглушка)."""
    score = 1.0
    score -= features.max_missing_share
    
    if features.n_rows < 1000:
        score -= 0.2
    if features.n_cols > 100:
        score -= 0.1
    # Штраф за дисбаланс типов (пример логики)
    if features.numeric_cols == 0 and features.categorical_cols > 0:
        score -= 0.1
        
    return max(0.0, min(1.0, score))

# --- Эндпоинты ---

@app.get(
    "/health", 
    response_model=ServiceStatus, 
    tags=["system"],
    summary="Проверка работоспособности"
)
async def health_check():
    """
    Возвращает статус сервиса и время работы (uptime).
    """
    return {
        "status": "ok",
        "service": "dataset-quality-api",
        "version": app.version,
        "uptime_seconds": round(perf_counter() - START_TIME, 2),
    }

@app.post(
    "/quality", 
    response_model=QualityResult, 
    tags=["analysis"],
    summary="Оценка по метаданным (без файла)"
)
async def analyze_metadata(features: DatasetFeatures):
    """
    Быстрая оценка качества на основе переданных метрик, без загрузки самого файла.
    Полезно, если метаданные уже известны.
    """
    t_start = perf_counter()
    logger.info(f"Received metadata request: {features}")

    score = _calculate_score_stub(features)
    is_ok = score >= 0.7
    
    msg = "Dataset looks viable." if is_ok else "Dataset quality is insufficient based on metrics."

    # Генерируем флаги "на лету" для заглушки
    stub_flags = {
        "warning_few_rows": features.n_rows < 1000,
        "warning_many_cols": features.n_cols > 100,
        "critical_missing": features.max_missing_share > 0.5
    }

    t_end = perf_counter()
    
    return QualityResult(
        ok_for_model=is_ok,
        quality_score=round(score, 2),
        message=msg,
        latency_ms=round((t_end - t_start) * 1000, 2),
        flags=stub_flags,
        dataset_shape={"n_rows": features.n_rows, "n_cols": features.n_cols}
    )

@app.post(
    "/quality-from-csv", 
    response_model=QualityResult, 
    tags=["analysis"],
    summary="Полный анализ CSV файла"
)
async def analyze_csv_file(file: UploadFile = File(...)):
    """
    Принимает CSV файл, читает его через Pandas и запускает полное EDA-ядро.
    Возвращает рассчитанный Quality Score и список обнаруженных проблем.
    """
    t_start = perf_counter()
    logger.info(f"Processing file upload: {file.filename}, type={file.content_type}")

    # 1. Валидация расширения
    if not file.filename.lower().endswith(".csv"):
        # Не блокируем жестко (иногда .txt это csv), но предупреждаем в лог
        logger.warning(f"File {file.filename} does not have .csv extension")

    # 2. Чтение файла
    try:
        # Pandas read_csv принимает file-like object (SpoolledTemporaryFile)
        # Пробуем читать. Если файл огромный, в проде это нужно делать чанками или асинхронно.
        df = pd.read_csv(file.file)
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="The uploaded CSV file is empty."
        )
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error for {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Could not parse CSV. Check delimiters and format."
        )
    except Exception as e:
        logger.error(f"Unexpected error reading {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Internal server error while reading file."
        )
    finally:
        # Важно закрыть дескриптор, хотя UploadFile часто делает это сам, но явно лучше
        file.file.close()

    if df.shape[0] == 0:
         raise HTTPException(status_code=400, detail="DataFrame has 0 rows.")

    logger.info(f"Loaded DataFrame: {df.shape}")

    # 3. Запуск логики ядра (функции из core.py)
    try:
        # Получаем структуру
        d_info = extract_dataset_info(df)
        # Получаем данные о пропусках
        miss_rep = get_missing_data_report(df)
        # Считаем метрики здоровья
        health_metrics = compute_health_metrics(d_info, miss_rep)
    except Exception as e:
        logger.exception("Error during EDA core processing")
        raise HTTPException(status_code=500, detail=f"EDA Core Logic Failed: {e}")

    # 4. Формирование ответа
    score = health_metrics.get("quality_index", 0.0)
    is_ok = score >= 0.7
    
    status_msg = (
        "Excellent quality for modeling." if score >= 0.85 else
        "Acceptable quality, minor cleanup needed." if is_ok else
        "Low quality. Please check flags."
    )

    # Фильтруем только булевы флаги для ответа JSON
    flags_cleaned = {
        k: v for k, v in health_metrics.items() 
        if isinstance(v, bool)
    }

    t_end = perf_counter()
    process_time = (t_end - t_start) * 1000

    logger.info(f"Analysis finished. Score: {score:.2f}, Time: {process_time:.1f}ms")

    return QualityResult(
        ok_for_model=is_ok,
        quality_score=round(score, 2),
        message=status_msg,
        latency_ms=round(process_time, 2),
        flags=flags_cleaned,
        dataset_shape={"n_rows": d_info.row_count, "n_cols": d_info.col_count}
    )