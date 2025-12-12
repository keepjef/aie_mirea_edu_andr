from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import pandas as pd
from pandas.api import types as ptypes

@dataclass
class FeatureStats:
    """Статистика по отдельной колонке (признаку)."""
    feat_name: str
    feat_type: str
    count_total: int
    count_null: int
    null_ratio: float
    n_unique: int
    sample_values: List[Any]
    is_numeric: bool
    # Числовые метрики
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    std_val: Optional[float] = None

    def as_dictionary(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DatasetMeta:
    """Общая мета-информация о датасете."""
    total_rows: int
    total_cols: int
    features: List[FeatureStats]

    def as_dictionary(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_cols": self.total_cols,
            "columns": [f.as_dictionary() for f in self.features],
        }

def get_dataset_overview(
    df: pd.DataFrame,
    samples_limit: int = 3,
) -> DatasetMeta:
    """
    Собирает статистику по датасету: размеры, типы, пропуски, примеры.
    """
    n_rows, n_cols = df.shape
    feats_list: List[FeatureStats] = []

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        
        cnt_valid = int(series.notna().sum())
        cnt_null = n_rows - cnt_valid
        ratio_null = float(cnt_null / n_rows) if n_rows > 0 else 0.0
        cnt_unique = int(series.nunique(dropna=True))

        # Берем несколько примеров для наглядности
        samples = (
            series.dropna().astype(str).unique()[:samples_limit].tolist()
            if cnt_valid > 0
            else []
        )

        is_num = bool(ptypes.is_numeric_dtype(series))
        
        # Считаем базовую статистику только для чисел
        mn, mx, avg, dev = None, None, None, None
        if is_num and cnt_valid > 0:
            mn = float(series.min())
            mx = float(series.max())
            avg = float(series.mean())
            dev = float(series.std())

        feats_list.append(
            FeatureStats(
                feat_name=col,
                feat_type=dtype_str,
                count_total=cnt_valid,
                count_null=cnt_null,
                null_ratio=ratio_null,
                n_unique=cnt_unique,
                sample_values=samples,
                is_numeric=is_num,
                min_val=mn,
                max_val=mx,
                mean_val=avg,
                std_val=dev,
            )
        )

    return DatasetMeta(total_rows=n_rows, total_cols=n_cols, features=feats_list)

def calculate_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает DataFrame с количеством и долей пропусков по каждой колонке.
    """
    if df.empty:
        return pd.DataFrame(columns=["null_count", "null_ratio"])
    
    counts = df.isna().sum()
    ratios = counts / len(df)
    
    result = pd.DataFrame({
        "null_count": counts,
        "null_ratio": ratios,
    }).sort_values("null_ratio", ascending=False)
    
    return result

def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Считает корреляцию Пирсона только для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)

def get_top_values(
    df: pd.DataFrame,
    max_cols_to_check: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для строковых/категориальных колонок находит top_k самых частых значений.
    """
    results: Dict[str, pd.DataFrame] = {}
    target_cols = []
    
    # Отбираем кандидатов (строки или категории)
    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            target_cols.append(name)

    # Обрабатываем только первые max_cols_to_check колонок
    for col in target_cols[:max_cols_to_check]:
        s = df[col]
        # value_counts уже сортирует по убыванию
        counts = s.value_counts(dropna=True).head(top_k)
        if counts.empty:
            continue
            
        shares = counts / counts.sum()
        df_stats = pd.DataFrame({
            "value": counts.index.astype(str),
            "count": counts.values,
            "share": shares.values,
        })
        results[col] = df_stats
        
    return results

def assess_data_quality(meta: DatasetMeta, missing_stats: pd.DataFrame) -> Dict[str, Any]:
    """
    Вычисляет эвристики качества данных (HW03 additions included).
    Возвращает словарь с флагами и итоговым скором.
    """
    indicators: Dict[str, Any] = {}
    
    # 1. Проверка размеров
    indicators["is_too_small"] = meta.total_rows < 100
    indicators["is_too_wide"] = meta.total_cols > 100
    
    # 2. Проверка пропусков
    peak_null_ratio = float(missing_stats["null_ratio"].max()) if not missing_stats.empty else 0.0
    indicators["critical_null_ratio"] = peak_null_ratio
    indicators["has_severe_missing"] = peak_null_ratio > 0.5

    # Данные для анализа колонок
    cols_df = pd.DataFrame([f.as_dictionary() for f in meta.features])

    # 3. Эвристика: Константные колонки (HW03)
    # Если уникальное значение всего 1 (и не всё NaN)
    if not cols_df.empty:
        indicators["has_constant_cols"] = bool(
            ((cols_df["n_unique"] == 1) & (cols_df["null_ratio"] < 1.0)).any()
        )
    else:
        indicators["has_constant_cols"] = False

    # 4. Эвристика: Высокая кардинальность категорий (HW03)
    # Много уникальных значений в нечисловых полях (например, сырой текст или ID)
    cardinality_limit = 50
    has_high_card = False
    for f in meta.features:
        if not f.is_numeric and f.n_unique > cardinality_limit:
            has_high_card = True
            break
    indicators["has_high_cardinality"] = has_high_card

    # 5. Эвристика: Подозрительные дубликаты ID (HW03)
    # Если есть колонка user_id/id, но она не уникальна
    suspicious_ids = False
    for f in meta.features:
        # Простая проверка по имени
        if f.feat_name.lower() in ['id', 'user_id', 'uid'] and f.n_unique < meta.total_rows:
            suspicious_ids = True
    indicators["has_id_duplicates"] = suspicious_ids

    # Расчет итогового балла (Quality Score)
    score = 1.0
    
    # Штраф за пропуски
    score -= peak_null_ratio 
    
    # Штрафы за структуру
    if indicators["is_too_small"]: score -= 0.2
    if indicators["is_too_wide"]: score -= 0.1
    
    # Штрафы за новые эвристики
    if indicators["has_constant_cols"]: score -= 0.1
    if indicators["has_high_cardinality"]: score -= 0.05
    if indicators["has_id_duplicates"]: score -= 0.15

    indicators["health_score"] = max(0.0, min(1.0, score))
    
    return indicators

def meta_to_dataframe(meta: DatasetMeta) -> pd.DataFrame:
    """
    Преобразует объект DatasetMeta в плоскую таблицу для печати/сохранения.
    """
    data = []
    for f in meta.features:
        data.append({
            "column": f.feat_name,
            "type": f.feat_type,
            "filled": f.count_total,
            "nulls": f.count_null,
            "null_pct": f.null_ratio,
            "unique": f.n_unique,
            "is_num": f.is_numeric,
            "min": f.min_val,
            "max": f.max_val,
            "mean": f.mean_val,
            "std": f.std_val,
        })
    return pd.DataFrame(data)