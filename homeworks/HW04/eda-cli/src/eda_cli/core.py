from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import pandas as pd
from pandas.api import types as ptypes

@dataclass
class ColumnProfile:
    """Детальная информация по одному столбцу."""
    col_label: str
    dtype_name: str
    total_valid: int
    total_missing: int
    missing_pct: float
    unique_count: int
    example_list: List[Any]
    is_numerical_type: bool
    # Статистики
    val_min: Optional[float] = None
    val_max: Optional[float] = None
    val_mean: Optional[float] = None
    val_std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DatasetInfo:
    """Сводная информация о всем датасете."""
    row_count: int
    col_count: int
    columns_data: List[ColumnProfile]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.row_count, # Ключи оставлены совместимыми для API/CLI
            "n_cols": self.col_count,
            "columns": [c.to_dict() for c in self.columns_data],
        }

def extract_dataset_info(
    df: pd.DataFrame,
    preview_size: int = 3,
) -> DatasetInfo:
    """
    Сканирует DataFrame и собирает метаданные.
    """
    rows, cols = df.shape
    cols_profile_list: List[ColumnProfile] = []

    for c_name in df.columns:
        series = df[c_name]
        t_str = str(series.dtype)
        
        valid = int(series.notna().sum())
        missing = rows - valid
        miss_ratio = float(missing / rows) if rows > 0 else 0.0
        uniq = int(series.nunique(dropna=True))

        # Примеры значений
        previews = (
            series.dropna().astype(str).unique()[:preview_size].tolist()
            if valid > 0
            else []
        )

        is_num = bool(ptypes.is_numeric_dtype(series))
        
        # Расчет статистики
        stats = {}
        if is_num and valid > 0:
            stats['min'] = float(series.min())
            stats['max'] = float(series.max())
            stats['mean'] = float(series.mean())
            stats['std'] = float(series.std())

        cols_profile_list.append(
            ColumnProfile(
                col_label=c_name,
                dtype_name=t_str,
                total_valid=valid,
                total_missing=missing,
                missing_pct=miss_ratio,
                unique_count=uniq,
                example_list=previews,
                is_numerical_type=is_num,
                val_min=stats.get('min'),
                val_max=stats.get('max'),
                val_mean=stats.get('mean'),
                val_std=stats.get('std'),
            )
        )

    return DatasetInfo(row_count=rows, col_count=cols, columns_data=cols_profile_list)

def get_missing_data_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает таблицу с анализом пропусков.
    """
    if df.empty:
        return pd.DataFrame(columns=["cnt_missing", "pct_missing"])
    
    total = df.isna().sum()
    pct = total / len(df)
    
    return pd.DataFrame({
        "cnt_missing": total,
        "pct_missing": pct,
    }).sort_values("pct_missing", ascending=False)

def get_numeric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Матрица корреляций.
    """
    nums = df.select_dtypes(include="number")
    if nums.empty:
        return pd.DataFrame()
    return nums.corr(numeric_only=True)

def find_top_categories(
    df: pd.DataFrame,
    limit_columns: int = 5,
    k_elements: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Топ-K значений для категориальных признаков.
    """
    result_map: Dict[str, pd.DataFrame] = {}
    cat_cols = []
    
    for col in df.columns:
        s = df[col]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            cat_cols.append(col)

    for col in cat_cols[:limit_columns]:
        s = df[col]
        freqs = s.value_counts(dropna=True).head(k_elements)
        if freqs.empty:
            continue
            
        proportions = freqs / freqs.sum()
        df_res = pd.DataFrame({
            "item": freqs.index.astype(str),
            "freq": freqs.values,
            "prop": proportions.values,
        })
        result_map[col] = df_res
        
    return result_map

def compute_health_metrics(
        dataset_info: DatasetInfo, 
        missing_report: pd.DataFrame
) -> Dict[str, Any]:
    """
    Эвристический анализ качества (Health Check).
    """
    metrics: Dict[str, Any] = {}
    
    # 1. Размерность
    metrics["warning_few_rows"] = dataset_info.row_count < 100
    metrics["warning_many_cols"] = dataset_info.col_count > 100
    
    # 2. Пропуски
    max_miss_rate = float(missing_report["pct_missing"].max()) if not missing_report.empty else 0.0
    metrics["max_missing_rate"] = max_miss_rate
    metrics["critical_missing"] = max_miss_rate > 0.5

    # Данные столбцов для анализа
    cols_meta = pd.DataFrame([c.to_dict() for c in dataset_info.columns_data])

    # 3. Константные столбцы
    if not cols_meta.empty:
        metrics["has_const_cols"] = bool(
            ((cols_meta["unique_count"] == 1) & (cols_meta["missing_pct"] < 1.0)).any()
        )
    else:
        metrics["has_const_cols"] = False

    # 4. Кардинальность (слишком много уникальных категорий)
    card_thresh = 50
    high_card_detected = False
    for c in dataset_info.columns_data:
        if not c.is_numerical_type and c.unique_count > card_thresh:
            high_card_detected = True
            break
    metrics["has_high_cardinality"] = high_card_detected

    # 5. Дубликаты ID
    id_dups = False
    for c in dataset_info.columns_data:
        if c.col_label.lower() in ['id', 'user_id', 'uuid', 'pk'] and c.unique_count < dataset_info.row_count:
            id_dups = True
    metrics["has_id_duplicates"] = id_dups

    # Итоговая оценка (Score)
    final_score = 1.0
    final_score -= max_miss_rate 
    
    if metrics["warning_few_rows"]: final_score -= 0.2
    if metrics["warning_many_cols"]: final_score -= 0.1
    if metrics["has_const_cols"]: final_score -= 0.1
    if metrics["has_high_cardinality"]: final_score -= 0.05
    if metrics["has_id_duplicates"]: final_score -= 0.15

    metrics["quality_index"] = max(0.0, min(1.0, final_score))
    
    return metrics

def flatten_dataset_info(info: DatasetInfo) -> pd.DataFrame:
    """
    Превращает иерархическую структуру info в плоскую таблицу.
    """
    flat_data = []
    for c in info.columns_data:
        flat_data.append({
            "column": c.col_label,
            "dtype": c.dtype_name,
            "valid": c.total_valid,
            "missing": c.total_missing,
            "missing_pct": c.missing_pct,
            "unique": c.unique_count,
            "is_num": c.is_numerical_type,
            "min": c.val_min,
            "max": c.val_max,
            "mean": c.val_mean,
            "std": c.val_std,
        })
    return pd.DataFrame(flat_data)