from __future__ import annotations
from pathlib import Path
import pandas as pd
import typer
# Импортируем наши переименованные функции из core
from .core import (
    DatasetMeta,
    assess_data_quality,
    calculate_correlations,
    meta_to_dataframe,
    calculate_missing_stats,
    get_dataset_overview,
    get_top_values,
)
# Предполагаем, что модуль viz тоже был отрефакторен или адаптирован. 
# Используем здесь "логические" имена функций визуализации.
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="CLI утилита для EDA (HW03 version)")

def _read_csv(
    file_path: Path,
    separator: str = ",",
    encoding_type: str = "utf-8",
) -> pd.DataFrame:
    if not file_path.exists():
        raise typer.BadParameter(f"Файл не найден: '{file_path}'")
    try:
        return pd.read_csv(file_path, sep=separator, encoding=encoding_type)
    except Exception as e:
        raise typer.BadParameter(f"Ошибка чтения CSV: {e}") from e

@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к файлу данных."),
    sep: str = typer.Option(",", help="Разделитель столбцов."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Вывести базовую информацию о датасете в консоль.
    """
    df = _read_csv(Path(path), separator=sep, encoding_type=encoding)
    
    meta: DatasetMeta = get_dataset_overview(df)
    summary_table = meta_to_dataframe(meta)
    
    typer.echo(f"Всего строк: {meta.total_rows}")
    typer.echo(f"Всего колонок: {meta.total_cols}")
    typer.echo("\nСводка по признакам:")
    typer.echo(summary_table.to_string(index=False))

@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к входному CSV."),
    out_dir: str = typer.Option("reports", help="Папка для сохранения отчета."),
    sep: str = typer.Option(",", help="Разделитель."),
    encoding: str = typer.Option("utf-8", help="Кодировка."),
    # Новые параметры HW03
    max_hist_columns: int = typer.Option(6, help="Лимит числовых колонок для гистограмм."),
    title: str = typer.Option("# EDA Report", help="Заголовок внутри Markdown отчета."),
    top_k_categories: int = typer.Option(5, help="Сколько топ-значений выводить для категорий."),
    min_missing_share: float = typer.Option(0.5, help="Порог доли пропусков для предупреждения."),
) -> None:
    """
    Сгенерировать полный отчет (Markdown + CSV + PNG).
    """
    report_path = Path(out_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    df = _read_csv(Path(path), separator=sep, encoding_type=encoding)

    # 1. Расчет статистик
    meta = get_dataset_overview(df)
    summary_tbl = meta_to_dataframe(meta)
    missing_tbl = calculate_missing_stats(df)
    corr_tbl = calculate_correlations(df)
    # Используем параметр top_k_categories
    top_vals = get_top_values(df, top_k=top_k_categories)

    # 2. Оценка качества (эвристики)
    quality_metrics = assess_data_quality(meta, missing_tbl)

    # 3. Сохранение таблиц
    summary_tbl.to_csv(report_path / "summary.csv", index=False)
    if not missing_tbl.empty:
        missing_tbl.to_csv(report_path / "missing.csv", index=True)
    if not corr_tbl.empty:
        corr_tbl.to_csv(report_path / "correlation.csv", index=True)
    
    save_top_categories_tables(top_vals, report_path / "top_categories")

    # 4. Генерация Markdown
    md_file = report_path / "report.md"
    with md_file.open("w", encoding="utf-8") as f:
        # Используем параметр title
        f.write(f"{title}\n\n")
        f.write(f"Файл: `{Path(path).name}`\n\n")
        f.write(f"Размер: **{meta.total_rows}** строк, **{meta.total_cols}** столбцов.\n\n")

        f.write("## Оценка качества данных\n\n")
        f.write(f"- Итоговый скор (Health Score): **{quality_metrics['health_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков: **{quality_metrics['critical_null_ratio']:.2%}**\n")
        
        # Вывод флагов из новых эвристик
        if quality_metrics['has_constant_cols']:
            f.write("- ⚠️ **Найдены константные колонки** (одно значение на весь столбец).\n")
        if quality_metrics['has_high_cardinality']:
            f.write("- ⚠️ **Обнаружена высокая кардинальность** в категориальных признаках.\n")
        if quality_metrics['has_id_duplicates']:
            f.write("- ⛔ **Дубликаты в ID**: Поле, похожее на идентификатор, не уникально.\n")
        
        if quality_metrics['is_too_small']:
            f.write("- ℹ️ Датасет слишком маленький (<100 строк).\n")

        f.write("\n## Детализация\n\n")
        f.write("1. **Сводка**: см. `summary.csv`\n")
        f.write("2. **Пропуски**: см. `missing.csv` и график `missing_matrix.png`\n")
        
        if not corr_tbl.empty:
            f.write("3. **Корреляции**: см. `correlation.csv` и `correlation_heatmap.png`\n")
        else:
            f.write("3. **Корреляции**: числовых колонок нет или их недостаточно.\n")
            
        if top_vals:
            f.write(f"4. **Категории**: топ-{top_k_categories} значений сохранены в папке `top_categories/`\n")
        
        f.write(f"5. **Распределения**: гистограммы (до {max_hist_columns} шт.) сохранены как `hist_*.png`\n")

    # 5. Генерация графиков (передаем параметры)
    # Используем max_hist_columns
    plot_histograms_per_column(df, report_path, max_columns=max_hist_columns)
    plot_missing_matrix(df, report_path / "missing_matrix.png")
    plot_correlation_heatmap(df, report_path / "correlation_heatmap.png")

    typer.echo(f"Готово! Отчет сохранен в: {report_path}")

if __name__ == "__main__":
    app()