from __future__ import annotations
from pathlib import Path
import pandas as pd
import typer
# Импорт обновленных сущностей
from .core import (
    DatasetInfo,
    compute_health_metrics,
    get_numeric_correlations,
    flatten_dataset_info,
    get_missing_data_report,
    extract_dataset_info,
    find_top_categories,
)
# Предполагается, что визуализация (viz.py) была адаптирована аналогично, 
# здесь используем условные имена функций визуализации
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="EDA Tool (S04 Version)")

def _load_data_file(
    fpath: Path,
    sep_char: str = ",",
    enc_type: str = "utf-8",
) -> pd.DataFrame:
    if not fpath.exists():
        raise typer.BadParameter(f"File path invalid: '{fpath}'")
    try:
        return pd.read_csv(fpath, sep=sep_char, encoding=enc_type)
    except Exception as e:
        raise typer.BadParameter(f"CSV read failed: {e}") from e

@app.command()
def overview(
    path: str = typer.Argument(..., help="Path to CSV dataset."),
    sep: str = typer.Option(",", help="Delimiter char."),
    encoding: str = typer.Option("utf-8", help="Encoding format."),
) -> None:
    """
    Print basic dataset info to stdout.
    """
    df = _load_data_file(Path(path), sep_char=sep, enc_type=encoding)
    
    info: DatasetInfo = extract_dataset_info(df)
    flat_df = flatten_dataset_info(info)
    
    typer.echo(f"Rows count: {info.row_count}")
    typer.echo(f"Cols count: {info.col_count}")
    typer.echo("\nAttributes preview:")
    typer.echo(flat_df.to_string(index=False))

@app.command()
def report(
    path: str = typer.Argument(..., help="Input CSV path."),
    out_dir: str = typer.Option("reports", help="Directory for results."),
    sep: str = typer.Option(",", help="CSV Separator."),
    encoding: str = typer.Option("utf-8", help="Encoding."),
    max_hist_columns: int = typer.Option(6, help="Limit hists for numeric cols."),
    title: str = typer.Option("# EDA Report", help="Markdown title."),
    top_k_categories: int = typer.Option(5, help="Top-K categories to show."),
) -> None:
    """
    Generates a comprehensive EDA report with metrics and charts.
    """
    target_path = Path(out_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    df = _load_data_file(Path(path), sep_char=sep, enc_type=encoding)

    # 1. Сбор метрик
    info = extract_dataset_info(df)
    summary_df = flatten_dataset_info(info)
    missing_stat = get_missing_data_report(df)
    corr_matrix = get_numeric_correlations(df)
    cat_tops = find_top_categories(df, k_elements=top_k_categories)

    # 2. Оценка качества (переименованная функция)
    quality = compute_health_metrics(info, missing_stat)

    # 3. Экспорт таблиц
    summary_df.to_csv(target_path / "summary.csv", index=False)
    if not missing_stat.empty:
        missing_stat.to_csv(target_path / "missing.csv", index=True)
    if not corr_matrix.empty:
        corr_matrix.to_csv(target_path / "correlation.csv", index=True)
    
    # 4. Экспорт категорий
    save_top_categories_tables(cat_tops, target_path / "top_categories")

    # 5. Генерация Markdown
    md_out = target_path / "report.md"
    with md_out.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n\n")
        f.write(f"Source: `{Path(path).name}`\n\n")
        f.write(f"Shape: **{info.row_count}** x **{info.col_count}**\n\n")

        f.write("## Data Health Check\n\n")
        f.write(f"- **Quality Index**: {quality['quality_index']:.2f}\n")
        f.write(f"- Peak missing ratio: {quality['max_missing_rate']:.2%}\n")
        
        if quality['has_const_cols']:
            f.write("- ⚠️ Contains constant columns.\n")
        if quality['has_high_cardinality']:
            f.write("- ⚠️ High cardinality detected in categories.\n")
        if quality['has_id_duplicates']:
            f.write("- ⛔ Possible ID duplicates found.\n")
        if quality['warning_few_rows']:
            f.write("- ℹ️ Dataset is very small.\n")

        f.write("\n## Artifacts\n\n")
        f.write("- See `summary.csv`, `missing.csv`, `correlation.csv` for raw data.\n")
        f.write("- Charts saved as PNG in this folder.\n")

    # 6. Рисование графиков
    plot_histograms_per_column(df, target_path, max_columns=max_hist_columns)
    plot_missing_matrix(df, target_path / "missing_matrix.png")
    plot_correlation_heatmap(df, target_path / "correlation_heatmap.png")

    typer.echo(f"Report completed at: {target_path}")

if __name__ == "__main__":
    app()