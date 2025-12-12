# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

Параметры:

- `---out-dir` – Каталог для отчёта (по умолчанию `reports`);
- `--sep` – Разделитель в CSV (по умолчанию `,`);
- `--encoding` – Кодировка файла (по умолчанию `utf-8`);
- `--max-hist-columns` – Максимум числовых колонок для гистограмм (по умолчанию `6`)
- `--title` – Заголовок отсчета в начале report.md (по умолчанию `# EDA-отчёт`);
- `--top-k-categories` – Количество top-значений в выводе категориальных признаков (по умолчанию `5`).

Пример использования параметров:

```bash
uv run eda-cli report data/example.csv --out-dir reports --title TestTitle --top-k-categories 3
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```
