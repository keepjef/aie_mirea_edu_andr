# HW06 – Report

> Файл: `homeworks/HW06/report.md`  

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-04.csv`
- Размер: 25 000 x 61; test_size = 8250
- Целевая переменная: `target` 
    - Класс `0`: ~95%
    - Класс `1`: ~5%
- Признаки: Все признаки числовые

## 2. Protocol

- Разбиение: train/test = 0.67 / 0.33, random_state = 42
- Подбор: CV = 5, оптимизация по ROC_AUC
- Метрики: в датасете выражен ярко дисбаланс классов, поэтому метрики F1-score и ROC_AUC здесь будут играть ключевую роль. Accuracy не будет отражать качество (из-за дисбаланса)

## 3. Models

Сравнивались следующие модели:

**DummyClassifier**
- Стратегия: most_frequent
- Используется как baseline

**LogisticRegression**

- Baseline из S05
- Подбор C, penalty
- Solver с поддержкой class_weight='balanced'

**DecisionTreeClassifier**
- Подбор: max_depth, min_samples_leaf

**RandomForestClassifier**

- Подбор: n_estimators, max_depth, min_samples_leaf
- Использован class_weight='balanced'
- OOB-score для контроля качества

**AdaBoostClassifier**
- Подбор: n_estimators, learning_rate
- Используется как бустинговый ансамбль

## 4. Results

| Model               | Accuracy | Precision | Recall | F1   | ROC-AUC |
| ------------------- | -------- | --------- | ------ | ---- | ------- |
| Dummy (baseline)    | 0.91     | 0.04      | 0.04   | 0.04 | 0.49    |
| Logistic Regression | 0.96     | 0.92      | 0.27   | 0.41 | 0.83    |
| Decision Tree       | 0.96     | 0.71      | 0.46   | 0.56 | 0.79    |
| Random Forest       | 0.97     | 0.99      | 0.42   | 0.59 | 0.89    |
| AdaBoost            | 0.97     | 0.94      | 0.42   | 0.58 | 0.88    |

## 5. Analysis

- Устойчивость: что будет, если поменять `random_state`? Метрики будут оставаться стабильными, хоть и будут наблюдаться небольшие колебания.
- Ошибки: confusion matrix для лучшей модели содержит основную часть ошибок в False Negative, False Positive при этом содержит минимум. Модель не переобучена. 
- Интерпретация permutation importance (top-12): наибольший вклад дают несколько ключевых числовых признаков, нет одного доминирующего. Остальные вносят крайне малый вклад.

## 6. Conclusion

Accuracy не подходит как основная метрика при дисбалансе. ROC-AUC — наиболее честный критерий сравнения моделей. Одиночные деревья склонны к переобучению. 

Ансамбли (RF, AdaBoost) значительно улучшают качество. 

Валидация только на train критична для честной оценки. Интерпретация через permutation importance остаётся возможной даже для ансамблей