# Вебинар 3: Деревья решений и ансамблевые методы — Random Forest и Boosting в финтех задачах

## Обзор темы

Ансамблевые методы — наиболее применяемые алгоритмы для табличных данных в финтехе. Random Forest и Gradient Boosting (XGBoost, LightGBM, CatBoost) показывают лучшие результаты на задачах кредитного скоринга, fraud detection и риск-менеджмента.

## 1. Деревья решений — основа

### Как работает дерево решений

```
Корень: credit_utilization <= 0.5?
── Да: income >= 50000?
│   ├── Да: credit_score >= 650?
│   │   ├── Да: ОДОБРИТЬ (вероятность дефолта: 5%)
│   │   └── Нет: credit_history >= 3 года?
│   │       ├── Да: ОДОБРИТЬ (вероятность дефолта: 15%)
│   │       └── Нет: ОТКАЗАТЬ (вероятность дефолта: 35%)
│   └── Нет: ОТКАЗАТЬ (вероятность дефолта: 40%)
└── Нет: num_delinquencies == 0?
    ├── Да: ОДОБРИТЬ с условием (вероятность дефолта: 25%)
    └── Нет: ОТКАЗАТЬ (вероятность дефолта: 60%)
```

### Проблема переобучения одного дерева

```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb

# === Данные ===
np.random.seed(42)
n = 10000

data = {
    'income': np.random.lognormal(10, 0.8, n),
    'loan_amount': np.random.lognormal(9, 1, n),
    'credit_score': np.random.randint(300, 850, n),
    'employment_years': np.random.randint(0, 30, n),
    'num_open_accounts': np.random.randint(0, 15, n),
    'delinquencies_2y': np.random.poisson(0.3, n),
    'credit_utilization': np.random.uniform(0, 1, n),
    'num_inquiries_6m': np.random.poisson(1, n),
    'debt_to_income': np.random.uniform(0, 0.8, n),
    'home_ownership': np.random.choice([0, 1], n, p=[0.35, 0.65]),
}

df = pd.DataFrame(data)

default_prob = (
    0.5
    - 0.0001 * df['income']
    + 0.00001 * df['loan_amount']
    - 0.001 * df['credit_score']
    + 0.15 * df['delinquencies_2y']
    + 0.4 * df['credit_utilization']
    + 0.3 * df['debt_to_income']
    - 0.1 * df['home_ownership']
    + np.random.normal(0, 0.12, n)
)
df['default'] = (default_prob > 0.35).astype(int)

from sklearn.model_selection import train_test_split

X = df.drop('default', axis=1)
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# === Сравнение: одно дерево vs ансамбли ===
print("=" * 60)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

# 1. Одно дерево (склонно к переобучению)
tree_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_deep = DecisionTreeClassifier(max_depth=15, random_state=42)

# 2. Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

# 3. Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)

# 4. XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='auc')

# 5. LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)

models = {
    'DecisionTree (depth=3)': tree_shallow,
    'DecisionTree (depth=15)': tree_deep,
    'RandomForest': rf,
    'GradientBoosting': gb,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    # Кросс-валидация
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()

    print(f"{name:30s} | Test AUC: {auc:.4f} | CV AUC: {cv_auc:.4f} | Gap: {auc - cv_auc:+.4f}")
```

## 2. Random Forest — бэггинг на практике

### Принцип работы
- Строит множество деревьев на бутстрап-выборках
- Каждое дерево видит случайное подмножество признаков
- Итоговый прогноз = голосование (классификация) или усреднение (регрессия)

### Применение: Оценка кредитного риска

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# === Random Forest для кредитного скоринга ===
rf_credit = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_credit.fit(X_train, y_train)

# === Важность признаков ===
# 1. Gini importance (встроенная)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'gini_importance': rf_credit.feature_importances_
}).sort_values('gini_importance', ascending=False)

print("Gini Feature Importance:")
print(feature_importance.to_string(index=False))

# 2. Permutation importance (надёжнее)
perm_importance = permutation_importance(rf_credit, X_test, y_test, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    'feature': X.columns,
    'perm_importance': perm_importance.importances_mean,
    'perm_std': perm_importance.importances_std
}).sort_values('perm_importance', ascending=False)

print("\nPermutation Importance:")
print(perm_df.to_string(index=False))

# === Вероятности и калибровка ===
y_proba = rf_credit.predict_proba(X_test)[:, 1]

# Проверка калибровки
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
print("\nКалибровка (prob_true vs prob_pred):")
for pt, pp in zip(prob_true, prob_pred):
    print(f"  Predicted: {pp:.2f} | Actual: {pt:.2f}")

# Если калибровка плохая — используем CalibratedClassifierCV
calibrated_rf = CalibratedClassifierCV(rf_credit, method='isotonic', cv=5)
calibrated_rf.fit(X_train, y_train)
y_proba_calibrated = calibrated_rf.predict_proba(X_test)[:, 1]

# === OOB оценка (out-of-bag) ===
rf_oob = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=42)
rf_oob.fit(X_train, y_train)
print(f"\nOOB Score: {rf_oob.oob_score_:.4f}")
print(f"Test AUC:  {roc_auc_score(y_test, rf_oob.predict_proba(X_test)[:, 1]):.4f}")
```

## 3. Gradient Boosting — бустинг на практике

### Принцип работы
- Последовательное построение деревьев
- Каждое новое дерево исправляет ошибки предыдущих
- Learning rate контролирует вклад каждого дерева

### XGBoost для Fraud Detection

```python
# === Fraud Detection с XGBoost ===
np.random.seed(42)
n = 50000

# Сильный дисбаланс: только 1.5% мошеннических транзакций
fraud_data = {
    'transaction_amount': np.random.lognormal(3, 1.5, n),
    'hour_of_day': np.random.randint(0, 24, n),
    'distance_from_home_km': np.random.exponential(5, n),
    'time_since_last_transaction_min': np.random.exponential(60, n),
    'foreign_transaction': np.random.choice([0, 1], n, p=[0.9, 0.1]),
    'online_transaction': np.random.choice([0, 1], n, p=[0.6, 0.4]),
    'num_transactions_1h': np.random.poisson(2, n),
    'avg_transaction_amount_30d': np.random.lognormal(3, 1, n),
}

fraud_df = pd.DataFrame(fraud_data)

# Вероятность мошенничества
fraud_prob = (
    0.02
    + 0.05 * (fraud_df['transaction_amount'] > fraud_df['avg_transaction_amount_30d'] * 3).astype(int)
    + 0.08 * (fraud_df['hour_of_day'].isin([0, 1, 2, 3, 4, 5])).astype(int)
    + 0.1 * (fraud_df['distance_from_home_km'] > 50).astype(int)
    + 0.15 * fraud_df['foreign_transaction']
    + 0.05 * fraud_df['online_transaction']
    + 0.03 * (fraud_df['num_transactions_1h'] > 5).astype(int)
    + np.random.normal(0, 0.05, n)
)
fraud_df['is_fraud'] = (fraud_prob > 0.5).astype(int)

print(f"Fraud rate: {fraud_df['is_fraud'].mean():.2%}")

X_fraud = fraud_df.drop('is_fraud', axis=1)
y_fraud = fraud_df['is_fraud']

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
)

# === XGBoost с обработкой дисбаланса ===
scale_pos_weight = (y_train_f == 0).sum() / (y_train_f == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.1f}")

xgb_fraud = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='aucpr'  # Precision-Recall AUC для дисбаланса
)

xgb_fraud.fit(
    X_train_f, y_train_f,
    eval_set=[(X_test_f, y_test_f)],
    early_stopping_rounds=20,
    verbose=False
)

# === Оценка ===
y_proba_fraud = xgb_fraud.predict_proba(X_test_f)[:, 1]
print(f"\nROC-AUC: {roc_auc_score(y_test_f, y_proba_fraud):.4f}")

from sklearn.metrics import average_precision_score
ap = average_precision_score(y_test_f, y_proba_fraud)
print(f"Average Precision (PR-AUC): {ap:.4f}")

# === Threshold tuning для бизнеса ===
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test_f, y_proba_fraud)

# Бизнес-требование: Precision >= 0.8 (минимум ложных срабатываний)
valid_mask = precision >= 0.8
if valid_mask.any():
    best_idx = np.where(valid_mask)[0][-1]  # Максимальный recall при Precision >= 0.8
    best_threshold = thresholds[best_idx]
    print(f"\nПорог для Precision >= 0.8: {best_threshold:.4f}")
    print(f"  Precision: {precision[best_idx]:.4f}")
    print(f"  Recall: {recall[best_idx]:.4f}")

# === SHAP values для интерпретации ===
try:
    import shap
    explainer = shap.TreeExplainer(xgb_fraud)
    shap_values = explainer.shap_values(X_test_f)

    # Топ-признаки для конкретного случая мошенничества
    fraud_case = X_test_f[y_test_f == 1].iloc[0]
    print(f"\nSHAP values для мошеннической транзакции:")
    shap_series = pd.Series(shap_values[0], index=X.columns)
    print(shap_series.abs().sort_values(ascending=False).head(5))
except ImportError:
    print("\nSHAP не установлен. Установите: pip install shap")
```

## 4. LightGBM и CatBoost — современные бустинги

### LightGBM — скорость на больших данных

```python
# === LightGBM для скоринга ===
lgb_credit = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1
)

lgb_credit.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
)

y_proba_lgb = lgb_credit.predict_proba(X_test)[:, 1]
print(f"LightGBM ROC-AUC: {roc_auc_score(y_test, y_proba_lgb):.4f}")

# === CatBoost — работа с категориальными признаками ===
import catboost as cb

# Добавим категориальные признаки
X_cat = X.copy()
X_cat['education'] = np.random.choice(['high_school', 'bachelor', 'master', 'phd'], n)
X_cat['employment_type'] = np.random.choice(['employed', 'self_employed', 'unemployed', 'retired'], n)

X_train_cat, X_test_cat, _, _ = train_test_split(X_cat, y, test_size=0.25, random_state=42)

cat_features = ['education', 'employment_type']

cat_model = cb.CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=0,
    auto_class_weights='Balanced'
)

cat_model.fit(
    X_train_cat, y_train,
    eval_set=(X_test_cat, y_test),
    cat_features=cat_features,
    early_stopping_rounds=30
)

y_proba_cat = cat_model.predict_proba(X_test_cat)[:, 1]
print(f"CatBoost ROC-AUC: {roc_auc_score(y_test, y_proba_cat):.4f}")

# Важность признаков CatBoost
cb_importance = pd.DataFrame({
    'feature': X_cat.columns,
    'importance': cat_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nCatBoost Feature Importance:")
print(cb_importance.to_string(index=False))
```

## 5. Ансамблирование моделей (Stacking)

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# === Stacking: объединяем лучшие модели ===
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)),
    ('lgb', lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)),
]

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

stacking_model.fit(X_train, y_train)
y_proba_stack = stacking_model.predict_proba(X_test)[:, 1]
print(f"\nStacking ROC-AUC: {roc_auc_score(y_test, y_proba_stack):.4f}")
```

## Сравнительная таблица методов

| Метод | Скорость | Качество | Интерпретируемость | Дисбаланс | Большие данные |
|-------|----------|----------|-------------------|-----------|----------------|
| Decision Tree | Быстро | Среднее | Высокая | Плохо | Отлично |
| Random Forest | Средне | Хорошее | Средняя | Средне | Хорошо |
| XGBoost | Средне | Отличное | Средняя | Хорошо | Хорошо |
| LightGBM | Быстро | Отличное | Средняя | Хорошо | Отлично |
| CatBoost | Средне | Отличное | Средняя | Отлично | Хорошо |

## Задание для практики

1. Загрузите датасет Credit Card Fraud Detection с Kaggle
2. Постройте XGBoost модель с PR-AUC > 0.7
3. Настройте порог классификации для баланса Precision/Recall
4. Постройте Stacking ансамбль из 3 моделей и сравните с лучшей одиночной моделью
5. Визуализируйте SHAP values для топ-10 признаков
