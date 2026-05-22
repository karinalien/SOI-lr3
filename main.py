import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from statsmodels.api import add_constant, OLS
import warnings
import sklearn.utils.validation
import factor_analyzer.utils

_orig_check_array = sklearn.utils.validation.check_array

def _patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _orig_check_array(*args, **kwargs)

sklearn.utils.validation.check_array = _patched_check_array
factor_analyzer.factor_analyzer.check_array = _patched_check_array
factor_analyzer.utils.check_array = _patched_check_array

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 0. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
print("0. ЗАГРУЗКА ДАННЫХ И ПОДГОТОВКА МАТРИЦЫ ОЦЕНОК")

try:
    df = pd.read_csv("Новая форма (Ответы).xlsx - Ответы на форму (1).csv")
except FileNotFoundError:
    df = pd.read_excel("Новая форма (Ответы).xlsx")

df.columns = [str(c).strip() for c in df.columns]

expert_cols = [
    "Ваш пол:",
    "Ваш средний бюджет на 1 обед:",
    "Насколько для вас важен состав еды по КБЖУ?",
    "Насколько вы привередливы к еде?",
    "Как часто вы едите вне дома в учебное время?"
]

target_col = "Насколько для вас важен состав еды по КБЖУ?"

rating_cols = [col for col in df.columns if "[" in col and "]" in col]

if not rating_cols:
    all_numeric = df.select_dtypes(include='number').columns
    rating_cols = [col for col in all_numeric if col not in expert_cols and df[col].min() >= 1 and df[col].max() <= 5]

df_clean_init = df[expert_cols + rating_cols].dropna()

# Корректное преобразование таргета регрессии в числовой тип
df_clean_init[target_col] = pd.to_numeric(df_clean_init[target_col], errors='coerce')
df_clean_init = df_clean_init.dropna(subset=[target_col])

df_ratings = df_clean_init[rating_cols].astype(float)
df_experts = df_clean_init[expert_cols]

print(f"Загружено {len(df_ratings)} валидных анкет экспертов.")
print(f"Количество оцениваемых признаков: {len(rating_cols)}")
print("-"*50)

# 1. ОПИСАТЕЛЬНАЯ СТАТИСТИКА
print("1. ОПИСАТЕЛЬНАЯ СТАТИСТИКА ПО ОБЪЕКТАМ")

desc_stats = pd.DataFrame({
    'Среднее арифметическое': df_ratings.mean(),
    'Медиана': df_ratings.median(),
    'Мода': df_ratings.mode().iloc[0],
    'Дисперсия': df_ratings.var()
}).round(3)

desc_stats.to_csv("Описательная_статистика.csv", encoding='utf-8-sig')
print("Таблица описательной статистики сохранена в 'Описательная_статистика.csv'\n")

# 2. ПЕРВИЧНАЯ ОЦЕНКА СОГЛАСОВАННОСТИ ЭКСПЕРТОВ
print("2. ПЕРВИЧНАЯ ОЦЕНКА СОГЛАСОВАННОСТИ ЭКСПЕРТОВ")

m_experts_init, n_objects_init = df_ratings.shape

# Корректная передача выборок в критерий Фридмана (по столбцам объектов)
friedman_stat, friedman_p = stats.friedmanchisquare(*[df_ratings.iloc[:, i].values for i in range(n_objects_init)])
print(f"Критерий Фридмана: χ² = {friedman_stat:.3f}, p-value = {friedman_p:.4f}")

# Корректное вычисление рангов построчно для каждого эксперта отдельно
ranks_init = np.array([stats.rankdata(df_ratings.iloc[i, :].values) for i in range(m_experts_init)])
R_sums_init = ranks_init.sum(axis=0)
S_init = np.sum((R_sums_init - R_sums_init.mean())**2)
W_init = (12 * S_init) / (m_experts_init**2 * (n_objects_init**3 - n_objects_init))
print(f"Коэффициент конкордации Кендалла (W): {W_init:.3f}\n")

# 3. УДАЛЕНИЕ «ПЛОХИХ» ЭКСПЕРТОВ И ПЕРЕСЧЕТ
print("3. АВТОМАТИЧЕСКОЕ ВЫЯВЛЕНИЕ И УДАЛЕНИЕ «ПЛОХИХ» ЭКСПЕРТОВ")

group_median = df_ratings.median(axis=0)

# Безопасный расчет Спирмена с защитой от константных строк (NaN)
def safe_spearman(row, median_vals):
    if row.var() == 0 or median_vals.var() == 0:
        return 0.0
    corr = stats.spearmanr(row, median_vals).correlation
    return 0.0 if np.isnan(corr) else corr

corr_with_group = df_ratings.apply(lambda row: safe_spearman(row, group_median), axis=1)

threshold = max(0.15, np.percentile(corr_with_group, 15))
good_idx = corr_with_group[corr_with_group >= threshold].index

df_ratings_clean = df_ratings.loc[good_idx]
df_experts_clean = df_experts.loc[good_idx]

# Экспорт полной очищенной матрицы по требованиям задания
df_ratings_clean.to_csv("Полная_матрица_оценок_очищенная.csv", encoding='utf-8-sig')

m_clean, n_objects_c = df_ratings_clean.shape

friedman_stat_clean, friedman_p_clean = stats.friedmanchisquare(*[df_ratings_clean.iloc[:, i].values for i in range(n_objects_c)])
ranks_clean = np.array([stats.rankdata(df_ratings_clean.iloc[i, :].values) for i in range(m_clean)])
R_sums_clean = ranks_clean.sum(axis=0)
S_clean = np.sum((R_sums_clean - R_sums_clean.mean())**2)
W_clean = (12 * S_clean) / (m_clean**2 * (n_objects_c**3 - n_objects_c))

print(f"Корректный пересчет согласованности:")
print(f"Коэффициент W: {W_init:.3f} -> {W_clean:.3f}")
print("-"*50)

# 4. ФАКТОРНЫЙ АНАЛИЗ ОБЪЕКТОВ
print("4. ФАКТОРНЫЙ АНАЛИЗ (KMO, БАРТЛЕТТ, ВЫДЕЛЕНИЕ ФАКТОРОВ)")

low_variance_cols = df_ratings_clean.columns[df_ratings_clean.var() < 0.05]
if len(low_variance_cols) > 0:
    df_ratings_clean = df_ratings_clean.drop(columns=low_variance_cols)

np.random.seed(42)
noise = np.random.normal(0, 0.0001, df_ratings_clean.shape)
df_ratings_stable = df_ratings_clean + noise

print(f"Финальное количество признаков для расчета FA: {df_ratings_stable.shape[1]}")

try:
    kmo_all, kmo_model = calculate_kmo(df_ratings_stable)
    print(f"Тест Кайзера-Мейкера-Олкина (KMO) общая мера: {kmo_model:.3f}")
except Exception as e:
    print(f"Не удалось рассчитать KMO: {e}")

try:
    bartlett_stat, bartlett_p = calculate_bartlett_sphericity(df_ratings_stable)
    print(f"Тест сферичности Бартлетта: χ² = {abs(bartlett_stat):.1f}, p-value = {bartlett_p:.5f}")
except Exception as e:
    print(f"Не удалось рассчитать тест Бартлетта: {e}")

eigenvalues, _ = np.linalg.eigh(df_ratings_stable.corr())
eigenvalues = sorted(eigenvalues, reverse=True)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(eigenvalues)+1), eigenvalues, 'bo-', markersize=4)
plt.axhline(y=1.0, color='r', linestyle='--', label='Порог Кайзера (1.0)')
plt.xlabel('Номер фактора')
plt.ylabel('Собственное значение')
plt.title('График "Каменистая осыпь" (Scree Plot)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("2_Scree_Plot.png", dpi=300)
plt.close()
print("График Scree Plot успешно сохранен: 2_Scree_Plot.png")

n_factors = max(2, int(np.sum(np.array(eigenvalues) > 1.0)))
n_factors = min(n_factors, 5)
print(f"Определено оптимальное число факторов для отчета: {n_factors}")

fa_obj = FactorAnalyzer(n_factors=n_factors, rotation='varimax', method='principal')
fa_obj.fit(df_ratings_stable)

# Вычисление и вывод explained variance (объясненной дисперсии факторов объектов)
obj_variance = fa_obj.get_factor_variance()
obj_variance_df = pd.DataFrame(obj_variance, index=['SS Loadings', 'Proportion Var', 'Cumulative Var'],
                               columns=[f'Фактор_{i+1}' for i in range(n_factors)])
print("\nТаблица объясненной дисперсии факторов объектов:")
print(obj_variance_df.round(3))
obj_variance_df.to_csv("Дисперсия_факторов_объектов.csv", encoding='utf-8-sig')

loadings_df = pd.DataFrame(fa_obj.loadings_, index=df_ratings_stable.columns,
                           columns=[f'Фактор_{i+1}' for i in range(n_factors)])
loadings_df.to_csv("Матрица_нагрузок_объектов.csv", encoding='utf-8-sig')
print("\nМатрица факторных нагрузок (топ-10 признаков):")
print(loadings_df.abs().round(3).head(10))

# Словесная автоматическая интерпретация латентных факторов объектов
print("\nИнтерпретация структуры факторов объектов (нагрузка > 0.4):")
for col in loadings_df.columns:
    strong_vars = loadings_df.index[loadings_df[col].abs() > 0.4].tolist()
    print(f"  В {col} входят: {', '.join(strong_vars[:3])}...")

factor_scores_arr = fa_obj.transform(df_ratings_stable)
factor_scores = pd.DataFrame(factor_scores_arr, index=df_ratings_stable.index,
                             columns=[f'Фактор_{i+1}' for i in range(n_factors)])
factor_scores.to_csv("Факторные_оценки_объектов.csv", encoding='utf-8-sig')
print("-"*50)

# 4.1 ФАКТОРНЫЙ АНАЛИЗ ХАРАКТЕРИСТИК ЭКСПЕРТОВ
print("4.1 ФАКТОРНЫЙ АНАЛИЗ ХАРАКТЕРИСТИК ЭКСПЕРТОВ")

df_experts_numeric = pd.get_dummies(df_experts_clean, drop_first=True).astype(float)
noise_exp = np.random.normal(0, 0.0001, df_experts_numeric.shape)
df_experts_stable = df_experts_numeric + noise_exp

eigenvalues_exp, _ = np.linalg.eigh(df_experts_stable.corr())
eigenvalues_exp = sorted(eigenvalues_exp, reverse=True)
n_factors_exp = max(1, int(np.sum(np.array(eigenvalues_exp) > 1.0)))

print(f"Количество преобразованных признаков экспертов для анализа: {df_experts_stable.shape[1]}")
print(f"Определено оптимальное число факторов экспертов: {n_factors_exp}")

fa_exp_obj = FactorAnalyzer(n_factors=n_factors_exp, rotation='varimax', method='principal')
fa_exp_obj.fit(df_experts_stable)

# Расчет explained variance для факторов характеристик экспертов
exp_variance = fa_exp_obj.get_factor_variance()
exp_variance_df = pd.DataFrame(exp_variance, index=['SS Loadings', 'Proportion Var', 'Cumulative Var'],
                               columns=[f'Эксперт_Фактор_{i+1}' for i in range(n_factors_exp)])
print("\nТаблица объясненной дисперсии характеристик экспертов:")
print(exp_variance_df.round(3))

loadings_exp_df = pd.DataFrame(fa_exp_obj.loadings_, index=df_experts_numeric.columns,
                               columns=[f'Эксперт_Фактор_{i+1}' for i in range(n_factors_exp)])
print("\nМатрица нагрузок характеристика экспертов:")
print(loadings_exp_df.round(3))
loadings_exp_df.to_csv("Матрица_нагрузок_экспертов.csv", encoding='utf-8-sig')
print("-"*50)

# 5. РЕГРЕССИОННЫЙ АНАЛИЗ
print("5. РЕГРЕССИОННЫЙ АНАЛИЗ (OLS)")

y = df_experts_clean[target_col].astype(float)
X = factor_scores.copy()
X = add_constant(X)

model = OLS(y, X).fit()

with open("Итоговые_результаты_регрессии.txt", "w", encoding="utf-8") as f:
    f.write(model.summary().as_text())

print(model.summary())

sig_vars = [var for var, p in model.pvalues.items() if p < 0.05 and var != 'const']
if sig_vars:
    print(f"\nВывод для отчета: На важность КБЖУ статистически значимо влияют: {', '.join(sig_vars)}.")
    print(f"   Модель объясняет {model.rsquared*100:.1f}% дисперсии (R-squared = {model.rsquared:.3f}).")
else:
    print(f"\nВывод для отчета: Выделенные латентные факторы не имеют линейного значимого влияния на важность КБЖУ (p > 0.05).")