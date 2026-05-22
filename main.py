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
print("-"*20)

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

# ИНТЕГРАЛЬНЫЕ ОЦЕНКИ ЗАВЕДЕНИЙ
places = [
    "Усы лисы",
    "Теремок",
    "Вкусно – и точка",
    "Burger King",
    "Столовая на БМ",
    "Rostic's",
    "Евразия"
]
places_averages = {}

for place in places:
    place_cols = [col for col in df_ratings.columns if f"[{place}]" in col]

    if place_cols:
        # Считаем среднее арифметическое по всем оценкам для этого заведения
        places_averages[place] = df_ratings[place_cols].mean().mean()
    else:
        place_cols_lazy = [col for col in df_ratings.columns if place.split()[0] in col]
        if place_cols_lazy:
            places_averages[place] = df_ratings[place_cols_lazy].mean().mean()
        else:
            places_averages[place] = np.nan

# датафрейм
table3 = pd.DataFrame.from_dict(places_averages, orient='index', columns=['Средняя оценка'])
table3 = table3.sort_values(by='Средняя оценка', ascending=False).round(2)
print(table3)

# 2. АГРЕГАЦИЯ И ПРОВЕРКА РАЗЛИЧИЙ МЕЖДУ ЗАВЕДЕНИЯМИ
print("\n2. АГРЕГАЦИЯ И ПРОВЕРКА РАЗЛИЧИЙ МЕЖДУ ЗАВЕДЕНИЯМИ")

place_ratings = pd.DataFrame(index=df_ratings.index)
for place in places:
    place_cols = [col for col in df_ratings.columns if f"[{place}]" in col]
    if place_cols:
        place_ratings[place] = df_ratings[place_cols].mean(axis=1)
    else:
        place_cols_lazy = [col for col in df_ratings.columns if place.split()[0].lower() in col.lower()]
        if place_cols_lazy:
            place_ratings[place] = df_ratings[place_cols_lazy].mean(axis=1)
        else:
            place_ratings[place] = np.nan

place_ratings_clean = place_ratings.dropna()
m, k = place_ratings_clean.shape

friedman_stat, friedman_p = stats.friedmanchisquare(
    *[place_ratings_clean[col].values for col in place_ratings_clean.columns]
)
print(f"\nFriedman test: chi2 = {friedman_stat:.3f}, p-value = {friedman_p:.4f}")
if friedman_p < 0.05:
    print("H0 rejected: Significant differences exist between venues.")
else:
    print("H0 not rejected: No significant differences between venues.")

ranks = place_ratings_clean.rank(axis=1, method='average')
R_sums = ranks.sum(axis=0)
S = np.sum((R_sums - R_sums.mean())**2)
W = (12 * S) / (m**2 * (k**3 - k))
print(f"Kendall W concordance coefficient: {W:.3f}")

if friedman_p < 0.05:
    from itertools import combinations
    alpha_corrected = 0.05 / (k * (k - 1) / 2)
    for p1, p2 in combinations(place_ratings_clean.columns, 2):
        _, p_val = stats.wilcoxon(place_ratings_clean[p1], place_ratings_clean[p2])
        sig = "*" if p_val < alpha_corrected else ""
        print(f"{p1} vs {p2}: p={p_val:.4f} {sig}")
    print(f"Bonferroni threshold: {alpha_corrected:.4f}")

# 3. КОМПЛЕКСНАЯ ОЧИСТКА ВЫБОРКИ
print("\n3. ИДЕНТИФИКАЦИЯ И УДАЛЕНИЕ СЛАБЫХ ЭКСПЕРТОВ")

# Логический контроль
bad_logic_experts = set()
for idx, row in df_ratings.iterrows():
    for place in places:
        q1_col = [c for c in df_ratings.columns if place in c and "Порции в этом заведении достаточно большие" in c]
        q3_col = [c for c in df_ratings.columns if place in c and "чувство сытости" in c]
        if q1_col and q3_col:
            val1 = row[q1_col[0]]
            val3 = row[q3_col[0]]
            if pd.notna(val1) and pd.notna(val3) and abs(val1 - val3) >= 3:
                bad_logic_experts.add(idx)

# Фильтрация по согласованности (Корреляция Спирмена)
group_median = df_ratings.median(axis=0)

def safe_spearman(row, median_vals):
    if row.var() == 0 or median_vals.var() == 0:
        return 0.0
    corr = stats.spearmanr(row, median_vals).correlation
    return 0.0 if np.isnan(corr) else corr

corr_with_group = df_ratings.apply(lambda row: safe_spearman(row, group_median), axis=1)
threshold = max(0.15, np.percentile(corr_with_group, 15))
bad_stat_experts = set(corr_with_group[corr_with_group < threshold].index)

# Объединение и удаление
all_bad_experts = bad_logic_experts.union(bad_stat_experts)
good_idx = [i for i in df_ratings.index if i not in all_bad_experts]

df_ratings_clean = df_ratings.loc[good_idx]
df_experts_clean = df_experts.loc[good_idx]

# Пересчет согласованности для вывода
n_clean = df_ratings_clean.shape[0]
k_clean = df_ratings_clean.shape[1]
stat_f_clean, p_val_f_clean = stats.friedmanchisquare(*[df_ratings_clean.iloc[i, :] for i in range(n_clean)])
W_clean = stat_f_clean / (n_clean * (k_clean - 1))

print(f"Удалено по логическому противоречию (q1 vs q3): {len(bad_logic_experts)}")
print(f"Удалено по низкой согласованности (Спирмен < {threshold:.2f}): {len(bad_stat_experts)}")
print(f"Всего удалено уникальных экспертов: {len(all_bad_experts)}")
print(f"Осталось валидных анкет (n): {n_clean}")
print(f"Коэффициент конкордации Кендалла (W) после очистки: {W_clean:.3f}")
print("-" * 50)

# 4. ФАКТОРНЫЙ АНАЛИЗ
print("\n4. ФАКТОРНЫЙ АНАЛИЗ")
records = []
q_names = [f"q{i}" for i in range(1, 12)]

for idx, row in df_ratings_clean.iterrows():
    for place in places:
        place_cols = [col for col in df_ratings_clean.columns if f"[{place}]" in col]
        if len(place_cols) == 11:
            record = {'expert_id': idx, 'place': place}
            for i, col in enumerate(place_cols):
                record[q_names[i]] = row[col]
            records.append(record)

df_long = pd.DataFrame(records)
df_fa_input = df_long[q_names].astype(float)

print(f"Размерность данных для FA: {df_fa_input.shape[0]} наблюдений на {df_fa_input.shape[1]} признаков.")

# 4.2 Тесты KMO и Бартлетта
kmo_all, kmo_model = calculate_kmo(df_fa_input)
print(f"Тест KMO: {kmo_model:.3f}")

bartlett_stat, bartlett_p = calculate_bartlett_sphericity(df_fa_input)
print(f"Тест Бартлетта: χ² = {bartlett_stat:.1f}, p-value = {bartlett_p:.5f}")

n_factors_target = 3
fa_obj = FactorAnalyzer(n_factors=n_factors_target, rotation='varimax', method='principal')
fa_obj.fit(df_fa_input)


loadings_df = pd.DataFrame(fa_obj.loadings_, index=q_names,
                           columns=['Качество (F1)', 'Время (F2)', 'Экономия (F3)'])
print("\nМатрица факторных нагрузок (11 вопросов x 3 фактора):")
print(loadings_df.round(3))
loadings_df.to_csv("Матрица_нагрузок_11_вопросов.csv", encoding='utf-8-sig')

# факторные оценки
factor_scores = fa_obj.transform(df_fa_input)
df_long[['Score_Quality', 'Score_Time', 'Score_Economy']] = factor_scores

# 5. РЕГРЕССИОННЫЙ АНАЛИЗ (ПО ТЕОРЕТИЧЕСКИМ БЛОКАМ)
print("\n5. РЕГРЕССИОННЫЙ АНАЛИЗ (ТЕОРЕТИЧЕСКИЕ ИНДЕКСЫ)")

# Определяем ключевые слова для поиска нужных столбцов по трем факторам
q_quality = ["Порции в этом заведении достаточно большие", "уверенность в свежести", "чувство сытости", "чистота и порядок"]
q_time = ["Время ожидания", "быстрое обслуживание", "Путь до заведения"]
q_economy = ["Цена блюд соответствуют", "Порции в заведении соответствуют", "выгодные цены", "различные акции"]

expert_indices = pd.DataFrame(index=df_ratings_clean.index)

def get_cols(keywords):
    return [c for c in df_ratings_clean.columns if any(kw in c for kw in keywords)]

# Считаем средний балл каждого эксперт
expert_indices['Фактор_Качество'] = df_ratings_clean[get_cols(q_quality)].mean(axis=1)
expert_indices['Фактор_Время'] = df_ratings_clean[get_cols(q_time)].mean(axis=1)
expert_indices['Фактор_Экономия'] = df_ratings_clean[get_cols(q_economy)].mean(axis=1)

y = pd.to_numeric(df_experts_clean["Насколько для вас важен состав еды по КБЖУ?"], errors='coerce')
reg_data = pd.concat([y, expert_indices], axis=1).dropna()

y_clean = reg_data["Насколько для вас важен состав еды по КБЖУ?"]
X_clean = add_constant(reg_data[['Фактор_Качество', 'Фактор_Время', 'Фактор_Экономия']])

model = OLS(y_clean, X_clean).fit()

with open("Итоговые_результаты_регрессии_KBZHU.txt", "w", encoding="utf-8") as f:
    f.write(model.summary().as_text())

print(model.summary())