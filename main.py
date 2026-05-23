import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from statsmodels.api import add_constant, OLS
import warnings
import sklearn.utils.validation
import factor_analyzer.utils
import os

_orig_check_array = sklearn.utils.validation.check_array
def _patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _orig_check_array(*args, **kwargs)
sklearn.utils.validation.check_array = _patched_check_array
factor_analyzer.factor_analyzer.check_array = _patched_check_array
factor_analyzer.utils.check_array = _patched_check_array

def safe_spearman(row, median_vals):
    if row.var() == 0 or median_vals.var() == 0: return 0.0
    corr = stats.spearmanr(row, median_vals).correlation
    return 0.0 if np.isnan(corr) else corr

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('Результаты', exist_ok=True)

print("0. ЗАГРУЗКА ДАННЫХ")
try:
    df = pd.read_csv("Новая форма (Ответы).xlsx - Ответы на форму (1).csv")
except FileNotFoundError:
    df = pd.read_excel("Новая форма (Ответы).xlsx")

if len(df) < 30:
    raise ValueError(f"Ошибка: респондентов {len(df)} < 30 (требование п. 1.1)")

df.columns = [str(c).strip() for c in df.columns]
expert_cols = ["Ваш пол:", "Ваш средний бюджет на 1 обед:", "Насколько для вас важен состав еды по КБЖУ?",
               "Насколько вы привередливы к еде?", "Как часто вы едите вне дома в учебное время?"]
target_col = "Насколько для вас важен состав еды по КБЖУ?"
rating_cols = [col for col in df.columns if "[" in col and "]" in col]

df_clean = df[expert_cols + rating_cols].dropna()
df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
df_clean = df_clean.dropna(subset=[target_col])

df_ratings = df_clean[rating_cols].astype(float)
df_experts = df_clean[expert_cols]
print(f"Загружено {len(df_ratings)} анкет. Признаков: {len(rating_cols)}")

print("\n1. ОПИСАТЕЛЬНАЯ СТАТИСТИКА И ГРАФИК")
desc = pd.DataFrame({'Среднее': df_ratings.mean(), 'Медиана': df_ratings.median(),
                     'Мода': df_ratings.mode().iloc[0], 'Дисперсия': df_ratings.var()}).round(3)
desc.to_csv("Результаты/Описательная_статистика.csv", encoding='utf-8-sig')

places = ["Усы лисы", "Теремок", "Вкусно – и точка", "Burger King", "Столовая на БМ", "Rostic's", "Евразия"]
place_cols_map = {}
for p in places:
    cols = [c for c in df_ratings.columns if f"[{p}]" in c]
    if cols: place_cols_map[p] = cols

places_avg = {p: df_ratings[cols].mean().mean() for p, cols in place_cols_map.items() if cols}
tbl = pd.DataFrame.from_dict(places_avg, orient='index', columns=['Средняя оценка']).sort_values(by='Средняя оценка', ascending=False)

plt.figure(figsize=(9, 5))
plt.bar(tbl.index, tbl['Средняя оценка'], color='skyblue', edgecolor='black')
plt.title('Средние оценки заведений экспертами')
plt.ylabel('Оценка (1-5)')
plt.xlabel('Заведение')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Результаты/График_оценок.png', dpi=300)
plt.show()
df_ratings.to_csv("Результаты/Матрица_оценок_полная.csv", encoding='utf-8-sig')
print("График и матрица сохранены.")


# --- 1.1. ОПИСАТЕЛЬНАЯ СТАТИСТИКА ХАРАКТЕРИСТИК ЭКСПЕРТОВ (ДО ОЧИСТКИ) ---
def parse_budget(val):
    v = str(val).lower()
    if 'до 300' in v: return 1.0
    if '300' in v and '500' in v: return 2.0
    if 'свыше' in v or 'более' in v or '500' in v: return 3.0
    return np.nan

budget_map = df_experts["Ваш средний бюджет на 1 обед:"].apply(parse_budget)
likert_cols = ["Насколько для вас важен состав еды по КБЖУ?",
               "Насколько вы привередливы к еде?",
               "Как часто вы едите вне дома в учебное время?"]
exp_num_data = pd.DataFrame({
    "Бюджет (1-3)": budget_map,
    "Важность КБЖУ (1-5)": pd.to_numeric(df_experts[likert_cols[0]], errors='coerce'),
    "Привередливость (1-5)": pd.to_numeric(df_experts[likert_cols[1]], errors='coerce'),
    "Частота питания (1-5)": pd.to_numeric(df_experts[likert_cols[2]], errors='coerce')
})

exp_desc_rows = []
for col in exp_num_data.columns:
    v = exp_num_data[col].dropna()
    if v.empty: continue
    mode_val = v.mode().iloc[0] if not v.mode().empty else np.nan
    exp_desc_rows.append({
        "Характеристика": col,
        "Среднее": round(v.mean(), 2),
        "Медиана": round(v.median(), 2),
        "Мода": round(mode_val, 2) if pd.notna(mode_val) else "",
        "Дисперсия": round(v.var(), 2)
    })

desc_exp_df = pd.DataFrame(exp_desc_rows)
desc_exp_df.to_csv("Результаты/Описательная_статистика_экспертов.csv", index=False, encoding='utf-8-sig')
print("\n Описательная статистика характеристик экспертов:")
print(desc_exp_df.to_string(index=False))

print("\n2. ОЦЕНКА СОГЛАСОВАННОСТИ ДО ОЧИСТКИ")
pr_init = pd.DataFrame({p: df_ratings[cols].mean(axis=1) for p, cols in place_cols_map.items()}).dropna()
m0, k0 = pr_init.shape
fr_stat0, fr_p0 = stats.friedmanchisquare(*[pr_init[c].values for c in pr_init.columns])
ranks0 = pr_init.rank(axis=1)
S0 = np.sum((ranks0.sum(axis=0) - ranks0.sum(axis=0).mean())**2)
W0 = (12 * S0) / (m0**2 * (k0**3 - k0))
print(f"Изначально: Friedman p={fr_p0:.4f}, W={W0:.3f}")

print("\n3. ОЧИСТКА ВЫБОРКИ")
bad_logic = set()
for idx, row in df_ratings.iterrows():
    for p in places:
        q1 = [c for c in df_ratings.columns if f"[{p}]" in c and "Порции в этом заведении достаточно большие" in c]
        q3 = [c for c in df_ratings.columns if f"[{p}]" in c and "чувство сытости" in c]
        if q1 and q3 and pd.notna(row[q1[0]]) and pd.notna(row[q3[0]]):
            if abs(row[q1[0]] - row[q3[0]]) >= 3:
                bad_logic.add(idx)

corr_vals = df_ratings.apply(lambda r: safe_spearman(r, df_ratings.median()), axis=1)
bad_corr = set(corr_vals[corr_vals < 0.15].index)
all_bad = bad_logic.union(bad_corr)
good_idx = [i for i in df_ratings.index if i not in all_bad]

df_ratings_c = df_ratings.loc[good_idx].reset_index(drop=True)
df_experts_c = df_experts.loc[good_idx].reset_index(drop=True)
print(f"Удалено экспертов: {len(all_bad)}. Осталось: {len(df_ratings_c)}")

print("\n4. ОЦЕНКА СОГЛАСОВАННОСТИ ПОСЛЕ ОЧИСТКИ")
pr_c = pd.DataFrame({p: df_ratings_c[cols].mean(axis=1) for p, cols in place_cols_map.items()}).dropna()
m1, k1 = pr_c.shape
fr_stat1, fr_p1 = stats.friedmanchisquare(*[pr_c[c].values for c in pr_c.columns])
ranks1 = pr_c.rank(axis=1)
S1 = np.sum((ranks1.sum(axis=0) - ranks1.sum(axis=0).mean())**2)
W1 = (12 * S1) / (m1**2 * (k1**3 - k1))
print(f"Friedman p={fr_p1:.4f}, W={W1:.3f}")
df_ratings_c.to_csv("Результаты/Матрица_оценок_очищенная.csv", encoding='utf-8-sig')

print("\n4.1. ФАКТОРНЫЙ АНАЛИЗ ОБЪЕКТОВ")
fa_data = pr_c
try:
    kmo_obj, kmo_m_obj = calculate_kmo(fa_data)
    bart_obj, bart_p_obj = calculate_bartlett_sphericity(fa_data)
except np.linalg.LinAlgError:
    kmo_m_obj, bart_p_obj = 0.0, 1.0
    print("Внимание: матрица корреляций сингулярна (N < p в исходных данных). Используется агрегированная форма.")

print(f"Пригодность данных (агрегировано): KMO={kmo_m_obj:.3f}, Bartlett p={bart_p_obj:.5f}")

if kmo_m_obj > 0.5 or bart_p_obj < 0.05:
    fa_obj = FactorAnalyzer(n_factors=3, rotation='varimax', method='principal')
    fa_obj.fit(fa_data)
    load_obj = pd.DataFrame(fa_obj.loadings_, index=fa_data.columns, columns=['F1', 'F2', 'F3'])
    print("Матрица нагрузок (объекты):")
    print(load_obj.round(3))
    load_obj.to_csv("Результаты/ФА_объекты.csv", encoding='utf-8-sig')
else:
    print("Условия KMO/Bartlett не выполнены для объектов. ФА пропущен.")

print("\n5. РЕГРЕССИОННЫЙ АНАЛИЗ")
def get_cols(keywords):
    return [c for c in df_ratings_c.columns if any(kw in c for kw in keywords)]

cols_q = get_cols(["сытости", "свежести", "Порции", "чистота"])
cols_t = get_cols(["Время", "обслуживание", "Путь"])
cols_e = get_cols(["Цена", "выгодные", "акции"])

print(f"Найдено колонок: Quality={len(cols_q)}, Time={len(cols_t)}, Economy={len(cols_e)}")

def safe_mean(cols, df):
    return df[cols].mean(axis=1) if cols else pd.Series(np.nan, index=df.index)

reg_df = pd.DataFrame({
    'Quality': safe_mean(cols_q, df_ratings_c),
    'Time': safe_mean(cols_t, df_ratings_c),
    'Economy': safe_mean(cols_e, df_ratings_c)
}).dropna()

y = pd.to_numeric(df_experts_c[target_col], errors='coerce').loc[reg_df.index]

if len(reg_df) > 5 and y.notna().sum() > 5:
    model = OLS(y, add_constant(reg_df)).fit()
    print(model.summary())
    with open("Результаты/Регрессия_KBZHU.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())
else:
    print("Недостаточно валидных данных для регрессии.")