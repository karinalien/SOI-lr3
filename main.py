import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from statsmodels.api import add_constant, OLS
import warnings
import sklearn.utils.validation
import factor_analyzer.utils
import os
import re

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
def format_p(p_val, is_russian=True):
 #Форматирует p-value
    if p_val < 0.0001:
        formatted = f"{p_val:.2e}".replace(".", ",")
        if is_russian:
            formatted = formatted.replace("e", "·10^")
        return formatted
    else:
        return f"{p_val:.4f}".replace(".", ",")

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

def normalize(s):
    return re.sub(r'[^a-zа-яё0-9]', '', str(s).lower())

places = ["Усы лисы", "Теремок", "Вкусно – и точка", "Burger King", "Столовая на БМ", "Rostic's", "Евразия"]
place_cols_map = {}
for p in places:
    norm_p = normalize(p)
    cols = [c for c in df_ratings.columns if norm_p in normalize(c)]
    if cols:
        place_cols_map[p] = cols
    else:
        print(f"Колонки для '{p}' не найдены. Проверьте написание в исходном файле.")
print(f"Успешно найдено заведений для анализа: {len(place_cols_map)}/{len(places)}")

print("\n1. ОПИСАТЕЛЬНАЯ СТАТИСТИКА И ГРАФИК")
desc = pd.DataFrame({'Среднее': df_ratings.mean(), 'Медиана': df_ratings.median(),
                     'Мода': df_ratings.mode().iloc[0], 'Дисперсия': df_ratings.var()}).round(3)
desc.to_csv("Результаты/Описательная_статистика.csv", encoding='utf-8-sig')

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

# 1.1. ОПИСАТЕЛЬНАЯ СТАТИСТИКА ХАРАКТЕРИСТИК ЭКСПЕРТОВ
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
print("\nОписательная статистика характеристик экспертов:")
print(desc_exp_df.to_string(index=False))

# ОПИСАТЕЛЬНАЯ СТАТИСТИКА ПО КРИТЕРИЯМ
criteria_map = [
    ("Размер порций (q1)", ["Порции", "порции", "большие"]),
    ("Свежесть продуктов (q2)", ["свежести", "Свежесть"]),
    ("Сытость до конца дня (q3)", ["сытости", "сытость", "наедаюсь"]),
    ("Чистота в зале (q4)", ["чистота", "Чистота"]),
    ("Ожидание ≤ 15 мин (q5)", ["Время", "ожидание", "очередь"]),
    ("Быстрое обслуживание (q6)", ["обслуживание", "персонал", "вежлив"]),
    ("Расстояние невелико (q7)", ["Путь", "расстояние", "дорога"]),
    ("Цена = качество (q8)", ["Цена", "качество", "соответствует"]),
    ("Порции = цена (q9)", ["Порции", "цена", "соотнош"]),
    ("Выгодные цены (q10)", ["выгодные", "Выгодные", "доступные"]),
    ("Акции / программы (q11)", ["акции", "программы", "бонусы", "скидки"])
]

crit_desc_rows = []
for crit_name, keywords in criteria_map:
    matched_cols = [c for c in df_ratings.columns if any(kw.lower() in c.lower() for kw in keywords)]
    if not matched_cols: continue
    values = pd.concat([df_ratings[c].dropna() for c in matched_cols]).values
    if len(values) == 0: continue
    mode_val = pd.Series(values).mode()
    crit_desc_rows.append({
        "Критерий": crit_name,
        "Среднее": round(float(np.mean(values)), 2),
        "Медиана": round(float(np.median(values)), 2),
        "Мода": int(mode_val.iloc[0]) if not mode_val.empty else "",
        "Дисперсия": round(float(np.var(values, ddof=1)), 3)
    })

crit_desc_df = pd.DataFrame(crit_desc_rows)
crit_desc_df.to_csv("Результаты/Описательная_статистика_критериев.csv", index=False, encoding='utf-8-sig')
print("\nОписательная статистика по критериям (объединено по всем заведениям):")
print(crit_desc_df.to_string(index=False))

print("\n1.2. ПРОВЕРКА ПРИМЕНИМОСТИ ФА К ХАРАКТЕРИСТИКАМ ЭКСПЕРТОВ")
try:
    kmo_exp, kmo_m_exp = calculate_kmo(exp_num_data.dropna())
    _, bart_p_exp = calculate_bartlett_sphericity(exp_num_data.dropna())
    print(f"KMO = {kmo_m_exp:.3f}, Bartlett p = {bart_p_exp:.4f}")
    pd.DataFrame({
        "Показатель": ["KMO", "Bartlett p-value", "Статус"],
        "Значение": [round(kmo_m_exp, 3), round(bart_p_exp, 4), "Неприменим"]
    }).to_csv("Результаты/Применимость_ФА_экспертов.csv", index=False, encoding='utf-8-sig')
except Exception as e:
    print(f"Ошибка расчёта: {e}")

print("\n2. ОЦЕНКА СОГЛАСОВАННОСТИ ДО ОЧИСТКИ")
pr_init = pd.DataFrame({p: df_ratings[cols].mean(axis=1) for p, cols in place_cols_map.items()}).dropna()
m0, k0 = pr_init.shape
fr_stat0, fr_p0 = stats.friedmanchisquare(*[pr_init[c].values for c in pr_init.columns])
ranks0 = pr_init.rank(axis=1)
S0 = np.sum((ranks0.sum(axis=0) - ranks0.sum(axis=0).mean())**2)
W0 = (12 * S0) / (m0**2 * (k0**3 - k0))
print(f"Изначально: Friedman p={fr_p0:.4f}, W={W0:.3f}")

alpha = 0.05
decision = "H₀ отвергается" if fr_p0 < alpha else "H₀ не отвергается"
friedman_table = pd.DataFrame({
    "Показатель": [
        "Количество экспертов (n)",
        "Количество оцениваемых признаков (k)",
        "Статистика Фридмана (χ²)",
        "p-value",
        "Уровень значимости α",
        "Решение"
    ],
    "Значение": [
        m0,
        k0,
        f"{fr_stat0:.3f}".replace(".", ","),
        f"{fr_p0:.4f}".replace(".", ","),
        f"{alpha:.2f}".replace(".", ","),
        decision
    ]
})
friedman_table.to_csv("Результаты/Фридман_до_очистки.csv", index=False, encoding='utf-8-sig')
print("\nРезультаты критерия Фридмана (до очистки данных):")
print(friedman_table.to_string(index=False))

places_list = list(pr_init.columns)
n_comparisons = len(places_list) * (len(places_list) - 1) // 2
alpha_bonf = 0.05 / n_comparisons

print(f"\nPost-hoc: попарные сравнения Уилкоксона, поправка Бонферрони")
print(f"Критический порог: α* = {alpha_bonf:.4f}")
print(f"{'Заведение 1':<20} {'Заведение 2':<20} {'p-значение':>12} {'Значимо?':>10}")
print("-" * 64)

posthoc_results = []
for i in range(len(places_list)):
    for j in range(i+1, len(places_list)):
        p1, p2 = places_list[i], places_list[j]
        stat, p_val = wilcoxon(pr_init[p1], pr_init[p2], zero_method='pratt')
        is_sig = p_val < alpha_bonf
        posthoc_results.append({
            "Заведение 1": p1,
            "Заведение 2": p2,
            "p-value": p_val,
            "Значимо (α*)": "Да" if is_sig else "Нет"
        })
        print(f"{p1:<20} {p2:<20} {p_val:>12.4f} {'Да' if is_sig else 'Нет':>10}")

pd.DataFrame(posthoc_results).to_csv("Результаты/PostHoc_Фридман.csv",
                                     index=False, encoding='utf-8-sig')

print("\n2.1. ФАКТОРНЫЙ АНАЛИЗ ОБЪЕКТОВ (ДО ОЧИСТКИ)")
fa_data_init = pr_init
print(f"Матрица данных: {fa_data_init.shape[0]} экспертов × {fa_data_init.shape[1]} заведений")

try:
    kmo_init, kmo_m_init = calculate_kmo(fa_data_init)
    bart_stat_init, bart_p_init = calculate_bartlett_sphericity(fa_data_init)

    print(f"KMO = {kmo_m_init:.3f}, Бартлетт: χ² = {bart_stat_init:.3f}, p = {format_p(bart_p_init)}")

    with open("Результаты/Тесты_ФА_объекты_до_очистки.txt", "w", encoding="utf-8") as f:
        f.write(f"KMO: {kmo_m_init:.3f}\n")
        f.write(f"Bartlett chi2: {bart_stat_init:.3f}\n")
        f.write(f"Bartlett p: {format_p(bart_p_init)}\n")

    n_factors_init = 3
    fa_obj_init = FactorAnalyzer(n_factors=n_factors_init, rotation='varimax', method='principal')
    fa_obj_init.fit(fa_data_init)

    # Матрица нагрузок
    load_obj_init = pd.DataFrame(
        fa_obj_init.loadings_,
        index=fa_data_init.columns,
        columns=[f'F{i + 1}' for i in range(n_factors_init)]
    ).round(3)

    print("\nМатрица нагрузок (объекты, до очистки):")
    print(load_obj_init.to_string())
    load_obj_init.to_csv("Результаты/ФА_объекты_до_очистки.csv", encoding='utf-8-sig')

    # Объяснённая дисперсия
    variance_init = fa_obj_init.get_factor_variance()
    variance_init_dict = {
        "Показатель": ["SS Loadings", "Proportion Var", "Cumulative Var"]
    }
    for j in range(n_factors_init):
        variance_init_dict[f"Фактор {j + 1}"] = [
            round(float(variance_init[0][j]), 4),
            round(float(variance_init[1][j]), 4),
            round(float(variance_init[2][j]), 4)
        ]

    variance_init_df = pd.DataFrame(variance_init_dict)
    variance_init_df.to_csv("Результаты/Объяснённая_дисперсия_объекты_до_очистки.csv",
                            index=False, encoding='utf-8-sig')
    print("\nОбъяснённая дисперсия (объекты, до очистки):")
    print(variance_init_df.to_string(index=False))

except np.linalg.LinAlgError:
    print("⚠ Ошибка: матрица сингулярна, тесты и ФА для объектов (до очистки) не вычислены.")
except Exception as e:
    print(f"Ошибка при ФА объектов (до очистки): {e}")

print("\n3. ОЧИСТКА ВЫБОРКИ")
bad_logic = set()
for idx, row in df_ratings.iterrows():
    for p, cols in place_cols_map.items():
        q1 = [c for c in cols if "Порции" in c]
        q3 = [c for c in cols if "сытости" in c or "сытость" in c]
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

df_ratings_c.to_csv("Результаты/Полная_матрица_оценок_очищенная.csv", encoding='utf-8-sig')
print("Очищенная матрица оценок сохранена: Полная_матрица_оценок_очищенная.csv")
decision1 = "H₀ отвергается" if fr_p1 < alpha else "H₀ не отвергается"

friedman_table_clean = pd.DataFrame({
    "Показатель": [
        "Количество экспертов (n)",
        "Количество оцениваемых признаков (k)",
        "Статистика Фридмана (χ²)",
        "p-value",
        "Уровень значимости α",
        "Решение"
    ],
    "Значение": [
        m1,
        k1,
        f"{fr_stat1:.3f}".replace(".", ","),
        f"{fr_p1:.4f}".replace(".", ","),
        f"{alpha:.2f}".replace(".", ","),
        decision1
    ]
})
friedman_table_clean.to_csv("Результаты/Фридман_после_очистки.csv", index=False, encoding='utf-8-sig')
print("\n Результаты критерия Фридмана (после очистки):")
print(friedman_table_clean.to_string(index=False))
print("\n4.1. ФАКТОРНЫЙ АНАЛИЗ ОБЪЕКТОВ")

fa_data = pr_c
try:
    kmo_obj, kmo_m_obj = calculate_kmo(fa_data)
    bart_stat, bart_p = calculate_bartlett_sphericity(fa_data)

    print(f"KMO = {kmo_m_obj}, Бартлетт: χ² = {bart_stat}, p = {format_p(bart_p)}")

    n_factors = 3
    fa_obj = FactorAnalyzer(n_factors=n_factors, rotation='varimax', method='principal')
    fa_obj.fit(fa_data)

    # Матрица нагрузок
    load_obj = pd.DataFrame(fa_obj.loadings_, index=fa_data.columns, columns=[f'F{i + 1}' for i in range(n_factors)])
    print("\nМатрица нагрузок (объекты):")
    print(load_obj.round(3))
    load_obj.to_csv("Результаты/ФА_объекты.csv", encoding='utf-8-sig')

    # Объяснённая дисперсия
    variance = fa_obj.get_factor_variance()
    ss_loadings = variance[0]
    prop_var = variance[1]
    cum_var = variance[2]

    variance_dict = {
        "Показатель": [
            "SS Loadings (собств. знач.)",
            "Proportion Var (доля дисп.)",
            "Cumulative Var (накопл.)"
        ]
    }
    for j in range(n_factors):
        variance_dict[f"Фактор {j + 1}"] = [
            round(float(ss_loadings[j]), 4),
            round(float(prop_var[j]), 4),
            round(float(cum_var[j]), 4)
        ]

    variance_df = pd.DataFrame(variance_dict)
    variance_df.to_csv("Результаты/Объяснённая_дисперсия.csv",
                       index=False, encoding='utf-8-sig')
    print("\nОбъяснённая дисперсия факторного анализа объектов:")
    print(variance_df.to_string(index=False))

except np.linalg.LinAlgError:
    print("Ошибка: матрица сингулярна, тесты и ФА не вычислены.")

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