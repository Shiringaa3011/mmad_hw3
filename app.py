import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent
TARGET = "target"

st.set_page_config(page_title="Wine Quality", layout="wide")

# cache_resource: модель - объект, который не нужно копировать на каждый вызов
# нужен один общий экземпляр на весь сервер. cache_data: таблицы и массивы - обычные данные
# Streamlit хранит копию и отдаёт независимую копию на каждый вызов
# Без декоратора joblib.load читал бы файл с диска при каждом движении слайдера
@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model.joblib")


@st.cache_data
def load_config() -> dict:
    return json.loads((APP_DIR / "config.json").read_text())


@st.cache_data
def load_test() -> pd.DataFrame:
    return pd.read_csv(APP_DIR / "test.csv")


@st.cache_data
def test_proba(_model, X: pd.DataFrame) -> np.ndarray:
    # вероятности не зависят от порога, считаем один раз
    return _model.predict_proba(X)[:, 1]


model = load_model()
config = load_config()
test_df = load_test()

FEATURES = [c for c in test_df.columns if c != TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET].to_numpy()

threshold = st.sidebar.slider("Порог классификации", 0.0, 1.0, float(config["threshold"]), 0.01,
                              key="threshold")
st.sidebar.caption("Вино считается хорошим, если вероятность не ниже порога.")

st.title("Рекомендации вин")
tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ["Одно вино", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"])

with tab_one:
    values, cols = {}, st.columns(3)
    for i, f in enumerate(FEATURES):
        with cols[i % 3]:
            if f == "is_red":
                values[f] = int(st.radio("Тип вина", ["белое", "красное"], horizontal=True) == "красное")
            else:
                lo, hi, med = (float(test_df[f].min()), float(test_df[f].max()), float(test_df[f].median()))
                values[f] = st.number_input(f, min_value=lo, max_value=hi, value=med,
                                            step=(hi - lo) / 100, format="%.4f")

    proba = float(model.predict_proba(pd.DataFrame([values])[FEATURES])[0, 1])
    c1, c2 = st.columns(2)
    c1.metric("Вероятность 'хорошести'", f"{proba:.1%}")
    c2.metric("Решение", "в подборку" if proba >= threshold else "не рекомендовать")


with tab_metrics:
    p = test_proba(model, X_test)
    pred = (p >= threshold).astype(int)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{metrics.precision_score(y_test, pred, zero_division=0):.3f}")
    m2.metric("Recall", f"{metrics.recall_score(y_test, pred):.3f}")
    m3.metric("F1", f"{metrics.f1_score(y_test, pred, zero_division=0):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics.ConfusionMatrixDisplay(cm, display_labels=["обычное", "хорошее"]).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Матрица ошибок")

    fpr, tpr, _ = metrics.roc_curve(y_test, p)
    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=.4)
    axes[1].scatter([fp / max(fp + tn, 1)], [tp / max(tp + fn, 1)], c="red", s=60, zorder=3, label="текущий порог")
    axes[1].set(title="ROC", xlabel="FPR", ylabel="TPR")
    axes[1].legend()

    pr_, rc_, _ = metrics.precision_recall_curve(y_test, p)
    axes[2].plot(rc_, pr_)
    axes[2].axhline(y_test.mean(), c="k", ls="--", alpha=.4, label="случайная модель")
    axes[2].scatter([tp / max(tp + fn, 1)], [tp / max(tp + fp, 1)], c="red", s=60, zorder=3, label="текущий порог")
    axes[2].set(title="PR", xlabel="Recall", ylabel="Precision")
    axes[2].legend()
    st.pyplot(fig)
    plt.close(fig)

    st.info("**Для сомелье.** Выше порог -> подборка короче, но чище: меньше посредственных вин "
            "(выше precision), зато часть хороших вин теряется (ниже recall). Ниже порог -> подборка шире "
            "и полнее, но в неё попадает больше разочаровывающих вин.")


with tab_batch:
    up = st.file_uploader("CSV с признаками вин (без столбца target)", type="csv")
    if up is not None:
        try:
            batch = pd.read_csv(up)
        except Exception as e:
            st.error(f"Не удалось прочитать файл: {e}")
        else:
            missing = [c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error(f"В файле не хватает столбцов: {', '.join(missing)}")
            elif batch[FEATURES].isna().any().any():
                st.error("В признаках есть пропуски - заполните их и загрузите файл снова.")
            else:
                # т.к. формально наличие столбца target не мешает нам предсказывать (мы его не берём),
                # то мы просто выведем предупреждение
                extra = [c for c in batch.columns if c not in FEATURES]
                if extra: 
                    st.info(f"Лишние столбцы не используются при предсказании: {', '.join(extra)}")
                out = batch.copy()
                out["proba_good"] = model.predict_proba(batch[FEATURES])[:, 1]
                out["decision"] = (out["proba_good"] >= threshold).astype(int)
                st.caption(f"Порог: {threshold:.2f} (из сайдбара). "
                           f"В подборку: {int(out['decision'].sum())} из {len(out)}")
                st.dataframe(out)
                st.download_button("Скачать результат", out.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv", "text/csv")


with tab_coef:
    coef = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(coef.index, coef.values, color=np.where(coef.values > 0, "tab:green", "tab:red"))
    ax.axvline(0, c="k", lw=.8)
    ax.set_xlabel("коэффициент (признаки масштабированы)")
    st.pyplot(fig)
    plt.close(fig)

    st.write("**За хорошее вино:** " + ", ".join(coef[coef > 0].sort_values(ascending=False).index[:3]))
    st.write("**Против:** " + ", ".join(coef[coef < 0].index[:3]))
