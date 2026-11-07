# Wine Quality — демо на Streamlit

Приложение рекомендует "хорошие" вина (оценка >= 7) с помощью логистической регрессии.
Модель внутри не обучается: файлы `model.joblib`, `config.json` и `test.csv` готовы.

**Запуск** (Python 3.14.0):
``` powershell
    pip install -r requirements.txt
    streamlit run app.py
```

**Кэширование.** Модель грузится через `st.cache_resource` (один общий объект на сервер), таблицы
и вероятности - через `st.cache_data` (копии данных). Без декоратора `joblib.load` читал бы файл
с диска при каждом движении слайдера