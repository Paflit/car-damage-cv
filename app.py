# app.py
# Streamlit-приложение: загрузка фото -> "есть damage или нет" + визуализация области + confidence
#
# Запуск:
#   pip install streamlit ultralytics opencv-python pillow numpy
#   streamlit run app.py
#
# ВАЖНО: поменяй MODEL_PATH на путь к твоему best.pt

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2


# -------------------- НАСТРОЙКИ --------------------
MODEL_PATH = "runs/segment/train10/weights/best.pt"  # <-- поменяй под себя
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25


# -------------------- ЗАГРУЗКА МОДЕЛИ --------------------
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def get_confidences(result):
    """
    Возвращает список confidence для найденных объектов.
    Для YOLOv8: result.boxes.conf -> tensor shape (N,)
    """
    if result.boxes is None or result.boxes.conf is None:
        return []
    return result.boxes.conf.cpu().numpy().tolist()


def main():
    st.set_page_config(page_title="Car Damage Detector", layout="centered")
    st.title("🚗 Car Damage Detector (YOLOv8)")

    st.sidebar.header("Настройки")
    conf_th = st.sidebar.slider("Порог confidence (conf)", 0.0, 1.0, DEFAULT_CONF, 0.01)
    imgsz = st.sidebar.selectbox("Размер изображения (imgsz)", [512, 640, 768, 1024], index=1)
    show_overlay = st.sidebar.checkbox("Показывать область (оверлей)", value=True)

    # Загружаем модель
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Не удалось загрузить модель по пути: {MODEL_PATH}\n\nОшибка: {e}")
        st.stop()

    uploaded = st.file_uploader("Загрузи фото автомобиля (jpg/png)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Загрузи изображение, чтобы получить результат.")
        return

    # PIL -> numpy (RGB)
    img_pil = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(img_pil)

    # Ultralytics принимает numpy (RGB тоже ок), но plot() вернёт BGR
    st.image(img_pil, caption="Исходное изображение", use_container_width=True)

    # Предсказание
    results = model.predict(img_rgb, conf=conf_th, imgsz=imgsz, verbose=False)
    r = results[0]

    confs = get_confidences(r)
    n = len(confs)

    # Логика "есть дамаг или нет"
    if n == 0:
        st.success("✅ Повреждения НЕ обнаружены")
        st.write("Объекты: 0")
    else:
        best_conf = float(max(confs))
        st.warning("⚠️ Обнаружено повреждение (damage)")
        st.write(f"Объекты: **{n}**")
        st.write(f"Максимальная уверенность: **{best_conf:.2f}**")

        # Табличка по объектам
        st.subheader("Уверенность по каждому найденному объекту")
        for i, c in enumerate(confs, 1):
            st.write(f"{i}. confidence = **{float(c):.2f}**")

    # Визуализация области (маски/боксы)
    if show_overlay:
        plotted_bgr = r.plot()  # ndarray BGR
        plotted_rgb = bgr_to_rgb(plotted_bgr)
        st.image(plotted_rgb, caption="Результат (маски/области + confidence)", use_container_width=True)

    # Дополнительно: показать маски есть/нет (для сегментации)
    if r.masks is None:
        st.caption("Маски: нет (res.masks is None)")
    else:
        st.caption(f"Маски: {r.masks.data.shape[0]} шт.")


if __name__ == "__main__":
    main()
