from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "data" / "iris_model.joblib"

FLOWERS = {
    0: {
        "name": "Iris Setosa",
        "image": BASE_DIR / "images" / "iris_setosa.png",
    },
    1: {
        "name": "Iris Versicolour",
        "image": BASE_DIR / "images" / "iris_versicolor.png",
    },
    2: {
        "name": "Iris Virginica",
        "image": BASE_DIR / "images" / "iris_virginica.png",
    },
}

st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")
st.title("🌸 Iris Flower Predictor")
st.write("Enter the flower measurements in centimeters, then click **Predict**.")

if not MODEL_PATH.exists():
    st.error(
        "The trained model is missing. Add the file "
        "`data/iris_model.joblib` to your GitHub repository."
    )
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

left, right = st.columns(2)

with left:
    st.subheader("Sepal measurements")
    sepal_length = st.number_input(
        "Sepal length (cm)", min_value=0.0, max_value=15.0, value=5.1, step=0.1
    )
    sepal_width = st.number_input(
        "Sepal width (cm)", min_value=0.0, max_value=15.0, value=3.5, step=0.1
    )

with right:
    st.subheader("Petal measurements")
    petal_length = st.number_input(
        "Petal length (cm)", min_value=0.0, max_value=15.0, value=1.4, step=0.1
    )
    petal_width = st.number_input(
        "Petal width (cm)", min_value=0.0, max_value=15.0, value=0.2, step=0.1
    )

st.divider()
button_col, result_col = st.columns([1, 1])

with button_col:
    predict = st.button("Predict flower", type="primary", use_container_width=True)

if predict:
    values = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = int(model.predict(values)[0])
    flower = FLOWERS.get(prediction)

    with result_col:
        if flower is None:
            st.error(f"Unknown prediction class: {prediction}")
        else:
            st.success(f"Prediction: **{flower['name']}**")
            if flower["image"].exists():
                st.image(str(flower["image"]), caption=flower["name"], use_container_width=True)
            else:
                st.warning(
                    f"Image not found: `images/{flower['image'].name}`. "
                    "Add the image to your repository to display it."
                )
