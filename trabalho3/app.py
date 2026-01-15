import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


MODEL_PATH = 'modelo_final_fase3.keras'
CLASSES = ['Apple', 'Banana', 'Kiwi', 'Orange', 'Peach'] 

st.set_page_config(page_title="Classificador Fase III", layout="centered")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Erro: O ficheiro '{MODEL_PATH}' não existe!")
        st.warning("Verifica se o ficheiro está na mesma pasta que este script.")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

model = load_model()

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # Adicionar import

def predict_image(image, model):

    img = image.resize((224, 224))
    
    img_array = np.array(img, dtype=np.float32)
    
    img_array = preprocess_input(img_array)
    
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    probabilities = predictions[0]
   
    class_idx = np.argmax(probabilities)
    confidence = np.max(probabilities)
    
    if class_idx < len(CLASSES):
        predicted_label = CLASSES[class_idx]
    else:
        predicted_label = f"Classe {class_idx}"

    return predicted_label, confidence, probabilities

st.title("🧠 Projeto IC - Fase III")
st.markdown("### Demonstração em Tempo Real")

option = st.sidebar.radio("Escolha a entrada:", ("📁 Carregar Imagem", "📷 Usar Câmara"))

input_image = None

if option == "📁 Carregar Imagem":
    uploaded_file = st.file_uploader("Escolha um ficheiro...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")

elif option == "📷 Usar Câmara":
    camera_input = st.camera_input("Tire uma foto")
    if camera_input is not None:
        input_image = Image.open(camera_input).convert("RGB")


if input_image is not None:
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(input_image, caption='Imagem de Entrada', use_container_width=True)
    
    with col2:
        if model is not None:
            if st.button('🔍 Classificar', type="primary"):
                with st.spinner('A processar...'):
                    label, conf, probs = predict_image(input_image, model)
                
                st.success(f"**Resultado:** {label}")
                st.info(f"**Confiança:** {conf*100:.2f}%")
                
                st.write("Probabilidades:")
                if len(CLASSES) == len(probs):
                    st.bar_chart(dict(zip(CLASSES, probs)))
                else:
                    st.bar_chart(probs)
        else:
            st.error("Modelo não carregado.")