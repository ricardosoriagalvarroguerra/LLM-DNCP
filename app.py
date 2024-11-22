import os
import streamlit as st
import easyocr
from pdf2image import convert_from_bytes
from typing import Generator
from groq import Groq
import pandas as pd

# Configuración de la página
st.set_page_config(page_icon="📄", layout="wide", page_title="Chatbot con PDF y GroqCloud")

st.subheader("Extracción de Datos Actas de Apertura")

# Inicializar el cliente de GroqCloud con la clave de API directamente
GROQ_API_KEY = "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"
client = Groq(
    api_key=GROQ_API_KEY,
)

# Inicializar EasyOCR
reader = easyocr.Reader(['es'])

# Inicializar el historial del chat
if "messages" not in st.session_state:
    st.session_state.messages = []

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = None

# Layout para subir PDFs y mostrar chat
uploaded_file = st.file_uploader("Sube un archivo PDF", type=["pdf"])

if uploaded_file:
    st.info("Procesando el archivo PDF...")

    # Convertir PDF a imágenes y extraer texto
    images = convert_from_bytes(uploaded_file.read())
    extracted_text = ""
    for image in images:
        result = reader.readtext(image)
        page_text = " ".join([text[1] for text in result])
        extracted_text += page_text + "\n"

    st.success("Texto extraído exitosamente.")
    st.session_state.extracted_text = extracted_text
    st.text_area("Texto extraído:", extracted_text, height=200)

# Selección del modelo
models = {
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
}

model_option = st.selectbox(
    "Elige un modelo de GroqCloud:",
    options=list(models.keys()),
    format_func=lambda x: models[x]["name"],
    index=0
)

max_tokens = st.slider(
    "Tokens máximos:",
    min_value=512,
    max_value=models[model_option]["tokens"],
    value=2048,
    step=512,
)

# Mostrar historial del chat
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Generador para manejar las respuestas de Groq API."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if prompt := st.chat_input("Escribe tu consulta..."):
    # Si se ingresó texto como prompt, agregar al historial del chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Preparar el texto para enviar a GroqCloud
    if st.session_state.extracted_text:
        context = f"Texto del PDF:\n{st.session_state.extracted_text}\n\nUsuario: {prompt}"
    else:
        context = prompt

    # Llamar a GroqCloud para generar respuesta
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True,
        )

        # Generar respuestas dinámicas
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)

    except Exception as e:
        st.error(f"Error al procesar: {e}", icon="🚨")

    # Guardar la respuesta completa en el historial
    if isinstance(full_response, str):
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append({"role": "assistant", "content": combined_response})

# Tabla estructurada
if st.session_state.extracted_text:
    st.subheader("Tabla Estructurada de Resultados")
    # Ejemplo simple para crear tabla
    # Ajustar según la lógica de tu prompt o extracción específica
    data = {
        "Campo 1": ["Valor A1", "Valor A2"],
        "Campo 2": ["Valor B1", "Valor B2"],
        "Campo 3": ["Valor C1", "Valor C2"],
    }
    df = pd.DataFrame(data)
    st.table(df)
