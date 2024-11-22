import os
import streamlit as st
import fitz  # PyMuPDF
import easyocr
import pandas as pd
import numpy as np
import re
from groq import Groq
from PIL import Image
import io

# Configuración de la página
st.set_page_config(page_icon="📄", layout="wide", page_title="Chatbot con PDF y GroqCloud")


# Inicializar el cliente de GroqCloud con la clave de API directamente
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# Cargar EasyOCR con caché para evitar múltiples descargas
@st.cache_resource
def initialize_easyocr():
    return easyocr.Reader(['es'])

reader = initialize_easyocr()

# Inicializar el historial del chat
if "messages" not in st.session_state:
    st.session_state.messages = []

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = None


def extract_images_from_pdf(file):
    """Convierte las páginas de un PDF en imágenes usando PyMuPDF."""
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    return images


def extract_bids_from_images(images):
    """Extrae nombres de oferentes y montos totales de ofertas desde imágenes."""
    # Variables para almacenar los datos extraídos
    offerors = []
    amounts = []

    for image in images:
        # Convertir la imagen (Pillow) a un array de NumPy
        image_np = np.array(image)

        # Realizar OCR en cada imagen
        results = reader.readtext(image_np, detail=0)

        # Filtrar el texto extraído para encontrar nombres y montos
        for line in results:
            # Busca posibles nombres de oferentes (generalmente en mayúsculas)
            if re.match(r"^[A-ZÑ\s]+\b", line):  # Ejemplo: "CALDETEC INGENIERÍA SRL"
                offerors.append(line.strip())

            # Busca montos de oferta (números grandes con puntos)
            elif re.search(r"\d+\.\d+\.\d+", line):  # Ejemplo: "98.641.138.385"
                amounts.append(line.strip())

    # Crear una tabla con los datos extraídos
    data = {'Nombre Oferente': offerors, 'Monto Total de la Oferta': amounts}
    df = pd.DataFrame(data)

    return df


# Subir archivo PDF
uploaded_file = st.file_uploader("Sube un archivo PDF", type=["pdf"])

if uploaded_file:
    st.info("Procesando el archivo PDF...")

    # Convertir el PDF a imágenes
    images = extract_images_from_pdf(uploaded_file)

    # Extraer nombres de oferentes y montos totales desde las imágenes
    bids_df = extract_bids_from_images(images)

    # Mostrar los resultados
    st.success("Datos extraídos exitosamente.")
    st.subheader("Tabla Estructurada de Resultados")
    st.table(bids_df)

# Selección del modelo de GroqCloud
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


def generate_chat_responses(chat_completion):
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
