import os
import streamlit as st
import easyocr
import pandas as pd
import numpy as np
import re
from groq import Groq
from PIL import Image
import fitz  # PyMuPDF
import io
import logging

# Configuración de la página
st.set_page_config(page_icon="📄", layout="wide", page_title="Chatbot con PDF y GroqCloud")

# Mostrar icono en la cabecera
def icon(emoji: str):
    """Muestra un emoji como icono al estilo Notion."""
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

icon("🤖")

st.subheader("Chatbot con PDF y GroqCloud")

# Inicializar el cliente de GroqCloud con la clave de API directamente
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# Cargar EasyOCR con caché para evitar múltiples descargas
@st.cache_resource
def initialize_easyocr():
    import os

    # Suprimir mensajes de información de EasyOCR
    logging.getLogger('easyocr').setLevel(logging.ERROR)

    # Especificar una ruta de caché persistente
    user_home = os.path.expanduser('~')
    easyocr_cache = os.path.join(user_home, '.EasyOCR')

    # Crear la carpeta si no existe
    if not os.path.exists(easyocr_cache):
        os.makedirs(easyocr_cache)

    # Establecer la variable de entorno para la ruta de caché
    os.environ['EASYOCR_MODULE_PATH'] = easyocr_cache

    # Inicializar el lector de EasyOCR con múltiples trabajadores
    return easyocr.Reader(
        ['es'],
        model_storage_directory=easyocr_cache,
        download_enabled=True,
        gpu=False,
        worker_count=4  # Ajusta este número según los núcleos de tu CPU
    )

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

def extract_data_from_images(images):
    """Extrae nombres de oferentes y montos totales desde imágenes."""
    # Lista para almacenar los datos extraídos como pares
    data = []
    current_offeror = None

    for image in images:
        # Convertir la imagen (Pillow) a un array de NumPy
        image_np = np.array(image)

        # Realizar OCR en cada imagen
        results = reader.readtext(image_np, detail=0)

        # Filtrar el texto extraído para encontrar nombres y montos
        for line in results:
            line = line.strip()
            # Buscar nombres de oferentes (mayúsculas)
            if re.match(r"^[A-ZÑ\s]+$", line):  # Ejemplo: "CALDETEC INGENIERÍA SRL"
                current_offeror = line

            # Buscar montos de oferta (números grandes con puntos)
            elif re.search(r"\d+\.\d+\.\d+", line):  # Ejemplo: "225.124.186.771"
                amount = line
                if current_offeror:
                    # Añadir el par al data
                    data.append({
                        'Nombre Oferente': current_offeror,
                        'Monto Total de la Oferta': amount
                    })
                    current_offeror = None  # Resetear el nombre del oferente para el siguiente

    # Crear una tabla con los datos extraídos
    return pd.DataFrame(data)

# Subir archivo PDF
uploaded_file = st.file_uploader("Sube un archivo PDF", type=["pdf"])

if uploaded_file:
    st.info("Procesando el archivo PDF...")

    # Convertir el PDF a imágenes
    images = extract_images_from_pdf(uploaded_file)

    # Extraer nombres de oferentes y montos totales desde las imágenes
    data_df = extract_data_from_images(images)

    # Mostrar los resultados
    if not data_df.empty:
        st.success("Datos extraídos exitosamente.")
        st.subheader("Tabla Estructurada de Resultados")
        st.table(data_df)
    else:
        st.warning("No se encontraron datos de oferentes y montos en las imágenes proporcionadas.")

# Selección del modelo de GroqCloud
models = {
    "llama3-8b-8192": {
        "name": "LLaMA3-8b-8192",
        "tokens": 8192,
        "developer": "Meta"
    },
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
            full_response = ""
            response_placeholder = st.empty()
            for response_chunk in chat_responses_generator:
                full_response += response_chunk
                response_placeholder.markdown(full_response)

    except Exception as e:
        st.error(f"Error al procesar: {e}", icon="🚨")

    # Guardar la respuesta completa en el historial
    if isinstance(full_response, str):
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append({"role": "assistant", "content": combined_response})
