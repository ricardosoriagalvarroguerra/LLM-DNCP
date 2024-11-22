import os
import streamlit as st
import easyocr
import pandas as pd
import re
from groq import Groq

# Configuración de la página
st.set_page_config(page_icon="📄", layout="wide", page_title="Chatbot con PDF y GroqCloud")

# Mostrar icono en la cabecera
def icon(emoji: str):
    """Muestra un emoji como icono al estilo Notion."""
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)


# Inicializar el cliente de GroqCloud con la clave de API directamente
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# Inicializar EasyOCR
reader = easyocr.Reader(['es'])

# Inicializar el historial del chat
if "messages" not in st.session_state:
    st.session_state.messages = []

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = None


def extract_bids_from_pdf(file):
    """Extrae nombres de oferentes y montos totales de ofertas desde un PDF escaneado."""
    results = reader.readtext(file.read(), detail=0)  # Realizar OCR

    # Variables para almacenar los datos extraídos
    offerors = []
    amounts = []

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

    # Extraer nombres de oferentes y montos totales
    bids_df = extract_bids_from_pdf(uploaded_file)

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
