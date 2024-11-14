import streamlit as st
import pandas as pd
import easyocr
from PIL import Image
import requests
import io

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Extractor de Datos de Licitaciones",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título de la aplicación
st.title("Extractor de Datos de Licitaciones usando OCR en Imágenes")

# Instrucciones para el usuario
st.markdown("""
Esta aplicación te permite subir una imagen que contiene información sobre licitaciones. 
El texto de la imagen será extraído y procesado para extraer datos específicos mediante una interfaz de chat.
""")

# **Configuración de la API de Hugging Face**
# Reemplaza estos valores con tus propias credenciales de Hugging Face
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
HUGGINGFACE_API_TOKEN = "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"  # Reemplaza con tu token de Hugging Face

# Encabezados para la API de Hugging Face
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

# Inicializar el lector de EasyOCR
@st.cache_resource
def initialize_easyocr():
    return easyocr.Reader(['es'])  # 'es' para español

reader = initialize_easyocr()

# Función para hacer OCR con EasyOCR
def extract_text_from_image(image):
    # Convertir la imagen a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Convertir la imagen a un array de bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    # Realizar OCR
    result = reader.readtext(img_bytes, detail=0, paragraph=True)
    return '\n'.join(result)

# Función para realizar la solicitud a la API de Hugging Face
def query(payload):
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error en la solicitud a Hugging Face: {response.status_code}")
            st.write(response.text)
            return None
    except Exception as e:
        st.error(f"Ocurrió un error al realizar la solicitud a Hugging Face: {e}")
        return None

# Función para generar texto usando el modelo GPT-Neo
def generate_response(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": False
        }
    }
    
    response = query(payload)
    
    if response and isinstance(response, list) and "generated_text" in response[0]:
        return response[0]["generated_text"]
    else:
        st.error("Error en la respuesta del modelo GPT-Neo")
        return None

# Función para convertir el texto estructurado en un DataFrame
def text_to_dataframe(structured_text):
    data = []
    lines = structured_text.strip().split("\n")
    for line in lines:
        # Separar por comas, asumiendo que los campos están separados por comas
        columns = [col.strip() for col in line.split(",")]
        if len(columns) == 10:  # Asegurarse de que hay 10 columnas
            data.append(columns)
    if not data:
        st.warning("No se encontraron datos estructurados válidos en el texto extraído.")
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=[
        "NOMBRE DE LA LICITACIÓN", 
        "ENTIDAD CONVOCANTE", 
        "PROCEDIMIENTO DE CONTRATACIÓN", 
        "OFERENTES", 
        "MONTO DE LOS OFERENTES", 
        "CANTIDAD DE OFERENTES", 
        "CANTIDAD DE CONSORCIOS", 
        "CANTIDAD DE EMPRESAS", 
        "MONTOS OFERTADOS POR LOS OFERENTES EN CADA LOTE", 
        "FINANCIADOR"
    ])
    return df

# Inicializar el estado de la sesión para el chat
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Interfaz para subir archivos de imagen
image_file = st.file_uploader("Sube una imagen con texto", type=["jpg", "png", "jpeg"])

if image_file:
    try:
        # Abrir la imagen con PIL
        image = Image.open(image_file)
        st.image(image, caption='Imagen Subida', use_column_width=True)

        # Aplicar OCR a la imagen
        with st.spinner("Extrayendo texto de la imagen..."):
            extracted_text = extract_text_from_image(image)
        
        st.subheader("Texto Extraído:")
        st.write(extracted_text if extracted_text else "No se pudo extraer texto de la imagen.")

        # Interfaz de Chat para ingresar prompts
        st.markdown("## Interfaz de Chat para Procesar el Texto Extraído")
        
        # Área de texto para el prompt del usuario
        user_input = st.text_input("Ingresa tu prompt para procesar el texto extraído:", "")

        if st.button("Enviar"):
            if user_input.strip() != "":
                with st.spinner("Procesando el prompt..."):
                    response = generate_response(user_input)
                
                if response:
                    # Actualizar el historial del chat
                    st.session_state['chat_history'].append({"user": user_input, "bot": response})
            else:
                st.warning("Por favor, ingresa un prompt antes de enviar.")

        # Mostrar el historial del chat
        if st.session_state['chat_history']:
            st.markdown("### Historial del Chat")
            for chat in st.session_state['chat_history']:
                st.markdown(f"**Tú:** {chat['user']}")
                st.markdown(f"**Bot:** {chat['bot']}")

    except Exception as e:
        st.error(f"Ocurrió un error al procesar la imagen: {e}")

# Opcional: Mostrar y descargar datos estructurados si están disponibles
if st.session_state['chat_history']:
    last_response = st.session_state['chat_history'][-1]['bot']
    df = text_to_dataframe(last_response)
    
    if not df.empty:
        st.subheader("Datos Extraídos:")
        st.dataframe(df)

        # Descargar los datos como archivo CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar resultados en CSV",
            data=csv,
            file_name="datos_licitacion.csv",
            mime="text/csv"
        )
