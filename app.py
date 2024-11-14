import streamlit as st
import pandas as pd
import easyocr
from PIL import Image
import requests
import io
import os

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
El texto de la imagen será extraído y procesado para extraer datos específicos en una tabla.
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
def extract_data_with_gptneo(text):
    prompt = f"""
    Extrae y organiza los datos en una tabla. En esta imagen, los datos de interés están en ubicaciones específicas.
    Por favor, sigue estas instrucciones para extraer:
    - NOMBRE DE LA LICITACIÓN
    - ENTIDAD CONVOCANTE
    - PROCEDIMIENTO DE CONTRATACIÓN
    - OFERENTES
    - MONTO DE LOS OFERENTES
    - CANTIDAD DE OFERENTES
    - CANTIDAD DE CONSORCIOS
    - CANTIDAD DE EMPRESAS
    - MONTOS OFERTADOS POR LOS OFERENTES EN CADA LOTE
    - FINANCIADOR

    Texto:
    {text}
    """
    
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

        # Procesar el texto extraído con GPT-Neo
        if extracted_text:
            with st.spinner("Procesando el texto para extraer datos estructurados..."):
                structured_data = extract_data_with_gptneo(extracted_text)
            
            if structured_data:
                st.subheader("Datos Extraídos:")
                st.write(structured_data)
                
                # Convertir el texto estructurado a un DataFrame
                df = text_to_dataframe(structured_data)
                
                if not df.empty:
                    # Mostrar la tabla en Streamlit
                    st.dataframe(df)

                    # Descargar los datos como archivo CSV
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Descargar resultados en CSV",
                        data=csv,
                        file_name="datos_licitacion.csv",
                        mime="text/csv"
                    )
    except Exception as e:
        st.error(f"Ocurrió un error al procesar la imagen: {e}")
