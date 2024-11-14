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
El texto de la imagen será extraído y procesado para extraer datos específicos de cada OFERENTE y el monto total pagado por oferente, organizándolos en una tabla.
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

# Función para generar respuesta usando GPT-Neo con prompt fijo
def extract_oferentes_y_ofertas(text):
    prompt = f"""
    Extrae el nombre de cada observación de la columan Oferentes o variaciones del nombre de la columna.
    Extrae el Monto total de oferta que son numeros que esta en la columna o seguido del texto Garantia de Mantenimiento de OFerta.
    crea una tabla con la columna Oferentes y el Monto total de la Oferta.

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
    
    # Verificar si hay encabezados y omitirlos
    if lines and ("Oferentes" in lines[0].upper() or "Garantia de Mantenimiento de OFerta" in lines[0].upper()):
        lines = lines[1:]
    
    for line in lines:
        # Separar por comas, asumiendo que los campos están separados por comas
        columns = [col.strip() for col in line.split(",")]
        if len(columns) == 2:  # Ahora solo hay 2 columnas
            try:
                oferente = columns[0]
                # Convertir ofertas a float, eliminando símbolos como '$' y comas
                ofertas = float(columns[1].replace('$', '').replace(',', '').strip())
                data.append([oferente, ofertas])
            except ValueError:
                st.warning(f"Formato incorrecto en la línea: {line}")
    if not data:
        st.warning("No se encontraron datos estructurados válidos en el texto extraído.")
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=[
        "OFERENTE", 
        "OFERTAS"
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
            with st.spinner("Procesando el texto para extraer OFERENTE y OFERTAS..."):
                structured_data = extract_oferentes_y_ofertas(extracted_text)
            
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

