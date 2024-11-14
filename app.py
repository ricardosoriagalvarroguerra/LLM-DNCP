import streamlit as st
import pandas as pd
import easyocr
from PIL import Image
import requests
import io

# Configuración de la API de Hugging Face
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
API_TOKEN = "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"  # Reemplaza con tu token de Hugging Face
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Inicializar el lector de EasyOCR
reader = easyocr.Reader(['es'])  # 'es' para español

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
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

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
    
    response = query({"inputs": prompt, "parameters": {"max_new_tokens": 500}})
    
    if response and isinstance(response, list) and "generated_text" in response[0]:
        return response[0]["generated_text"]
    else:
        st.error("Error en la respuesta del modelo")
        st.write(response)  # Para ver detalles del error en Streamlit
        return None

# Función para convertir el texto estructurado en un DataFrame
def text_to_dataframe(structured_text):
    data = []
    lines = structured_text.split("\n")
    for line in lines:
        columns = line.split(",")  
        if len(columns) == 10:  
            data.append(columns)
    df = pd.DataFrame(data, columns=[
        "NOMBRE DE LA LICITACIÓN", "ENTIDAD CONVOCANTE", "PROCEDIMIENTO DE CONTRATACIÓN", "OFERENTES", 
        "MONTO DE LOS OFERENTES", "CANTIDAD DE OFERENTES", "CANTIDAD DE CONSORCIOS", 
        "CANTIDAD DE EMPRESAS", "MONTOS OFERTADOS POR LOS OFERENTES EN CADA LOTE", "FINANCIADOR"
    ])
    return df

# Interfaz de Streamlit
st.title("Extractor de Datos de Licitaciones usando OCR en Imágenes")

# Subir archivo de imagen
image_file = st.file_uploader("Sube una imagen con texto", type=["jpg", "png", "jpeg"])

if image_file:
    # Abrir la imagen con PIL
    image = Image.open(image_file)

    # Mostrar la imagen subida
    st.image(image, caption='Imagen Subida', use_column_width=True)

    # Aplicar OCR a la imagen
    with st.spinner("Procesando el documento..."):
        extracted_text = extract_text_from_image(image)

    st.subheader("Texto Extraído:")
    st.write(extracted_text)

    # Extraer datos usando el modelo GPT-Neo
    if extracted_text:
        with st.spinner("Extrayendo datos estructurados..."):
            structured_data = extract_data_with_gptneo(extracted_text)
        
        if structured_data:
            st.subheader("Datos extraídos:")
            
            # Convertir los datos estructurados a un DataFrame
            df = text_to_dataframe(structured_data)
            
            # Mostrar la tabla en Streamlit
            st.write(df)
            
            # Descargar los datos como archivo CSV
            csv = df.to_csv(index=False)
            st.download_button(
                "Descargar resultados en CSV",
                data=csv,
                file_name="datos_licitacion.csv",
                mime="text/csv"
            )


