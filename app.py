import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io
import requests

# Configuración de la API de Hugging Face
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
API_TOKEN = "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"  # Reemplaza con tu token de Hugging Face
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Función para hacer OCR en una imagen
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Función para realizar la solicitud a la API de Hugging Face
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Función para generar texto usando el modelo GPT-Neo
def extract_data_with_gptneo(text):
    prompt = f"""
    Extrae y organiza los datos en una tabla. En este PDF escaneado, los datos de interés están en ubicaciones específicas.
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
    
    if response and "generated_text" in response[0]:
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
st.title("Extractor de Datos de Licitaciones en PDF Escaneado usando GPT-Neo")

# Subir archivo PDF
pdf_file = st.file_uploader("Sube un archivo PDF escaneado", type=["pdf"])

if pdf_file:
    # Convertir el PDF a imágenes
    images = convert_from_path(pdf_file)
    
    # Aplicar OCR a cada imagen y extraer el texto
    all_text = ""
    for img in images:
        all_text += extract_text_from_image(img) + "\n"
    
    # Extraer datos usando el modelo GPT-Neo
    with st.spinner("Procesando el documento..."):
        structured_data = extract_data_with_gptneo(all_text)
    
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


