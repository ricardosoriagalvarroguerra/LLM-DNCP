import streamlit as st
import requests
import fitz  # PyMuPDF para extracción de texto de PDF
import pandas as pd
import io

# Coloca tu API Token de Hugging Face aquí
API_TOKEN = "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"  # Reemplaza con tu clave de API
headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

# URL del modelo Llama 2 7B Chat en Hugging Face
model_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"

# Función para extraer texto del PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    # Abrir el archivo PDF desde el objeto en memoria (BytesIO) en lugar de una ruta
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Función para enviar el texto al modelo Llama 2 y extraer datos
def extract_data_with_llama2(text):
    prompt = f"""
    Extrae y organiza los datos en una tabla con las siguientes columnas:
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
    
    response = requests.post(
        model_url,
        headers=headers,
        json={"inputs": prompt, "parameters": {"max_length": 1000}}
    )
    
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        st.error(f"Error en la llamada al modelo: {response.status_code}")
        return None

# Función para convertir el texto estructurado en un DataFrame
def text_to_dataframe(structured_text):
    # Este es un ejemplo; adapta este proceso de acuerdo a cómo esté estructurado el texto
    data = []
    lines = structured_text.split("\n")
    for line in lines:
        columns = line.split(",")  # Asume que los datos están separados por comas
        if len(columns) == 10:  # Asegúrate de que cada línea tiene el número correcto de columnas
            data.append(columns)
    df = pd.DataFrame(data, columns=[
        "NOMBRE DE LA LICITACIÓN", "ENTIDAD CONVOCANTE", "PROCEDIMIENTO DE CONTRATACIÓN", "OFERENTES", 
        "MONTO DE LOS OFERENTES", "CANTIDAD DE OFERENTES", "CANTIDAD DE CONSORCIOS", 
        "CANTIDAD DE EMPRESAS", "MONTOS OFERTADOS POR LOS OFERENTES EN CADA LOTE", "FINANCIADOR"
    ])
    return df

# Interfaz de Streamlit
st.title("Extractor de Datos de Licitaciones en PDF usando Llama 2")

# Subir archivo PDF
pdf_file = st.file_uploader("Sube un archivo PDF", type=["pdf"])

if pdf_file:
    # Extraer texto del PDF
    text = extract_text_from_pdf(pdf_file)
    
    # Extraer datos usando el modelo Llama 2
    with st.spinner("Procesando el documento..."):
        structured_data = extract_data_with_llama2(text)
    
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

