import streamlit as st
import fitz  # PyMuPDF para extracción de texto de PDF
import pandas as pd
from huggingface_hub import InferenceClient

# Configuración de la API de Hugging Face
API_TOKEN = "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"  # Reemplaza con tu nuevo token de lectura
client = InferenceClient(api_key=API_TOKEN)

# Función para extraer texto del PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    # Abrir el archivo PDF desde el objeto en memoria (BytesIO) en lugar de una ruta
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Función para extraer datos usando el modelo Llama 2
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

    # Llamada al modelo Llama 2 usando client.post para enviar el prompt directamente
    response = client.post(
        f"https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
        json={"inputs": prompt, "parameters": {"max_new_tokens": 500}}
    )

    # Verificar si la respuesta contiene 'generated_text'
    if "generated_text" in response:
        return response["generated_text"]
    else:
        st.error("Error en la respuesta del modelo")
        st.write(response)  # Para ver detalles del error en Streamlit
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
