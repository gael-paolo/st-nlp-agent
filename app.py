import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import io
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from docx import Document
import base64

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Analytics Assistant Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Analytics Assistant Pro")
st.markdown("**Análisis inteligente con IA - 6 Funcionalidades**")

# =========================================================
# CONFIGURACIÓN DE API
# =========================================================
st.sidebar.header("🔧 Configuración de IA")

provider = st.sidebar.radio(
    "Proveedor de IA:",
    ["Google Gemini", "OpenAI"]
)

api_key = st.sidebar.text_input("API Key", type="password")

if not api_key:
    st.warning("⚠️ Introduce tu API Key para comenzar")
    st.info("💡 Gemini: https://makersuite.google.com/app/apikey")
    st.info("💡 OpenAI: https://platform.openai.com/api-keys")
    st.stop()

# Inicializar cliente
try:
    if provider == "Google Gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        modelo = "gemini-2.0-flash"
    else:
        import openai
        openai.api_key = api_key
        modelo = "gpt-4o-mini"
        
    st.sidebar.success(f"✅ {provider} configurado")
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()

# =========================================================
# FUNCIÓN PARA LLAMAR A LA IA
# =========================================================
def llamar_ia(prompt, temperatura=0.1, max_tokens=1000):
    """Función simple para llamar a la IA"""
    try:
        if provider == "Google Gemini":
            respuesta = client.models.generate_content(
                model=modelo,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperatura,
                    max_output_tokens=max_tokens
                )
            )
            return respuesta.text.strip()
        else:
            respuesta = openai.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": "Eres un asistente analítico experto."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperatura,
                max_tokens=max_tokens
            )
            return respuesta.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error en la IA: {str(e)}")
        return None

# =========================================================
# FUNCIÓN PARA GENERAR EMBEDDINGS
# =========================================================
def generar_embeddings(textos):
    """Genera embeddings usando la API de IA"""
    try:
        embeddings = []
        
        if provider == "Google Gemini":
            # Usar embeddings de Gemini
            for text in textos:
                emb = client.models.embed_content(
                    model="text-embedding-004",
                    contents=str(text)
                )
                embeddings.append(emb.embeddings[0].values)
        else:
            # Usar embeddings de OpenAI
            for text in textos:
                response = openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=str(text)
                )
                embeddings.append(response.data[0].embedding)
        
        return np.array(embeddings)
    except Exception as e:
        st.error(f"Error al generar embeddings: {str(e)}")
        return None

# =========================================================
# FUNCIONES PARA PROCESAR DOCUMENTOS
# =========================================================
def extraer_texto_pdf(archivo):
    """Extrae texto de archivos PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(archivo)
        texto = ""
        for pagina in pdf_reader.pages:
            texto += pagina.extract_text() + "\n"
        return texto
    except Exception as e:
        st.error(f"Error al leer PDF: {str(e)}")
        return ""

def extraer_texto_docx(archivo):
    """Extrae texto de archivos DOCX"""
    try:
        doc = Document(archivo)
        texto = ""
        for parrafo in doc.paragraphs:
            texto += parrafo.text + "\n"
        return texto
    except Exception as e:
        st.error(f"Error al leer DOCX: {str(e)}")
        return ""

def extraer_texto_txt(archivo):
    """Extrae texto de archivos TXT"""
    try:
        return archivo.read().decode("utf-8")
    except:
        try:
            archivo.seek(0)
            return archivo.read().decode("utf-8", errors='ignore')
        except Exception as e:
            st.error(f"Error al leer TXT: {str(e)}")
            return ""

def procesar_documento(archivo):
    """Procesa diferentes tipos de documentos y extrae texto"""
    nombre = archivo.name
    if nombre.endswith('.pdf'):
        return extraer_texto_pdf(archivo)
    elif nombre.endswith('.docx'):
        return extraer_texto_docx(archivo)
    elif nombre.endswith('.txt'):
        return extraer_texto_txt(archivo)
    else:
        st.error(f"Formato no soportado: {nombre}")
        return ""

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def agrupar_textos_manual(textos, n_clusters=3):
    """Agrupamiento básico de textos usando TF-IDF y similitud de coseno"""
    if len(textos) < n_clusters:
        n_clusters = len(textos)
    
    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer(stop_words='spanish', max_features=1000)
    try:
        X = vectorizer.fit_transform(textos)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Calcular similitud dentro de cada cluster
        resultados = []
        for i, (texto, cluster) in enumerate(zip(textos, clusters)):
            cluster_texts = [textos[j] for j in range(len(textos)) if clusters[j] == cluster]
            if len(cluster_texts) > 1:
                similitudes = cosine_similarity(X[i:i+1], X[clusters == cluster])[0]
                similitud_promedio = np.mean(similitudes) * 100
            else:
                similitud_promedio = 100
            
            # Obtener palabras clave del cluster
            cluster_indices = np.where(clusters == cluster)[0]
            if len(cluster_indices) > 0:
                # Calcular palabras más frecuentes en el cluster
                cluster_features = X[cluster_indices].sum(axis=0).A1
                top_feature_indices = cluster_features.argsort()[-5:][::-1]
                feature_names = vectorizer.get_feature_names_out()
                palabras_clave = [feature_names[idx] for idx in top_feature_indices if cluster_features[idx] > 0]
            else:
                palabras_clave = []
            
            resultados.append({
                "texto": texto[:100] + "..." if len(texto) > 100 else texto,
                "grupo": f"Grupo {cluster + 1}",
                "similitud_grupo": f"{similitud_promedio:.1f}%",
                "tamano_grupo": sum(clusters == cluster),
                "palabras_clave": ", ".join(palabras_clave[:3]) if palabras_clave else "N/A"
            })
        
        return resultados
        
    except Exception as e:
        # Fallback a agrupamiento simple por longitud
        longitudes = [len(t) for t in textos]
        if len(set(longitudes)) > 1:
            percentiles = np.percentile(longitudes, [33, 66])
            clusters = np.digitize(longitudes, percentiles)
        else:
            clusters = [0] * len(textos)
        
        resultados = []
        for i, (texto, cluster) in enumerate(zip(textos, clusters)):
            resultados.append({
                "texto": texto[:100] + "..." if len(texto) > 100 else texto,
                "grupo": f"Grupo {cluster + 1}",
                "similitud_grupo": "N/A",
                "tamano_grupo": sum(clusters == cluster),
                "palabras_clave": "Agrupado por longitud"
            })
        
        return resultados

# =========================================================
# CATÁLOGO DE FUNCIONALIDADES
# =========================================================
st.header("🎯 Selecciona una funcionalidad")

funcionalidades = {
    "1": {
        "nombre": "📝 Análisis de Sentimiento",
        "descripcion": "Clasifica textos en positivo, neutral o negativo",
        "ejemplo": "Comentarios de clientes, reseñas, feedback",
        "icono": "😊"
    },
    "2": {
        "nombre": "🔍 Extracción de Información",
        "descripcion": "Extrae datos estructurados de textos no estructurados",
        "ejemplo": "Descripciones de productos, necesidades de clientes",
        "icono": "📋"
    },
    "3": {
        "nombre": "⚠️ Clasificación de Textos",
        "descripcion": "Organiza textos en categorías personalizadas",
        "ejemplo": "Clasificar tickets, priorizar tareas",
        "icono": "🏷️"
    },
    "4": {
        "nombre": "📊 Agrupamiento de Textos",
        "descripcion": "Encuentra textos similares y agrupa automáticamente",
        "ejemplo": "Agrupar comentarios similares, organizar ideas",
        "icono": "👥"
    },
    "5": {
        "nombre": "✉️ Redactor de Correos",
        "descripcion": "Crea correos profesionales parametrizados",
        "ejemplo": "Correos comerciales, seguimientos, anuncios",
        "icono": "📧"
    },
    "6": {
        "nombre": "💡 Constructor de Prompts",
        "descripcion": "Crea prompts profesionales para IA",
        "ejemplo": "Mejorar prompts básicos, estructurar consultas",
        "icono": "🎯"
    }
}

# Mostrar funcionalidades en 3 columnas
cols = st.columns(3)

for idx, (key, func) in enumerate(funcionalidades.items()):
    with cols[idx % 3]:
        with st.container(border=True, height=200):
            st.markdown(f"### {func['icono']} {func['nombre']}")
            st.markdown(f"**{func['descripcion']}**")
            st.markdown(f"*Ejemplo:* {func['ejemplo']}")
            
            if st.button("Seleccionar", key=f"btn_{key}"):
                st.session_state.funcionalidad = key
                st.session_state.nombre_func = func['nombre']
                st.rerun()

# Inicializar estado
if 'funcionalidad' not in st.session_state:
    st.session_state.funcionalidad = None
    st.session_state.nombre_func = None
    st.session_state.df = None
    st.session_state.config = {}
    st.session_state.datos_listos = False
    st.session_state.documento_texto = ""

# =========================================================
# EJECUCIÓN DE FUNCIONALIDADES
# =========================================================
if st.session_state.funcionalidad:
    st.header(f"🔧 {st.session_state.nombre_func}")
    
    func_info = funcionalidades[st.session_state.funcionalidad]
    
    # =========================================================
    # 1. ANÁLISIS DE SENTIMIENTO
    # =========================================================
    if st.session_state.funcionalidad == "1":
        st.subheader("📝 Análisis de Sentimiento con IA")
        
        # Opciones para cargar datos
        opcion_datos = st.radio(
            "Fuente de textos:",
            ["Pegar textos", "Cargar archivo", "Usar ejemplo"]
        )
        
        textos = []
        
        if opcion_datos == "Pegar textos":
            input_texto = st.text_area(
                "Ingresa los textos (uno por línea):",
                "El producto es excelente, me encantó\nServicio regular, podría mejorar\nMuy mala experiencia, no lo recomiendo\nTodo bien, sin problemas",
                height=150
            )
            textos = [t.strip() for t in input_texto.split('\n') if t.strip()]
            
        elif opcion_datos == "Cargar archivo":
            archivo = st.file_uploader("Sube archivo TXT, CSV, PDF o DOCX", type=["txt", "csv", "pdf", "docx"])
            if archivo:
                if archivo.name.endswith('.csv'):
                    df_temp = pd.read_csv(archivo)
                    # Buscar columna con texto
                    posibles_columnas = [c for c in df_temp.columns if any(word in c.lower() for word in ['texto', 'comentario', 'review', 'mensaje'])]
                    if posibles_columnas:
                        textos = df_temp[posibles_columnas[0]].dropna().astype(str).tolist()
                    else:
                        textos = df_temp.iloc[:, 0].dropna().astype(str).tolist()
                elif archivo.name.endswith('.pdf') or archivo.name.endswith('.docx'):
                    contenido = procesar_documento(archivo)
                    if contenido:
                        textos = [parrafo.strip() for parrafo in contenido.split('\n') if parrafo.strip()]
                else:  # TXT
                    contenido = extraer_texto_txt(archivo)
                    textos = [t.strip() for t in contenido.split('\n') if t.strip()]
        
        else:  # Usar ejemplo
            textos_ejemplo = [
                "Excelente producto, superó mis expectativas",
                "El servicio fue regular, nada especial",
                "Muy mala atención al cliente",
                "Recomiendo ampliamente, muy bueno",
                "No funciona como se describe en la página",
                "Calidad aceptable por el precio",
                "Horrible experiencia, nunca más",
                "Rápido y eficiente, muy satisfecho"
            ]
            textos = textos_ejemplo
            st.info("📋 Textos de ejemplo cargados")
            for i, texto in enumerate(textos):
                st.write(f"{i+1}. {texto}")
        
        if textos:
            st.success(f"✅ {len(textos)} textos cargados")
            
            # Opciones de análisis
            col1, col2 = st.columns(2)
            
            with col1:
                incluir_detalles = st.checkbox("Mostrar detalles técnicos", value=True)
            
            with col2:
                temperatura = st.slider("Precisión/Temperatura:", 0.0, 1.0, 0.1, 0.1)
            
            if st.button("🔍 Analizar Sentimiento", type="primary"):
                with st.spinner("Analizando sentimientos con IA..."):
                    # Análisis con IA
                    prompt = f"""
                    Analiza el sentimiento de los siguientes textos y clasifícalos como POSITIVO, NEUTRAL o NEGATIVO.
                    
                    Para cada texto, proporciona:
                    1. Sentimiento (POSITIVO/NEUTRAL/NEGATIVO)
                    2. Confianza (0-100%)
                    3. Palabras clave que influyeron
                    4. Breve explicación
                    
                    Textos:
                    {chr(10).join([f'{i+1}. {t}' for i, t in enumerate(textos)])}
                    
                    Devuelve el resultado en formato JSON con esta estructura:
                    {{
                      "analisis": [
                        {{
                          "texto": "texto original",
                          "sentimiento": "POSITIVO/NEUTRAL/NEGATIVO",
                          "confianza": "85%",
                          "palabras_clave": ["palabra1", "palabra2"],
                          "explicacion": "Breve explicación"
                        }}
                      ],
                      "resumen": {{
                        "total": 10,
                        "positivos": 4,
                        "neutrales": 3,
                        "negativos": 3,
                        "sentimiento_promedio": "NEUTRAL"
                      }}
                    }}
                    """
                    
                    resultado = llamar_ia(prompt, temperatura=temperatura)
                    
                    if resultado:
                        try:
                            # Intentar extraer JSON
                            json_match = re.search(r'\{.*\}', resultado, re.DOTALL)
                            if json_match:
                                datos = json.loads(json_match.group())
                            else:
                                # Si no es JSON válido, mostrar texto directo
                                st.markdown("### Resultados del Análisis")
                                st.text(resultado)
                                datos = None
                            
                            if datos:
                                # Mostrar resultados
                                st.subheader("📊 Resultados del Análisis")
                                
                                # Resumen
                                if "resumen" in datos:
                                    res = datos["resumen"]
                                    col_res1, col_res2, col_res3 = st.columns(3)
                                    with col_res1:
                                        st.metric("Positivos", res.get("positivos", 0))
                                    with col_res2:
                                        st.metric("Neutrales", res.get("neutrales", 0))
                                    with col_res3:
                                        st.metric("Negativos", res.get("negativos", 0))
                                
                                # Detalle
                                if "analisis" in datos:
                                    df_resultados = pd.DataFrame(datos["analisis"])
                                    st.dataframe(df_resultados, use_container_width=True)
                                    
                                    # Gráfico
                                    if "sentimiento" in df_resultados.columns:
                                        fig, ax = plt.subplots(figsize=(8, 4))
                                        df_resultados["sentimiento"].value_counts().plot(kind='bar', ax=ax, color=['green', 'gray', 'red'])
                                        ax.set_title("Distribución de Sentimientos")
                                        ax.set_xlabel("Sentimiento")
                                        ax.set_ylabel("Cantidad")
                                        st.pyplot(fig)
                                
                        except json.JSONDecodeError:
                            st.markdown("### Resultados del Análisis")
                            st.text(resultado)
    
    # =========================================================
    # 2. EXTRACCIÓN DE INFORMACIÓN
    # =========================================================
    elif st.session_state.funcionalidad == "2":
        st.subheader("🔍 Extracción de Información Personalizada con IA")
        
        # Configurar campos personalizados
        st.write("### ⚙️ Configurar Campos a Extraer")
        
        campos_input = st.text_area(
            "Ingresa los campos que deseas extraer (uno por línea):",
            "nombre\nmonto\ntipo gasto\nCI\nfecha\nproveedor",
            height=100,
            help="Escribe cada campo en una línea separada. Ejemplo: nombre, monto, CI, etc."
        )
        
        campos_personalizados = [campo.strip() for campo in campos_input.split('\n') if campo.strip()]
        
        if not campos_personalizados:
            campos_personalizados = ["nombre", "monto", "fecha", "CI"]
            st.warning("Usando campos por defecto: nombre, monto, fecha, CI")
        
        # Opciones para cargar datos
        st.write("### 📂 Cargar Datos")
        opcion_datos = st.radio(
            "Fuente de datos:",
            ["Pegar textos", "Cargar archivo", "Usar ejemplo personalizado"]
        )
        
        textos = []
        
        if opcion_datos == "Pegar textos":
            st.info(f"💡 **Ejemplo de formato para extraer: {', '.join(campos_personalizados)}**")
            ejemplo = "\n".join([
                f"Gasto aprobado para Juan Pérez, monto: $1,500.00, tipo: viáticos, CI: V-12345678, fecha: 15/01/2024",
                f"Reembolso a María González por $750.50, concepto: materiales, cédula: 98765432, fecha: 20/01/2024",
                f"Pago a proveedor TechCorp, monto total: $5,200.00, tipo: servicios, RIF: J-301234567, fecha: 25/01/2024"
            ])
            
            input_texto = st.text_area(
                "Ingresa los textos para extraer información:",
                ejemplo,
                height=150
            )
            textos = [t.strip() for t in input_texto.split('\n') if t.strip()]
            
        elif opcion_datos == "Cargar archivo":
            archivo = st.file_uploader("Sube archivo TXT, CSV, PDF o DOCX", type=["txt", "csv", "pdf", "docx"])
            if archivo:
                if archivo.name.endswith('.csv'):
                    df_temp = pd.read_csv(archivo)
                    posibles_columnas = [c for c in df_temp.columns if any(word in c.lower() for word in ['texto', 'descripcion', 'contenido', 'mensaje', 'observacion'])]
                    if posibles_columnas:
                        textos = df_temp[posibles_columnas[0]].dropna().astype(str).tolist()
                    else:
                        textos = df_temp.iloc[:, 0].dropna().astype(str).tolist()
                elif archivo.name.endswith('.pdf') or archivo.name.endswith('.docx'):
                    contenido = procesar_documento(archivo)
                    if contenido:
                        textos = [parrafo.strip() for parrafo in contenido.split('\n') if parrafo.strip()]
                else:  # TXT
                    contenido = extraer_texto_txt(archivo)
                    textos = [t.strip() for t in contenido.split('\n') if t.strip()]
        
        else:  # Usar ejemplo personalizado
            st.info("📋 **Datos de ejemplo con los campos configurados:**")
            
            # Crear datos de ejemplo basados en los campos solicitados
            ejemplo_datos = [
                f"Gasto aprobado para Juan Pérez, monto: $1,500.00, tipo gasto: viáticos, CI: V-12345678, fecha: 15/01/2024, proveedor: N/A",
                f"Reembolso a María González por $750.50, tipo gasto: materiales, CI: 98765432, fecha: 20/01/2024, proveedor: María González",
                f"Pago a proveedor TechCorp, monto: $5,200.00, tipo gasto: servicios, CI: J-301234567, fecha: 25/01/2024, proveedor: TechCorp",
                f"Compra de equipo, nombre: Carlos Ruiz, monto: $3,750.00, tipo gasto: activos, CI: 11223344, fecha: 10/01/2024, proveedor: OfficeSupply",
                f"Gasto de representación, nombre: Ana Mendoza, monto: $980.00, tipo gasto: representación, CI: E-99887766, fecha: 05/01/2024, proveedor: Restaurant Elite"
            ]
            
            textos = ejemplo_datos
            
            # Mostrar tabla de ejemplo
            datos_ejemplo = []
            for i, texto in enumerate(textos, 1):
                datos_ejemplo.append({
                    "Registro": i,
                    "Texto": texto
                })
            
            df_ejemplo = pd.DataFrame(datos_ejemplo)
            st.dataframe(df_ejemplo, use_container_width=True)
        
        if textos:
            st.success(f"✅ {len(textos)} textos cargados")
            
            # Opciones de extracción
            st.write("### ⚙️ Opciones de Extracción")
            col1, col2 = st.columns(2)
            
            with col1:
                temperatura = st.slider("Precisión/Temperatura:", 0.0, 1.0, 0.1, 0.1)
                max_resultados = st.slider("Máximo resultados por campo:", 1, 10, 3)
            
            with col2:
                formatear_tabla = st.checkbox("Formatear como tabla", value=True)
            
            if st.button("🔍 Extraer Información", type="primary"):
                with st.spinner("Extrayendo información con IA..."):
                    prompt = f"""
                    Extrae la siguiente información de los textos proporcionados:
                    CAMPOS SOLICITADOS: {', '.join(campos_personalizados)}
                    
                    Para cada texto, extrae TODA la información relevante de los campos solicitados.
                    Si un campo no está presente en el texto, déjalo vacío.
                    
                    Textos:
                    {chr(10).join([f'{i+1}. {t}' for i, t in enumerate(textos)])}
                    
                    Devuelve el resultado en formato JSON con esta estructura:
                    {{
                      "extracciones": [
                        {{
                          "texto_original": "texto completo",
                          "campos_extraidos": {{
                            "campo1": "valor1",
                            "campo2": "valor2",
                            ...
                          }}
                        }}
                      ],
                      "resumen": {{
                        "total_textos": {len(textos)},
                        "campos_encontrados": ["campo1", "campo2"],
                        "textos_con_datos": 5
                      }}
                    }}
                    
                    Solo devuelve valores extraídos, no inventes información.
                    """
                    
                    resultado = llamar_ia(prompt, temperatura=temperatura, max_tokens=2000)
                    
                    if resultado:
                        try:
                            json_match = re.search(r'\{.*\}', resultado, re.DOTALL)
                            if json_match:
                                datos = json.loads(json_match.group())
                                
                                st.subheader("📋 Información Extraída")
                                
                                if "extracciones" in datos:
                                    # Crear DataFrame para mostrar
                                    filas = []
                                    for extraccion in datos["extracciones"]:
                                        fila = {"Texto": extraccion['texto_original'][:100] + "..."}
                                        campos = extraccion.get('campos_extraidos', {})
                                        for campo in campos_personalizados:
                                            fila[campo] = campos.get(campo, 'No encontrado')
                                        filas.append(fila)
                                    
                                    df_resultados = pd.DataFrame(filas)
                                    
                                    if formatear_tabla:
                                        st.dataframe(df_resultados, use_container_width=True)
                                    else:
                                        for i, fila in enumerate(filas):
                                            with st.expander(f"Registro {i+1}: {fila['Texto']}"):
                                                for campo, valor in fila.items():
                                                    if campo != 'Texto':
                                                        st.write(f"**{campo.title()}:** {valor}")
                                
                                # Mostrar estadísticas
                                if "resumen" in datos:
                                    st.subheader("📊 Estadísticas de Extracción")
                                    res = datos["resumen"]
                                    cols_stats = st.columns(3)
                                    with cols_stats[0]:
                                        st.metric("Total textos", res.get("total_textos", 0))
                                    with cols_stats[1]:
                                        st.metric("Textos con datos", res.get("textos_con_datos", 0))
                                    with cols_stats[2]:
                                        campos_encontrados = res.get("campos_encontrados", [])
                                        st.metric("Campos encontrados", len(campos_encontrados))
                                    
                                    # Mostrar campos encontrados
                                    st.write("**Campos extraídos exitosamente:**")
                                    st.write(", ".join(campos_encontrados) if campos_encontrados else "Ninguno")
                            
                        except json.JSONDecodeError:
                            st.markdown("### Resultados de Extracción")
                            st.text(resultado)
    
    # =========================================================
    # 3. CLASIFICACIÓN DE TEXTOS
    # =========================================================
    elif st.session_state.funcionalidad == "3":
        st.subheader("⚠️ Clasificación de Textos con IA")
        
        # Opciones para cargar datos
        opcion_datos = st.radio(
            "Fuente de textos:",
            ["Pegar textos", "Cargar archivo", "Usar ejemplo"]
        )
        
        textos = []
        
        if opcion_datos == "Pegar textos":
            input_texto = st.text_area(
                "Ingresa los textos para clasificar:",
                "Error en el sistema de pago\nSolicitud de nuevo usuario\nConsulta sobre facturación\nProblema con inicio de sesión\nPetición de característica nueva",
                height=150
            )
            textos = [t.strip() for t in input_texto.split('\n') if t.strip()]
            
        elif opcion_datos == "Cargar archivo":
            archivo = st.file_uploader("Sube archivo TXT, CSV, PDF o DOCX", type=["txt", "csv", "pdf", "docx"])
            if archivo:
                if archivo.name.endswith('.csv'):
                    df_temp = pd.read_csv(archivo)
                    posibles_columnas = [c for c in df_temp.columns if any(word in c.lower() for word in ['texto', 'descripcion', 'asunto', 'mensaje'])]
                    if posibles_columnas:
                        textos = df_temp[posibles_columnas[0]].dropna().astype(str).tolist()
                    else:
                        textos = df_temp.iloc[:, 0].dropna().astype(str).tolist()
                elif archivo.name.endswith('.pdf') or archivo.name.endswith('.docx'):
                    contenido = procesar_documento(archivo)
                    if contenido:
                        textos = [parrafo.strip() for parrafo in contenido.split('\n') if parrafo.strip()]
                else:  # TXT
                    contenido = extraer_texto_txt(archivo)
                    textos = [t.strip() for t in contenido.split('\n') if t.strip()]
        
        else:  # Usar ejemplo
            textos_ejemplo = [
                "El sistema no permite cargar archivos grandes",
                "¿Cómo puedo resetear mi contraseña?",
                "Necesito acceso a la base de datos de clientes",
                "La aplicación se cierra inesperadamente",
                "Consulta sobre los precios de los planes"
            ]
            textos = textos_ejemplo
            st.info("📋 Textos de ejemplo cargados")
        
        if textos:
            st.success(f"✅ {len(textos)} textos cargados")
            
            # Configurar categorías
            st.subheader("⚙️ Configurar Categorías")
            
            categorias_input = st.text_area(
                "Ingresa las categorías (una por línea):",
                "SOPORTE TÉCNICO\nCONSULTAS GENERALES\nSOLICITUDES DE ACCESO\nREPORTE DE ERRORES\nFACTURACIÓN",
                height=100
            )
            
            categorias = [c.strip() for c in categorias_input.split('\n') if c.strip()]
            
            if not categorias:
                categorias = ["CATEGORÍA 1", "CATEGORÍA 2", "CATEGORÍA 3"]
                st.warning("Usando categorías por defecto")
            
            temperatura = st.slider("Precisión/Temperatura:", 0.0, 1.0, 0.1, 0.1)
            
            if st.button("🏷️ Clasificar Textos", type="primary"):
                with st.spinner("Clasificando textos con IA..."):
                    prompt = f"""
                    Clasifica los siguientes textos en estas categorías:
                    Categorías disponibles: {', '.join(categorias)}
                    
                    Para cada texto:
                    1. Asigna la categoría más apropiada
                    2. Proporciona una confianza del 0-100%
                    3. Da una breve justificación
                    
                    Textos:
                    {chr(10).join([f'{i+1}. {t}' for i, t in enumerate(textos)])}
                    
                    Devuelve el resultado en formato JSON con esta estructura:
                    {{
                      "clasificaciones": [
                        {{
                          "texto": "texto original",
                          "categoria": "CATEGORÍA ASIGNADA",
                          "confianza": "85%",
                          "justificacion": "Breve explicación"
                        }}
                      ],
                      "distribucion": {{
                        "CATEGORÍA 1": 3,
                        "CATEGORÍA 2": 2,
                        "CATEGORÍA 3": 1
                      }},
                      "categoria_mas_comun": "CATEGORÍA 1"
                    }}
                    """
                    
                    resultado = llamar_ia(prompt, temperatura=temperatura)
                    
                    if resultado:
                        try:
                            json_match = re.search(r'\{.*\}', resultado, re.DOTALL)
                            if json_match:
                                datos = json.loads(json_match.group())
                                
                                st.subheader("📊 Resultados de Clasificación")
                                
                                # Mostrar distribución
                                if "distribucion" in datos:
                                    st.write("**Distribución por categoría:**")
                                    dist = datos["distribucion"]
                                    fig, ax = plt.subplots(figsize=(10, 4))
                                    categorias_dist = list(dist.keys())
                                    valores = list(dist.values())
                                    ax.bar(categorias_dist, valores)
                                    ax.set_title("Distribución de Clasificaciones")
                                    ax.set_xlabel("Categoría")
                                    ax.set_ylabel("Cantidad")
                                    plt.xticks(rotation=45)
                                    st.pyplot(fig)
                                
                                # Mostrar detalles
                                if "clasificaciones" in datos:
                                    df_clasificaciones = pd.DataFrame(datos["clasificaciones"])
                                    st.dataframe(df_clasificaciones, use_container_width=True)
                            
                        except json.JSONDecodeError:
                            st.markdown("### Resultados de Clasificación")
                            st.text(resultado)
    
    # =========================================================
    # 4. AGRUPAMIENTO DE TEXTOS
    # =========================================================
    elif st.session_state.funcionalidad == "4":
        st.subheader("📊 Agrupamiento de Textos")
        
        # Opciones para cargar datos
        opcion_datos = st.radio(
            "Fuente de textos:",
            ["Pegar textos", "Cargar archivo", "Usar ejemplo"]
        )
        
        textos = []
        
        if opcion_datos == "Pegar textos":
            input_texto = st.text_area(
                "Ingresa los textos para agrupar:",
                "Me encanta este producto, es muy útil\nEl servicio al cliente es excelente\nNo funciona correctamente, estoy decepcionado\nMuy mala calidad, no lo recomiendo\nBuen precio por lo que ofrece\nRegular, esperaba más por el precio",
                height=150
            )
            textos = [t.strip() for t in input_texto.split('\n') if t.strip()]
            
        elif opcion_datos == "Cargar archivo":
            archivo = st.file_uploader("Sube archivo TXT, CSV, PDF o DOCX", type=["txt", "csv", "pdf", "docx"])
            if archivo:
                if archivo.name.endswith('.csv'):
                    df_temp = pd.read_csv(archivo)
                    posibles_columnas = [c for c in df_temp.columns if any(word in c.lower() for word in ['texto', 'comentario', 'review', 'mensaje'])]
                    if posibles_columnas:
                        textos = df_temp[posibles_columnas[0]].dropna().astype(str).tolist()
                    else:
                        textos = df_temp.iloc[:, 0].dropna().astype(str).tolist()
                elif archivo.name.endswith('.pdf') or archivo.name.endswith('.docx'):
                    contenido = procesar_documento(archivo)
                    if contenido:
                        textos = [parrafo.strip() for parrafo in contenido.split('\n') if parrafo.strip()]
                else:  # TXT
                    contenido = extraer_texto_txt(archivo)
                    textos = [t.strip() for t in contenido.split('\n') if t.strip()]
        
        else:  # Usar ejemplo
            textos_ejemplo = [
                "Excelente atención al cliente, muy amables",
                "El producto llegó dañado, mala experiencia",
                "Buen servicio, rápido y eficiente",
                "No respetan los tiempos de entrega",
                "Calidad premium, vale cada peso",
                "Pésima comunicación con el vendedor",
                "Recomendado 100%, volveré a comprar",
                "No cumple con lo prometido"
            ]
            textos = textos_ejemplo
            st.info("📋 Textos de ejemplo cargados")
        
        if textos:
            st.success(f"✅ {len(textos)} textos cargados")
            
            # Configurar agrupamiento
            st.subheader("⚙️ Configurar Agrupamiento")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_grupos = st.slider(
                    "Número de grupos:",
                    min_value=2,
                    max_value=min(10, len(textos)),
                    value=min(3, len(textos))
                )
                
                mostrar_palabras_clave = st.checkbox("Mostrar palabras clave por grupo", value=True)
            
            with col2:
                metodo = st.radio(
                    "Método de agrupamiento:",
                    ["Usar IA (semántico)", "Agrupamiento con K-means (Embeddings)"]
                )
                
                if metodo == "Usar IA (semántico)":
                    temperatura = st.slider("Creatividad:", 0.0, 1.0, 0.2, 0.1)
            
            if st.button("👥 Agrupar Textos", type="primary"):
                with st.spinner("Agrupando textos..."):
                    if metodo == "Usar IA (semántico)":
                        prompt = f"""
                        Analiza los siguientes textos y agrupalos en {n_grupos} grupos basándote en su contenido semántico.
                        
                        Para cada grupo, proporciona:
                        1. Nombre del grupo que represente el tema común
                        2. Descripción del patrón encontrado
                        3. Palabras clave que definen el grupo
                        4. Los textos que pertenecen a este grupo
                        5. Porcentaje de similitud promedio dentro del grupo
                        
                        Textos:
                        {chr(10).join([f'{i+1}. {t}' for i, t in enumerate(textos)])}
                        
                        Devuelve el resultado en formato JSON con esta estructura:
                        {{
                          "grupos": [
                            {{
                              "nombre": "Nombre del grupo",
                              "descripcion": "Descripción del patrón",
                              "palabras_clave": ["palabra1", "palabra2", "palabra3"],
                              "similitud_promedio": "85%",
                              "textos": [
                                {{
                                  "texto": "texto original",
                                  "similitud": "Por qué pertenece a este grupo"
                                }}
                              ],
                              "tamano": 4
                            }}
                          ],
                          "resumen": {{
                            "total_grupos": {n_grupos},
                            "total_textos": {len(textos)},
                            "grupo_mas_grande": "Nombre del grupo",
                            "grupo_mas_pequeno": "Nombre del grupo",
                            "similitud_promedio_total": "78%"
                          }}
                        }}
                        """
                        
                        resultado = llamar_ia(prompt, temperatura=temperatura)
                        
                        if resultado:
                            try:
                                json_match = re.search(r'\{.*\}', resultado, re.DOTALL)
                                if json_match:
                                    datos = json.loads(json_match.group())
                                    
                                    st.subheader("👥 Grupos Encontrados")
                                    
                                    if "grupos" in datos:
                                        for i, grupo in enumerate(datos["grupos"]):
                                            with st.expander(f"Grupo {i+1}: {grupo['nombre']} ({grupo.get('tamano', 0)} textos) - Similitud: {grupo.get('similitud_promedio', 'N/A')}"):
                                                st.write(f"**Descripción:** {grupo.get('descripcion', 'N/A')}")
                                                
                                                if "palabras_clave" in grupo:
                                                    st.write(f"**Palabras clave:** {', '.join(grupo['palabras_clave'])}")
                                                
                                                st.write(f"**Similitud promedio:** {grupo.get('similitud_promedio', 'N/A')}")
                                                
                                                st.write("**Textos en este grupo:**")
                                                for texto_item in grupo.get("textos", []):
                                                    st.write(f"- {texto_item.get('texto', 'N/A')}")
                                                    if "similitud" in texto_item:
                                                        st.write(f"  *Razón:* {texto_item['similitud']}")
                                    
                                    # Mostrar resumen
                                    if "resumen" in datos:
                                        st.subheader("📊 Resumen de Agrupamiento")
                                        res = datos["resumen"]
                                        col_res1, col_res2, col_res3 = st.columns(3)
                                        with col_res1:
                                            st.metric("Total grupos", res.get("total_grupos", 0))
                                            st.metric("Total textos", res.get("total_textos", 0))
                                        with col_res2:
                                            st.metric("Grupo más grande", res.get("grupo_mas_grande", "N/A"))
                                            st.metric("Grupo más pequeño", res.get("grupo_mas_pequeno", "N/A"))
                                        with col_res3:
                                            st.metric("Similitud promedio", res.get("similitud_promedio_total", "N/A"))
                                
                            except json.JSONDecodeError:
                                st.markdown("### Resultados de Agrupamiento")
                                st.text(resultado)
                    
                    else:  # Agrupamiento con K-means usando embeddings
                        
                        # Generar embeddings
                        embeddings = generar_embeddings(textos)
                        
                        if embeddings is not None:
                            # Aplicar K-means
                            kmeans = KMeans(n_clusters=n_grupos, random_state=42, n_init=10)
                            clusters = kmeans.fit_predict(embeddings)
                            
                            # Crear resultados
                            resultados = []
                            for i, (texto, cluster) in enumerate(zip(textos, clusters)):
                                # Calcular similitud con el centroide
                                centroide = kmeans.cluster_centers_[cluster]
                                distancia = np.linalg.norm(embeddings[i] - centroide)
                                similitud = max(0, 100 - distancia * 10)  # Convertir distancia a similitud aproximada
                                
                                resultados.append({
                                    "texto": texto[:100] + "..." if len(texto) > 100 else texto,
                                    "grupo": f"Grupo {cluster + 1}",
                                    "similitud_grupo": f"{similitud:.1f}%",
                                    "tamano_grupo": sum(clusters == cluster),
                                    "distancia_centroide": f"{distancia:.3f}"
                                })
                            
                            df_resultados = pd.DataFrame(resultados)
                            
                            st.subheader("📊 Clusters Encontrados (K-means con Embeddings)")
                            st.dataframe(df_resultados, use_container_width=True)
                            
                            # Mostrar estadísticas por grupo
                            st.write("**Distribución por grupo:**")
                            distribucion = df_resultados["grupo"].value_counts().sort_index()
                            
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # Gráfico de barras
                            distribucion.plot(kind='bar', ax=ax1, color='skyblue')
                            ax1.set_title("Cantidad de Textos por Grupo")
                            ax1.set_xlabel("Grupo")
                            ax1.set_ylabel("Cantidad")
                            
                            # Gráfico de pastel
                            ax2.pie(distribucion.values, labels=distribucion.index, autopct='%1.1f%%')
                            ax2.set_title("Proporción por Grupo")
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        else:
                            st.error("No se pudieron generar embeddings. Usando método alternativo...")
                            # Fallback a agrupamiento tradicional
                            resultados = agrupar_textos_manual(textos, n_grupos)
                            df_resultados = pd.DataFrame(resultados)
                            st.dataframe(df_resultados, use_container_width=True)
    
    # =========================================================
    # 5. REDACTOR DE CORREOS
    # =========================================================
    elif st.session_state.funcionalidad == "5":
        st.subheader("📧 Redactor de Correos Profesionales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tipo_correo = st.text_input(
                "Tipo de correo (personalizado):",
                "Seguimiento comercial",
                placeholder="Ej: Presentación, Propuesta, Agradecimiento, etc."
            )
            
            destinatario = st.text_input("Destinatario:", "cliente@empresa.com")
            
            tono = st.select_slider(
                "Tono:",
                options=["Muy formal", "Formal", "Neutral", "Amigable", "Muy amigable"],
                value="Formal"
            )
        
        with col2:
            longitud = st.selectbox(
                "Longitud:",
                ["Corto (3-5 líneas)", "Medio (1 párrafo)", "Largo (varios párrafos)"]
            )
            
            objetivo = st.text_area(
                "Objetivo/Contenido principal:",
                "Seguimiento de propuesta comercial enviada la semana pasada",
                height=100
            )
            
            info_adicional = st.text_area(
                "Información adicional (opcional):",
                "Incluir detalles del producto, beneficios, CTA",
                height=80
            )
        
        # Configuración avanzada
        with st.expander("⚙️ Configuración avanzada"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                incluir_asunto = st.checkbox("Generar asunto", value=True)
                incluir_firma = st.checkbox("Incluir firma", value=True)
                idioma = st.selectbox("Idioma:", ["Español", "Inglés", "Portugués"])
            
            with col_b:
                empresa = st.text_input("Nombre de la empresa (opcional):", "")
                remitente = st.text_input("Remitente:", "Equipo Comercial")
                urgencia = st.select_slider(
                    "Urgencia:",
                    options=["Baja", "Media", "Alta", "Urgente"],
                    value="Media"
                )
        
        # Botón para generar
        if st.button("✉️ Generar Correo", type="primary"):
            with st.spinner("Redactando correo..."):
                prompt = f"""
                Escribe un correo electrónico profesional con las siguientes características:
                
                - Tipo de correo: {tipo_correo}
                - Destinatario: {destinatario}
                - Tono: {tono}
                - Longitud: {longitud}
                - Objetivo principal: {objetivo}
                - Información adicional: {info_adicional}
                - Idioma: {idioma}
                - Urgencia: {urgencia}
                {f"- Empresa: {empresa}" if empresa else ""}
                - Remitente: {remitente}
                
                Instrucciones específicas:
                {"1. Incluye un asunto apropiado" if incluir_asunto else ""}
                2. Saludo profesional acorde al tono
                3. Cuerpo del mensaje claro y conciso
                4. Llamado a la acción (CTA) claro
                {"5. Despedida y firma" if incluir_firma else "5. Despedida apropiada"}
                
                El correo debe ser {tono.lower()} y efectivo para lograr el objetivo.
                {"Marca la urgencia de manera apropiada." if urgencia != "Media" else ""}
                """
                
                correo = llamar_ia(prompt, temperatura=0.2, max_tokens=1500)
                
                if correo:
                    st.success("✅ Correo generado exitosamente")
                    
                    # Mostrar en pestañas
                    tab1, tab2 = st.tabs(["📄 Correo Completo", "📋 Estructura"])
                    
                    with tab1:
                        st.markdown("### ✉️ Correo Generado")
                        st.markdown(correo)
                        
                        # Contador de palabras
                        palabras = len(correo.split())
                        st.caption(f"📝 {palabras} palabras")
                    
                    with tab2:
                        # Analizar estructura
                        st.write("**Estructura del correo:**")
                        lineas = correo.split('\n')
                        secciones = {
                            "Asunto": [],
                            "Saludo": [],
                            "Cuerpo": [],
                            "CTA": [],
                            "Despedida": [],
                            "Firma": []
                        }
                        
                        seccion_actual = "Asunto"
                        for i, linea in enumerate(lineas[:30]):  # Mostrar primeras 30 líneas
                            if linea.strip():
                                linea_limpia = linea.strip()
                                # Detectar cambios de sección
                                if "estimado" in linea_limpia.lower() or "hola" in linea_limpia.lower() or "querido" in linea_limpia.lower():
                                    seccion_actual = "Saludo"
                                elif "saludos" in linea_limpia.lower() or "atentamente" in linea_limpia.lower() or "cordiales" in linea_limpia.lower():
                                    seccion_actual = "Despedida"
                                elif "www." in linea_limpia.lower() or "@" in linea_limpia.lower() or "tel:" in linea_limpia.lower():
                                    seccion_actual = "Firma"
                                elif len(linea_limpia.split()) < 5 and i < 3:
                                    seccion_actual = "Asunto"
                                
                                if seccion_actual in secciones:
                                    secciones[seccion_actual].append(linea_limpia)
                                
                                st.write(f"{i+1}. [{seccion_actual}] {linea_limpia[:80]}...")
                        
                        # Mostrar resumen de secciones
                        st.write("**Resumen por sección:**")
                        for seccion, contenido in secciones.items():
                            if contenido:
                                st.write(f"**{seccion}:** {len(contenido)} líneas")
                    
                    # Opciones de descarga
                    st.download_button(
                        "📥 Descargar como .txt",
                        correo,
                        file_name=f"correo_{tipo_correo.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
    
    # =========================================================
    # 6. CONSTRUCTOR DE PROMPTS
    # =========================================================
    elif st.session_state.funcionalidad == "6":
        st.subheader("💡 Constructor de Prompts Profesionales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            idea_basica = st.text_area(
                "Describe tu idea básica:",
                "Quiero analizar datos de ventas del último trimestre",
                height=100
            )
            
            objetivo = st.selectbox(
                "Objetivo del prompt:",
                ["Análisis", "Generación", "Clasificación", "Extracción", 
                 "Resumen", "Traducción", "Edición", "Ideación", "Evaluación"]
            )
        
        with col2:
            contexto = st.text_area(
                "Contexto adicional (opcional):",
                "Los datos están en Excel, necesito insights para presentación ejecutiva",
                height=80
            )
            
            nivel_detalle = st.select_slider(
                "Nivel de detalle:",
                options=["Básico", "Intermedio", "Avanzado", "Experto"],
                value="Intermedio"
            )
        
        # Especificaciones avanzadas
        with st.expander("⚙️ Especificaciones avanzadas"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                formato_respuesta = st.selectbox(
                    "Formato de respuesta:",
                    ["Texto libre", "Lista", "Tabla", "JSON", "XML", "Markdown", "CSV"]
                )
                
                incluir_ejemplos = st.checkbox("Incluir ejemplos en el prompt", value=True)
                
                estilo_respuesta = st.selectbox(
                    "Estilo de respuesta:",
                    ["Técnico", "Ejecutivo", "Creativo", "Académico", "Periodístico"]
                )
            
            with col_b:
                restricciones = st.multiselect(
                    "Restricciones:",
                    ["Máximo 500 palabras", "Basado en hechos", "Sin opiniones personales",
                     "Citas de fuentes", "Estructurado", "Lenguaje técnico", "Lenguaje simple",
                     "Sin jerga", "Con estadísticas", "Con ejemplos prácticos"]
                )
                
                temperatura = st.slider("Temperatura sugerida:", 0.0, 1.0, 0.3, 0.1)
                
                considerar_audiencia = st.checkbox("Considerar audiencia específica", value=False)
                if considerar_audiencia:
                    audiencia_prompt = st.text_input("Audiencia:", "Ejecutivos senior")
        
        if st.button("🔨 Construir Prompt", type="primary"):
            with st.spinner("Construyendo prompt profesional..."):
                # Construir prompt
                prompt_construccion = f"""
                Basándote en esta idea básica, construye un prompt profesional para IA:
                
                IDEA BÁSICA: {idea_basica}
                OBJETIVO: {objetivo}
                CONTEXTO: {contexto}
                NIVEL DE DETALLE: {nivel_detalle}
                FORMATO RESPUESTA: {formato_respuesta}
                ESTILO RESPUESTA: {estilo_respuesta}
                {"AUDIENCIA: " + audiencia_prompt if considerar_audiencia else ""}
                RESTRICCIONES: {', '.join(restricciones) if restricciones else 'Ninguna'}
                
                El prompt debe ser:
                - Claro, específico y no ambiguo
                - Con instrucciones paso a paso bien estructuradas
                - {f"Incluir ejemplos relevantes del tipo: {formato_respuesta}" if incluir_ejemplos else ""}
                - Sugerir temperatura: {temperatura}
                - Adaptado para {provider}
                - Optimizado para obtener la mejor respuesta posible
                
                Devuelve SOLO el prompt profesional, sin explicaciones adicionales.
                El prompt debe estar listo para copiar y pegar en la IA.
                """
                
                prompt_generado = llamar_ia(prompt_construccion, temperatura=0.1, max_tokens=1500)
                
                if prompt_generado:
                    st.success("✅ Prompt construido exitosamente")
                    
                    # Mostrar el prompt
                    st.subheader("🎯 Prompt Profesional Generado")
                    st.code(prompt_generado, language="text")
                    
                    # Calcular estadísticas
                    palabras = len(prompt_generado.split())
                    lineas = prompt_generado.count('\n') + 1
                    
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    with col_stats1:
                        st.metric("Palabras", palabras)
                    with col_stats2:
                        st.metric("Líneas", lineas)
                    with col_stats3:
                        st.metric("Caracteres", len(prompt_generado))
                    
                    # Botón para copiar
                    st.download_button(
                        "📋 Copiar Prompt",
                        prompt_generado,
                        file_name=f"prompt_profesional_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
                    # Explicación del prompt
                    with st.expander("📖 Análisis del prompt generado"):
                        st.write("""
                        **Características de este prompt profesional:**
                        
                        1. **Claridad y especificidad**: Instrucciones no ambiguas
                        2. **Estructura lógica**: Organización paso a paso
                        3. **Contexto completo**: Incluye toda información necesaria
                        4. **Especificaciones técnicas**: Define formato y restricciones
                        5. **Optimización**: Adaptado para obtener mejor respuesta
                        6. **Ejemplos ilustrativos**: Ayudan a guiar la respuesta esperada
                        
                        **Recomendaciones de uso:**
                        - Copia y pega directamente en la IA
                        - Ajusta la temperatura según necesidad
                        - Revisa que todas las especificaciones sean relevantes
                        - Prueba con diferentes variaciones si es necesario
                        """)

    # Botón para volver al catálogo
    if st.button("↩️ Volver al catálogo"):
        st.session_state.funcionalidad = None
        st.session_state.nombre_func = None
        st.session_state.df = None
        st.session_state.datos_listos = False
        st.session_state.documento_texto = ""
        st.rerun()

# =========================================================
# INSTRUCCIONES INICIALES
# =========================================================
else:
    st.info("""
    ### 🎯 6 Funcionalidades Disponibles:
    
    **Análisis de Texto:**
    1. 📝 Análisis de Sentimiento
    2. 🔍 Extracción de Información  
    3. ⚠️ Clasificación de Textos
    4. 📊 Agrupamiento de Textos
    
    **Generación y Redacción:**
    5. ✉️ Redactor de Correos
    6. 💡 Constructor de Prompts
    
    ### 📋 Cómo usar:
    1. Configura tu API Key
    2. Selecciona una funcionalidad
    3. Sigue las instrucciones
    4. Obtén resultados inmediatos
    """)

# =========================================================
# PIE DE PÁGINA
# =========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Información")

if st.session_state.get('funcionalidad'):
    func_info = funcionalidades[st.session_state.funcionalidad]
    st.sidebar.info(f"""
    **Funcionalidad:**
    {func_info['nombre']}
    
    **Proveedor:**
    {provider}
    """)

st.sidebar.markdown("---")
st.sidebar.caption("🤖 Analytics Assistant Pro v3.0")

if st.sidebar.button("🔄 Resetear Todo"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()