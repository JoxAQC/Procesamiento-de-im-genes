import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(page_title="Análisis de Sentimientos", page_icon="📊", layout="wide")
st.title("📊 Análisis de Sentimientos con 2 Modelos NLP")

# Inicializar variables en session_state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'modelos_entrenados' not in st.session_state:
    st.session_state.modelos_entrenados = False

# Función para cargar datos con cache
@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo)
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            st.error("El archivo debe contener columnas llamadas 'review' y 'sentiment'")
            return None
        if df['sentiment'].dtype == 'object':
            df['sentiment'] = df['sentiment'].map({
                'positive': 1, 'Positive': 1, 'positivo': 1, 'Positivo': 1,
                'negative': 0, 'Negative': 0, 'negativo': 0, 'Negativo': 0
            })
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None

# Sidebar - Carga de archivo
with st.sidebar:
    st.header("Configuración")
    archivo = st.file_uploader("📂 Sube tu archivo CSV", type=["csv"])
    
    if archivo is not None:
        with st.spinner("Cargando archivo..."):
            st.session_state.df = cargar_datos(archivo)
    
    if st.session_state.df is not None:
        st.subheader("Parámetros del Modelo")
        test_size = st.slider("Tamaño del conjunto de prueba (%):", 10, 40, 30)
        max_features = st.slider("Máximo de características TF-IDF:", 500, 5000, 2000, step=500)
        
        if st.button("🚀 Entrenar Modelos"):
            with st.spinner("Entrenando modelos..."):
                try:
                    df = st.session_state.df.copy()
                    X = df['review']
                    y = df['sentiment']
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42)
                    
                    tfidf = TfidfVectorizer(max_features=max_features)
                    X_train_tfidf = tfidf.fit_transform(X_train)
                    X_test_tfidf = tfidf.transform(X_test)
                    
                    nb_model = MultinomialNB()
                    nb_model.fit(X_train_tfidf, y_train)
                    y_pred_nb = nb_model.predict(X_test_tfidf)
                    
                    def get_sentiment_textblob(text):
                        analysis = TextBlob(text)
                        return 1 if analysis.sentiment.polarity > 0 else 0
                    
                    y_pred_tb = X_test.apply(get_sentiment_textblob)
                    
                    st.session_state.modelos_entrenados = True
                    st.session_state.resultados = {
                        'X_test': X_test,
                        'y_test': y_test,
                        'y_pred_nb': y_pred_nb,
                        'y_pred_tb': y_pred_tb,
                        'get_sentiment_textblob': get_sentiment_textblob,
                        'nb_model': nb_model,
                        'tfidf': tfidf
                    }
                    
                    st.success("✅ Modelos entrenados exitosamente!")
                except Exception as e:
                    st.error(f"Error durante el entrenamiento: {e}")

# Mostrar dataset si está cargado
if st.session_state.df is not None:
    st.subheader("📝 Dataset Cargado")
    st.dataframe(st.session_state.df)
    
    st.subheader("📈 Análisis Exploratorio")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Distribución de Sentimientos")
        fig, ax = plt.subplots(figsize=(8, 5))
        st.session_state.df['sentiment'].value_counts().plot(kind='bar', ax=ax)
        ax.set_xticklabels(['Negativo', 'Positivo'], rotation=0)
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Longitud de las Reseñas")
        if 'length' not in st.session_state.df.columns:
            st.session_state.df['length'] = st.session_state.df['review'].apply(len)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=st.session_state.df, x='length', hue='sentiment', bins=30, ax=ax)
        st.pyplot(fig)

# Mostrar resultados si los modelos están entrenados
if st.session_state.modelos_entrenados:
    resultados = st.session_state.resultados
    
    st.subheader("📊 Resultados de los Modelos")
    tab1, tab2 = st.tabs(["Naive Bayes", "TextBlob"])
    
    with tab1:
        st.markdown("### Naive Bayes con TF-IDF")
        st.write(f"**Accuracy:** {accuracy_score(resultados['y_test'], resultados['y_pred_nb']):.4f}")
        st.text(classification_report(resultados['y_test'], resultados['y_pred_nb'], 
               target_names=['Negativo', 'Positivo']))
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(confusion_matrix(resultados['y_test'], resultados['y_pred_nb']), 
                    annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negativo', 'Positivo'], 
                    yticklabels=['Negativo', 'Positivo'])
        plt.title('Matriz de Confusión - Naive Bayes')
        st.pyplot(fig)
    
    with tab2:
        st.markdown("### TextBlob (Análisis basado en reglas)")
        st.write(f"**Accuracy:** {accuracy_score(resultados['y_test'], resultados['y_pred_tb']):.4f}")
        st.text(classification_report(resultados['y_test'], resultados['y_pred_tb'], 
               target_names=['Negativo', 'Positivo']))
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(confusion_matrix(resultados['y_test'], resultados['y_pred_tb']), 
                    annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Negativo', 'Positivo'], 
                    yticklabels=['Negativo', 'Positivo'])
        plt.title('Matriz de Confusión - TextBlob')
        st.pyplot(fig)

    st.subheader("🔮 Analizar Nuevo Texto")
    with st.form("analizar_texto"):
        nuevo_texto = st.text_area("Ingresa el texto a analizar:", 
                                 "This movie was fantastic! The acting was great.")
        submitted = st.form_submit_button("Predecir Sentimiento")
        
        if submitted:
            if nuevo_texto.strip() == "":
                st.warning("⚠️ Ingresa un texto válido para analizar.")
            else:
                st.markdown("### Resultados del Análisis")
                nb_pred = resultados['nb_model'].predict(
                    resultados['tfidf'].transform([nuevo_texto]))[0]
                st.write(f"**Naive Bayes:** {'Positivo' if nb_pred == 1 else 'Negativo'}")
                tb_pred = resultados['get_sentiment_textblob'](nuevo_texto)
                st.write(f"**TextBlob:** {'Positivo' if tb_pred == 1 else 'Negativo'}")
                
                st.markdown("### Nube de Palabras del Texto")
                wordcloud = WordCloud(width=800, height=400, max_words=200).generate(nuevo_texto)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

# Mensaje si no hay datos cargados
elif st.session_state.df is None:
    st.info("ℹ️ Por favor, carga un archivo CSV con columnas 'review' y 'sentiment' para comenzar.")
