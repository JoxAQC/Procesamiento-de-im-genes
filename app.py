import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(page_title="Análisis de Sentimientos IMDB", page_icon="🎬", layout="wide")
st.title("🎬 Analizador de Reviews de Películas (IMDB/Kaggle)")

# Inicializar variables en session_state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'modelos_entrenados' not in st.session_state:
    st.session_state.modelos_entrenados = False

# Función para cargar datos con cache
@st.cache_data
def cargar_datos(archivo, sample_size=5000):
    try:
        df = pd.read_csv(archivo)
        
        # Mapear columnas comunes de datasets de reviews
        review_col = next((col for col in df.columns if 'review' in col.lower() or 'text' in col.lower()), None)
        sentiment_col = next((col for col in df.columns if 'sentiment' in col.lower() or 'label' in col.lower() or 'rating' in col.lower()), None)
        
        if not review_col or not sentiment_col:
            st.error("No se encontraron columnas de review o sentimiento. Las columnas disponibles son: " + ", ".join(df.columns))
            return None
        
        # Limitar el tamaño del dataset para pruebas rápidas
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)
            st.warning(f"⚠️ Se ha tomado una muestra aleatoria de {sample_size} registros para agilizar el procesamiento")
        
        # Normalizar sentimientos
        df = df.rename(columns={review_col: 'review', sentiment_col: 'sentiment'})
        
        if df['sentiment'].dtype == 'object':
            df['sentiment'] = df['sentiment'].map({
                'positive': 1, 'Positive': 1, 'positivo': 1, 'Positivo': 1, '1': 1,
                'negative': 0, 'Negative': 0, 'negativo': 0, 'Negativo': 0, '0': 0,
                'pos': 1, 'neg': 0
            })
        elif df['sentiment'].dtype == 'int':
            # Si son ratings (1-10), convertir a binario (1 si >=7, 0 si <=4)
            if df['sentiment'].max() > 1:
                df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x >= 7 else (0 if x <= 4 else np.nan))
                df = df.dropna(subset=['sentiment'])
        
        return df[['review', 'sentiment']]
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None

# Sidebar - Carga de archivo
with st.sidebar:
    st.header("Configuración")
    archivo = st.file_uploader("📂 Sube tu archivo CSV (IMDB/Kaggle)", type=["csv"])
    
    if archivo is not None:
        with st.spinner("Cargando y muestreando archivo..."):
            st.session_state.df = cargar_datos(archivo, sample_size=3000)  # Muestra más pequeña para pruebas
    
    if st.session_state.df is not None:
        st.subheader("Opciones Rápidas")
        test_size = st.slider("Tamaño del conjunto de prueba (%):", 10, 40, 20)  # % más pequeño
        max_features = st.slider("Máximo de características TF-IDF:", 500, 3000, 1000, step=500)  # Menos features
        
        if st.button("⚡ Entrenar Modelo"):
            with st.spinner("Entrenando modelos con configuración rápida..."):
                try:
                    df = st.session_state.df.copy()
                    X = df['review']
                    y = df['sentiment']
                    
                    # Muestra más pequeña para entrenamiento rápido
                    if len(X) > 2000:
                        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=2000, random_state=42, stratify=y)
                    else:
                        X_sample, y_sample = X, y
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_sample, y_sample, test_size=test_size/100, random_state=42)
                    
                    # Vectorización más rápida
                    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 1))
                    X_train_tfidf = tfidf.fit_transform(X_train)
                    X_test_tfidf = tfidf.transform(X_test)
                    
                    # Modelo Naive Bayes (rápido)
                    nb_model = MultinomialNB()
                    nb_model.fit(X_train_tfidf, y_train)
                    y_pred_nb = nb_model.predict(X_test_tfidf)
                    
                    # Modelo Regresión Logística (configuración rápida)
                    lr_model = LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)
                    lr_model.fit(X_train_tfidf, y_train)
                    y_pred_lr = lr_model.predict(X_test_tfidf)
                    
                    # Modelo TextBlob (muestra más pequeña para velocidad)
                    def get_sentiment_textblob(text):
                        analysis = TextBlob(text)
                        return 1 if analysis.sentiment.polarity > 0 else 0
                    
                    # Usar solo 100 ejemplos para TextBlob (es lento)
                    sample_size_tb = min(100, len(X_test))
                    y_pred_tb = X_test.sample(sample_size_tb, random_state=42).apply(get_sentiment_textblob)
                    y_test_tb = y_test[X_test.sample(sample_size_tb, random_state=42).index]
                    
                    st.session_state.modelos_entrenados = True
                    st.session_state.resultados = {
                        'X_test': X_test,
                        'y_test': y_test,
                        'y_test_tb': y_test_tb,  # Solo para TextBlob
                        'y_pred_nb': y_pred_nb,
                        'y_pred_lr': y_pred_lr,
                        'y_pred_tb': y_pred_tb,
                        'get_sentiment_textblob': get_sentiment_textblob,
                        'nb_model': nb_model,
                        'lr_model': lr_model,
                        'tfidf': tfidf
                    }
                    
                    st.success(f"✅ Modelos entrenados! (Muestra: {len(X_train)} train, {len(X_test)} test)")
                except Exception as e:
                    st.error(f"Error durante el entrenamiento: {e}")

# Mostrar dataset si está cargado
if st.session_state.df is not None:
    st.subheader("📝 Dataset Cargado (Muestra)")
    st.write(f"Total de registros: {len(st.session_state.df)}")
    st.dataframe(st.session_state.df.head(3))  # Mostrar solo las primeras filas
    
    st.subheader("📈 Análisis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Distribución de Sentimientos")
        fig, ax = plt.subplots(figsize=(6, 4))
        st.session_state.df['sentiment'].value_counts().plot(kind='bar', ax=ax)
        ax.set_xticklabels(['Negativo', 'Positivo'], rotation=0)
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Longitud de las Reseñas")
        st.session_state.df['length'] = st.session_state.df['review'].str.len()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=st.session_state.df, x='sentiment', y='length', ax=ax)
        ax.set_xticklabels(['Negativo', 'Positivo'])
        st.pyplot(fig)

# Mostrar resultados si los modelos están entrenados
if st.session_state.modelos_entrenados:
    resultados = st.session_state.resultados
    
    st.subheader("⚡ Resultados")
    tab1, tab2, tab3 = st.tabs(["Naive Bayes", "Regresión Logística", "TextBlob"])
    
    with tab1:
        st.markdown("### Naive Bayes")
        st.write(f"**Accuracy:** {accuracy_score(resultados['y_test'], resultados['y_pred_nb']):.2f}")
        st.text(classification_report(resultados['y_test'], resultados['y_pred_nb'], 
               target_names=['Negativo', 'Positivo']))
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(confusion_matrix(resultados['y_test'], resultados['y_pred_nb']), 
                    annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xticklabels(['Negativo', 'Positivo'])
        ax.set_yticklabels(['Negativo', 'Positivo'])
        st.pyplot(fig)
    
    with tab2:
        st.markdown("### Regresión Logística")
        st.write(f"**Accuracy:** {accuracy_score(resultados['y_test'], resultados['y_pred_lr']):.2f}")
        st.text(classification_report(resultados['y_test'], resultados['y_pred_lr'], 
               target_names=['Negativo', 'Positivo']))
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(confusion_matrix(resultados['y_test'], resultados['y_pred_lr']), 
                    annot=True, fmt='d', cmap='Reds', ax=ax)
        ax.set_xticklabels(['Negativo', 'Positivo'])
        ax.set_yticklabels(['Negativo', 'Positivo'])
        st.pyplot(fig)
    
    with tab3:
        st.markdown("### TextBlob (Muestra pequeña)")
        st.write(f"**Accuracy:** {accuracy_score(resultados['y_test_tb'], resultados['y_pred_tb']):.2f} (en {len(resultados['y_pred_tb'])} samples)")
        st.text(classification_report(resultados['y_test_tb'], resultados['y_pred_tb'], 
               target_names=['Negativo', 'Positivo']))
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(confusion_matrix(resultados['y_test_tb'], resultados['y_pred_tb']), 
                    annot=True, fmt='d', cmap='Greens', ax=ax)
        ax.set_xticklabels(['Negativo', 'Positivo'])
        ax.set_yticklabels(['Negativo', 'Positivo'])
        st.pyplot(fig)

    st.subheader("🔍 Probar con Nuevo Texto")
    with st.form("analizar_texto"):
        nuevo_texto = st.text_area("Escribe una review de película:", 
                                 "The movie was great! The acting was superb and the plot kept me engaged throughout.")
        submitted = st.form_submit_button("Predecir Sentimiento")
        
        if submitted:
            if nuevo_texto.strip() == "":
                st.warning("⚠️ Ingresa un texto válido para analizar.")
            else:
                st.markdown("### Resultados")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    nb_pred = resultados['nb_model'].predict(resultados['tfidf'].transform([nuevo_texto]))[0]
                    st.metric("Naive Bayes", "Positivo" if nb_pred == 1 else "Negativo")
                
                with col2:
                    lr_pred = resultados['lr_model'].predict(resultados['tfidf'].transform([nuevo_texto]))[0]
                    st.metric("Regresión Logística", "Positivo" if lr_pred == 1 else "Negativo")
                
                with col3:
                    tb_pred = resultados['get_sentiment_textblob'](nuevo_texto)
                    st.metric("TextBlob", "Positivo" if tb_pred == 1 else "Negativo")
                
                st.markdown("### Palabras Clave")
                wordcloud = WordCloud(width=600, height=300, max_words=50).generate(nuevo_texto)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

elif st.session_state.df is None:
    st.info("""
    ℹ️ Instrucciones:
    1. Sube un CSV con reviews de películas (ej. de IMDB o Kaggle)
    2. El archivo debe tener columnas para el texto (review/text) y sentimiento (sentiment/label/rating)
    3. Usaremos una muestra pequeña para análisis rápido
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Q8J27fD0d4Q3QZOSrXKvJg.png", width=300)
