import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, color, exposure, img_as_float, img_as_ubyte, transform
from scipy.ndimage import uniform_filter, gaussian_filter, median_filter
import os

# Configuración de la página
st.set_page_config(page_title="Procesamiento de Imágenes", layout="wide")
st.title("Herramienta de Procesamiento de Imágenes")

# Función para cargar imagen
def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    return img_as_float(np.array(img))

# Función para mostrar histograma con ajuste de canales
def plot_histogram(image, title="Histograma", r_factor=1.0, g_factor=1.0, b_factor=1.0):
    # Aplicar factores a los canales
    adjusted_image = image.copy()
    adjusted_image[..., 0] = np.clip(adjusted_image[..., 0] * r_factor, 0, 1)
    adjusted_image[..., 1] = np.clip(adjusted_image[..., 1] * g_factor, 0, 1)
    adjusted_image[..., 2] = np.clip(adjusted_image[..., 2] * b_factor, 0, 1)
    
    colors = ('red', 'green', 'blue')
    plt.figure(figsize=(6, 4))
    for i, color_name in enumerate(colors):
        hist, bins = np.histogram(adjusted_image[..., i], bins=256, range=(0, 1))
        plt.plot(bins[:-1], hist, color=color_name, label=color_name.upper())
    plt.title(title)
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(True)
    return plt, adjusted_image

# Función para ecualización corregida
def equalize_image(image):
    img_yuv = color.rgb2yuv(image)
    img_yuv[:,:,0] = exposure.equalize_hist(img_yuv[:,:,0])
    img_eq = color.yuv2rgb(img_yuv)
    # Aseguramos que los valores estén en [0, 1]
    return np.clip(img_eq, 0, 1)

# Función para rotación
def rotate_image(image, angle):
    return transform.rotate(image, angle, resize=True)

# Función para traslación
def translate_image(image, tx, ty):
    tform = transform.AffineTransform(translation=(tx, ty))
    return transform.warp(image, tform.inverse)

# Interfaz de usuario
uploaded_file = st.file_uploader("Cargar imagen", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Mostrar imagen original
    img = load_image(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Imagen Original", use_container_width=True)
        # Mostrar dimensiones de la imagen
        st.write(f"Dimensiones originales: {img.shape[1]}x{img.shape[0]} píxeles")
    
    with col2:
        # Histograma inicial sin ajustes
        st.pyplot(plot_histogram(img)[0])
    
    # Menú de operaciones
    operation = st.sidebar.selectbox(
        "Seleccione una operación",
        ["Reescalado", "Histograma (ajuste canales)", "Ecualización", "Filtros Lineales", 
         "Filtros No Lineales", "Traslación", "Rotación"]
    )
    
    if operation == "Reescalado":
        col1, col2 = st.columns(2)
        with col1:
            scale_option = st.radio("Método de escalado", ["Por factor", "Por dimensiones"])
            
            if scale_option == "Por factor":
                scale_factor = st.slider("Factor de escala", 0.1, 3.0, 1.0, 0.1)
                rescaled_img = transform.rescale(img, scale=scale_factor, channel_axis=2, anti_aliasing=True)
            else:
                new_width = st.slider("Ancho deseado (píxeles)", 10, 2000, img.shape[1])
                scale_factor = new_width / img.shape[1]
                rescaled_img = transform.resize(img, (int(img.shape[0] * scale_factor), new_width), anti_aliasing=True)
            
            st.image(rescaled_img, caption=f"Imagen reescalada - Nuevo tamaño: {rescaled_img.shape[1]}x{rescaled_img.shape[0]}", 
                    use_container_width=True)
        
        with col2:
            st.pyplot(plot_histogram(rescaled_img)[0])
    
    elif operation == "Histograma (ajuste canales)":
        st.subheader("Ajuste de canales de color")
        
        col1, col2 = st.columns(2)
        with col1:
            r_factor = st.slider("Factor Rojo (R)", 0.0, 2.0, 1.0, 0.1)
            g_factor = st.slider("Factor Verde (G)", 0.0, 2.0, 1.0, 0.1)
            b_factor = st.slider("Factor Azul (B)", 0.0, 2.0, 1.0, 0.1)
            
            hist_plot, adjusted_img = plot_histogram(img, "Histograma Ajustado", r_factor, g_factor, b_factor)
            st.pyplot(hist_plot)
        
        with col2:
            st.image(adjusted_img, caption="Imagen con canales ajustados", use_container_width=True)
    
    elif operation == "Ecualización":
        img_eq = equalize_image(img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_eq, caption="Imagen Ecualizada", use_container_width=True)
        with col2:
            st.pyplot(plot_histogram(img_eq, "Histograma Ecualizado")[0])
    
    elif operation == "Filtros Lineales":
        filter_type = st.sidebar.radio("Tipo de filtro", ["Media", "Gaussiano"])
        
        if filter_type == "Media":
            size = st.slider("Tamaño del kernel (media)", 3, 15, 3, 2)
            filtered_img = uniform_filter(img, size=size)
            st.image(filtered_img, caption=f"Filtro de Media ({size}x{size})", use_container_width=True)
        else:
            sigma = st.slider("Sigma (gaussiano)", 0.1, 5.0, 1.0, 0.1)
            filtered_img = gaussian_filter(img, sigma=sigma)
            st.image(filtered_img, caption=f"Filtro Gaussiano (σ={sigma})", use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_histogram(filtered_img)[0])
    
    elif operation == "Filtros No Lineales":
        size = st.slider("Tamaño del kernel (mediana)", 3, 15, 3, 2)
        filtered_img = median_filter(img, size=size)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(filtered_img, caption=f"Filtro Mediana ({size}x{size})", use_container_width=True)
        with col2:
            st.pyplot(plot_histogram(filtered_img)[0])
    
    elif operation == "Traslación":
        col1, col2 = st.columns(2)
        with col1:
            tx = st.slider("Traslación X (píxeles)", -500, 500, 0)
            ty = st.slider("Traslación Y (píxeles)", -500, 500, 0)
        translated_img = translate_image(img, tx, ty)
        st.image(translated_img, caption=f"Imagen Trasladada (X={tx}, Y={ty})", use_container_width=True)
    
    elif operation == "Rotación":
        angle = st.slider("Ángulo de rotación (grados)", -180, 180, 0)
        rotated_img = rotate_image(img, angle)
        st.image(rotated_img, caption=f"Imagen Rotada ({angle}°)", use_container_width=True)
        st.write(f"Nuevas dimensiones: {rotated_img.shape[1]}x{rotated_img.shape[0]} píxeles")
    
    # Guardar imagen procesada
    if st.button("💾 Guardar imagen procesada"):
        processed_img = None
        
        if operation == "Reescalado":
            processed_img = (rescaled_img * 255).astype(np.uint8)
        elif operation == "Histograma (ajuste canales)":
            processed_img = (adjusted_img * 255).astype(np.uint8)
        elif operation == "Ecualización":
            processed_img = (img_eq * 255).astype(np.uint8)
        elif operation == "Filtros Lineales":
            processed_img = (filtered_img * 255).astype(np.uint8)
        elif operation == "Filtros No Lineales":
            processed_img = (filtered_img * 255).astype(np.uint8)
        elif operation == "Traslación":
            processed_img = (translated_img * 255).astype(np.uint8)
        elif operation == "Rotación":
            processed_img = (rotated_img * 255).astype(np.uint8)
        
        if processed_img is not None:
            img_pil = Image.fromarray(processed_img)
            img_pil.save("imagen_procesada.png")
            st.success("✅ Imagen guardada como 'imagen_procesada.png'")
            with open("imagen_procesada.png", "rb") as file:
                st.download_button(
                    label="⬇️ Descargar imagen",
                    data=file,
                    file_name="imagen_procesada.png",
                    mime="image/png"
                )
        else:
            st.warning("⚠️ No hay imagen procesada para guardar")
else:
    st.info("ℹ️ Por favor, suba una imagen para comenzar el procesamiento")