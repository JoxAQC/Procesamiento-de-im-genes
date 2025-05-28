import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from utils.genetic_algorithm import run_genetic_algorithm
from utils.kubernetes_simulator import evaluate_configuration

st.set_page_config(page_title="Optimizador de Contenedores", layout="wide")

st.title("Optimización de Configuraciones de Contenedores")
st.markdown("""
Esta aplicación utiliza un Algoritmo Genético para encontrar la mejor configuración de recursos 
para tus contenedores en Kubernetes, balanceando costo y performance.
""")

# Sidebar con parámetros
with st.sidebar:
    st.header("Parámetros del Algoritmo Genético")
    population_size = st.slider("Tamaño de población", 10, 100, 30)
    generations = st.slider("Número de generaciones", 5, 100, 20)
    crossover_prob = st.slider("Probabilidad de crossover", 0.1, 1.0, 0.7)
    mutation_prob = st.slider("Probabilidad de mutación", 0.01, 0.5, 0.1)
    
    st.header("Restricciones del SLA")
    # Sidebar con parámetros más realistas para gran escala
    min_cpu = st.slider("CPU mínima (cores)", 1.0, 64.0, 8.0)  # Default: 8 cores
    max_cpu = st.slider("CPU máxima (cores)", 1.0, 64.0, 32.0)  # Default: 32 cores
    min_memory = st.slider("Memoria mínima (GB)", 1.0, 128.0, 16.0)  # Default: 16GB
    max_memory = st.slider("Memoria máxima (GB)", 1.0, 128.0, 64.0)  # Default: 64GB
    min_replicas = st.slider("Mínimo de réplicas", 1, 100, 10)  # Default: 10
    max_replicas = st.slider("Máximo de réplicas", 1, 100, 50)  # Default: 50
    
    workload = st.selectbox("Escenario de carga", ["Baja", "Media", "Alta"])

# Ejecutar optimización
if st.button("Ejecutar Optimización"):
    with st.spinner("Optimizando configuraciones..."):
        best_config, history = run_genetic_algorithm(
            population_size=population_size,
            generations=generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            cpu_bounds=(min_cpu, max_cpu),
            memory_bounds=(min_memory, max_memory),
            replicas_bounds=(min_replicas, max_replicas),
            workload=workload
        )
    
    st.success("¡Optimización completada!")
    
    # Mostrar mejor configuración
    st.subheader("Mejor configuración encontrada")
    col1, col2, col3 = st.columns(3)
    col1.metric("CPU (cores)", f"{best_config['cpu']:.1f}")
    col2.metric("Memoria (GB)", f"{best_config['memory']:.1f}")
    col3.metric("Réplicas", best_config['replicas'])

    # Gráfico de evolución por generación
    st.subheader("Evolución del Algoritmo Genético")
    df_history = pd.DataFrame(history)  # 'history' ya lo devuelve run_genetic_algorithm()

    # Gráfico de fitness (adaptación) por generación
    fig_fitness = px.line(
        df_history, 
        x='generation', 
        y=['best_fitness', 'avg_fitness', 'worst_fitness'],
        labels={'value': 'Fitness (menor es mejor)', 'generation': 'Generación'},
        title="Evolución del Fitness por Generación"
    )
    st.plotly_chart(fig_fitness, use_container_width=True)

    # Tabla con datos de cada generación
    st.write("Detalles por generación:")
    st.dataframe(df_history)
    
    # Mostrar métricas
    cost = evaluate_configuration(best_config)['cost']
    performance = evaluate_configuration(best_config)['performance']
    col1.metric("Costo estimado", f"${cost:,.1f}K/mes") 
    col2.metric("Performance", f"{performance*100:.1f}%")
    
    # Gráfico de evolución
    st.subheader("Evolución del Fitness")
    df_history = pd.DataFrame(history)
    fig = px.line(df_history, x='generation', y='best_fitness', 
                  title="Mejor Fitness por Generación")
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparación de configuraciones
    st.subheader("Comparación de Configuraciones")
    
    # Configuración aleatoria para comparar
    random_config = {
        'cpu': np.random.uniform(min_cpu, max_cpu),
        'memory': np.random.uniform(min_memory, max_memory),
        'replicas': np.random.randint(min_replicas, max_replicas+1)
    }
    
    # Evaluar ambas configuraciones
    best_eval = evaluate_configuration(best_config)
    random_eval = evaluate_configuration(random_config)
    
    # Crear DataFrame para comparación
    compare_df = pd.DataFrame({
        'Configuración': ['Optimizada', 'Aleatoria'],
        'Costo': [best_eval['cost'], random_eval['cost']],
        'Performance': [best_eval['performance'], random_eval['performance']],
        'CPU': [best_config['cpu'], random_config['cpu']],
        'Memoria': [best_config['memory'], random_config['memory']],
        'Réplicas': [best_config['replicas'], random_config['replicas']]
    })
    
    st.dataframe(compare_df)
    
    # Gráfico de comparación
    fig = px.bar(compare_df, x='Configuración', y=['Costo', 'Performance'],
                 barmode='group', title="Comparación: Optimizada vs Aleatoria")
    st.plotly_chart(fig, use_container_width=True)

# Sección de explicación
with st.expander("¿Cómo funciona esta optimización?"):
    st.markdown("""
    ### Simulador de Kubernetes
    - Simulamos diferentes configuraciones de recursos para contenedores
    - Parámetros ajustables: CPU, memoria y número de réplicas
    - Evaluamos cada configuración basada en costo y performance
    
    ### Algoritmo Genético
    - **Población inicial**: Configuraciones aleatorias dentro de los límites
    - **Selección**: Mantenemos las configuraciones con mejor fitness
    - **Crossover**: Combinamos configuraciones prometedoras
    - **Mutación**: Pequeños cambios aleatorios para exploración
    - **Evaluación**: Calculamos fitness basado en costo y SLA
    
    ### Función de Fitness
    - Minimizar: `Costo total (CPU + Memoria + Réplicas)`
    - Maximizar: `Performance (cumplimiento del SLA)`
    - Penalizamos configuraciones que no cumplen con los requisitos mínimos
    """)