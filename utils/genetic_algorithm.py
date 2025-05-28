import random
import numpy as np
from deap import base, creator, tools, algorithms
from .kubernetes_simulator import evaluate_configuration 

import sys
if sys.version_info >= (3, 13):
    print("⚠️ Advertencia: Python 3.13 puede causar problemas")

def evaluate_individual(individual, workload):
    """Evalúa una configuración individual"""
    # Asegurar valores positivos y dentro de rangos realistas
    cpu = max(0.1, individual[0])  # Mínimo 0.1 core
    memory = max(0.1, individual[1])  # Mínimo 0.1 GB
    replicas = max(1, int(individual[2]))  # Mínimo 1 réplica
    
    config = {
        'cpu': cpu,
        'memory': memory,
        'replicas': replicas
    }
    
    result = evaluate_configuration(config, workload)
    fitness = max(0.1, result['cost'] * (1.1 - result['performance']))  # Fitness mínimo 0.1
    return (fitness,)

def run_genetic_algorithm(population_size=30, generations=20, 
                         crossover_prob=0.7, mutation_prob=0.1,
                         cpu_bounds=(0.1, 2.0), memory_bounds=(0.1, 4.0),
                         replicas_bounds=(1, 5), workload="Media"):
    """Ejecuta el algoritmo genético y devuelve la mejor configuración"""
    
    # Configurar DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Definir atributos
    toolbox.register("attr_cpu", random.uniform, cpu_bounds[0], cpu_bounds[1])
    toolbox.register("attr_memory", random.uniform, memory_bounds[0], memory_bounds[1])
    toolbox.register("attr_replicas", random.randint, replicas_bounds[0], replicas_bounds[1])
    
    # Crear individuo y población
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.attr_cpu, toolbox.attr_memory, toolbox.attr_replicas), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Operadores genéticos
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_individual, workload=workload)
    
    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Crear población inicial
    pop = toolbox.population(n=population_size)
    
    # Ejecutar algoritmo
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob,
        ngen=generations, stats=stats, verbose=False
    )
    
    # Obtener mejor individuo
    best_individual = tools.selBest(pop, k=1)[0]
    best_config = {
        'cpu': best_individual[0],
        'memory': best_individual[1],
        'replicas': int(best_individual[2])
    }
    
    # Preparar historial para visualización
    history = []
    for gen, record in enumerate(logbook):
        history.append({
            'generation': gen,
            'best_fitness': record['min'],
            'avg_fitness': record['avg'],
            'worst_fitness': record['max'],
            'best_cpu': tools.selBest(pop, k=1)[0][0],
            'best_memory': tools.selBest(pop, k=1)[0][1],
            'best_replicas': int(tools.selBest(pop, k=1)[0][2])
        })
    
    return best_config, history
