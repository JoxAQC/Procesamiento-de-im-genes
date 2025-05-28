def evaluate_configuration(config, workload="Media"):
    # Validar inputs
    cpu = max(0.1, config['cpu'])
    memory = max(0.1, config['memory'])
    replicas = max(1, config['replicas'])
    
    # Costos (asegurar positivos)
    cpu_cost = max(0, 0.024 * cpu * 1000)  # $24/core/mes
    memory_cost = max(0, 0.008 * memory * 1000)  # $8/GB/mes
    replicas_cost = max(0, 0.001 * replicas * 1000)  # $1/réplica/mes
    
    total_cost = max(0, (cpu_cost + memory_cost) * replicas + replicas_cost)
    
    # Performance (asegurar entre 0% y 100%)
    required_cpu = 10 * replicas if workload == "Media" else (5 * replicas if workload == "Baja" else 20 * replicas)
    required_memory = 20 * replicas if workload == "Media" else (10 * replicas if workload == "Baja" else 40 * replicas)
    
    cpu_performance = min(1.0, max(0, (cpu * replicas) / required_cpu))
    memory_performance = min(1.0, max(0, (memory * replicas) / required_memory))
    performance = min(1.0, max(0, (cpu_performance + memory_performance) / 2))  # Promedio seguro
    
    return {
        'cost': total_cost,
        'performance': performance,
        'cpu_usage': cpu_performance,
        'memory_usage': memory_performance
    }