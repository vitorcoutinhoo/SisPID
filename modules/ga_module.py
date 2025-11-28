# pylint: disable="C0114, C0103, R0914, C0301"

import numpy as np
from db.db_module import salvar_historico_evolutivo
from model.model import simulate, model

def fitness_ga(solution, plant, t, setpoint):
    Kp, Ki, Kd = solution
    try:
        _, Yresp = simulate(plant, Kp, Ki, Kd, t, setpoint)
        mse = np.mean((Yresp - setpoint) ** 2)
    except Exception:
        mse = 1e6
    return -mse  # negativo porque queremos minimizar o erro


def tune_pid_ga(plant=None, t=None, setpoint=1.0, 
                generations=50, population_size=20,
                db_path="db/pid_results.db"):
    if plant is None:
        plant = model(59.81, 401.61)
    if t is None:
        t = np.linspace(0, 2000, 1000)

    # Inicialização da população
    pop = np.column_stack([
        np.random.uniform(0, 20, population_size),  # Kp
        np.random.uniform(0, 2, population_size),   # Ki
        np.random.uniform(0, 5, population_size)    # Kd
    ])

    for gen in range(generations):
        # Avaliação da população
        fitness_vals = np.array([fitness_ga(ind, plant, t, setpoint) for ind in pop])
        
        # Converte para MSE (valores positivos)
        mse_vals = -fitness_vals
        
        # Salva histórico da geração
        salvar_historico_evolutivo("GA", gen + 1, np.min(mse_vals), np.mean(mse_vals), np.max(mse_vals), db_path)

        # Seleção (torneio ou roleta)
        probs = (fitness_vals - fitness_vals.min()) + 1e-6
        probs /= probs.sum()
        parents_idx = np.random.choice(np.arange(population_size), size=population_size, p=probs)
        parents = pop[parents_idx]

        # Crossover
        children = []
        for i in range(0, population_size, 2):
            p1, p2 = parents[i], parents[(i+1) % population_size]
            alpha = np.random.rand()
            child1 = alpha * p1 + (1 - alpha) * p2
            child2 = (1 - alpha) * p1 + alpha * p2
            children.extend([child1, child2])
        children = np.array(children)

        # Mutação
        mutation_rate = 0.1
        for child in children:
            if np.random.rand() < mutation_rate:
                gene = np.random.randint(0, 3)
                if gene == 0:
                    child[gene] = np.random.uniform(0, 20)
                elif gene == 1:
                    child[gene] = np.random.uniform(0, 2)
                else:
                    child[gene] = np.random.uniform(0, 5)

        # Atualização da população
        pop = children

        # Melhor da geração
        best_idx = np.argmax(fitness_vals)
        print(f"Geração {gen+1}/{generations} | Melhor: {-fitness_vals[best_idx]:.6f} | Médio: {np.mean(mse_vals):.6f}")

    # Resultado final
    fitness_vals = np.array([fitness_ga(ind, plant, t, setpoint) for ind in pop])
    best_idx = np.argmax(fitness_vals)
    best_solution = pop[best_idx]

    Kp, Ki, Kd = best_solution
    print("\nParâmetros PID via GA:")
    print(f"Kp = {Kp:.4f}")
    print(f"Ki = {Ki:.4f}")
    print(f"Kd = {Kd:.4f}")
    print(f"Custo (MSE) = {-fitness_vals[best_idx]:.6f}")

    return Kp, Ki, Kd
