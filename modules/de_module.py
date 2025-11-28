# pylint: disable="C0114, C0103, R0914, C0301, W0612"

import numpy as np
from db.db_module import salvar_historico_evolutivo
from model.model import simulate, model


def _mse_response(pid, t, setpoint, plant):
    """
    Calcula o erro quadrático médio (MSE) para um conjunto de parâmetros PID.

    Parâmetros:
    pid : tuple/list/np.ndarray
        Contém [Kp, Ki, Kd].
    t : np.ndarray
        Vetor de tempo da simulação.
    setpoint : float
        Valor de referência (degrau desejado).
    plant : control.TransferFunction
        Planta simulada.

    Retorna:
    mse : float
        Erro quadrático médio da resposta.
    """
    Kp, Ki, Kd = pid
    try:
        _, Y_resp = simulate(plant, Kp, Ki, Kd, t, setpoint)
        mse = np.mean((Y_resp - setpoint) ** 2)
    except Exception:
        # Caso a simulação fique instável, penaliza fortemente
        mse = 1e6
    return mse


def tune_pid_de(plant=None, t=None, setpoint=1.0,
                pop_size=20, generations=50,
                F=0.8, CR=0.9,
                bounds=((0, 0, 0), (20, 2, 5)),
                db_path="db/pid_results.db"):
    """Ajuste PID usando Differential Evolution com histórico."""

    if plant is None:
        plant = model(59.81, 401.61)

    if t is None:
        t = np.linspace(0, 2000, 1000)

    lower_bounds, upper_bounds = np.array(bounds[0]), np.array(bounds[1])
    dim = 3

    # Inicialização da população
    pop = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(pop_size, dim))

    # Avalia custo inicial
    costs = np.array([_mse_response(ind, t, setpoint, plant) for ind in pop])
    best_idx = np.argmin(costs)
    best = pop[best_idx].copy()
    best_cost = costs[best_idx]

    # Salva histórico inicial
    salvar_historico_evolutivo("DE", 0, float(best_cost), np.mean(costs), np.max(costs), db_path)
    
    print(f"Inicialização -> Melhor custo = {best_cost:.6f}")

    # Loop principal
    for gen in range(generations):
        for i in range(pop_size):
            # Mutação
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + F * (b - c)

            # Restringe dentro dos limites
            mutant = np.clip(mutant, lower_bounds, upper_bounds)

            # Crossover
            cross = np.random.rand(dim) < CR
            jrand = np.random.randint(dim)
            cross[jrand] = True
            trial = np.where(cross, mutant, pop[i])

            # Seleção
            trial_cost = _mse_response(trial, t, setpoint, plant)
            if trial_cost < costs[i]:
                pop[i] = trial
                costs[i] = trial_cost

                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best = trial.copy()

        # Salva histórico da geração
        salvar_historico_evolutivo("DE", gen + 1, float(best_cost), np.mean(costs), np.max(costs), db_path)
        
        print(f"Geração {gen+1}/{generations} | Melhor: {best_cost:.6f} | Médio: {np.mean(costs):.6f}")

    Kp, Ki, Kd = best
    print("\nParâmetros PID via DE:")
    print(f"Kp = {Kp:.4f}")
    print(f"Ki = {Ki:.4f}")
    print(f"Kd = {Kd:.4f}")
    print(f"Custo (MSE) = {best_cost:.6f}")

    return Kp, Ki, Kd




