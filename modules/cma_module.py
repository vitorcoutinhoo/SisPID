# pylint: disable="C0114, C0103, R0914, C0301, W0612"

import numpy as np
from db.db_module import salvar_historico_evolutivo
from model.model import simulate, model


def _mse_response(pid, t, setpoint, plant):
    """
    Calcula o erro quadrático médio (MSE) da resposta do sistema
    para um conjunto de parâmetros PID.

    Parâmetros:
        pid : list | np.ndarray -> [Kp, Ki, Kd]
        t : np.ndarray -> vetor de tempo da simulação
        setpoint : float -> valor desejado
        plant : control.TransferFunction -> planta

    Retorna:
        float -> erro médio quadrático (quanto menor, melhor)
    """
    Kp, Ki, Kd = pid
    try:
        _, Y_resp = simulate(plant, Kp, Ki, Kd, t, setpoint)
        mse = np.mean((Y_resp - setpoint) ** 2)
    except Exception:
        mse = 1e6  # penaliza respostas instáveis
    return mse


def tune_pid_cma(plant=None, t=None, setpoint=1.0,
                 generations=50, population_size=None,
                 sigma0=0.3,
                 bounds=((0, 0, 0), (20, 2, 5)),
                 db_path="db/pid_results.db"):
    """Ajuste PID usando CMA-ES com histórico."""

    if plant is None:
        plant = model(59.81, 401.61)
    if t is None:
        t = np.linspace(0, 2000, 1000)

    lower_bounds, upper_bounds = np.array(bounds[0]), np.array(bounds[1])
    n = 3

    mean = (lower_bounds + upper_bounds) / 2

    if population_size is None:
        lam = 4 + int(3 * np.log(n))
    else:
        lam = population_size

    mu = lam // 2

    # Pesos de recombinação
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mu_eff = 1 / np.sum(weights**2)

    # Parâmetros de adaptação
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c1 = 2 / ((n + 1.3)**2 + mu_eff)
    c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + 2 * mu_eff / 2))
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

    # Inicializações
    sigma = sigma0
    cov = np.eye(n)
    p_c = np.zeros(n)
    p_sigma = np.zeros(n)

    best_cost = float("inf")
    best_solution = mean.copy()

    print(f"Inicialização CMA-ES -> sigma0 = {sigma0}, população = {lam}")

    for gen in range(generations):
        # Amostragem da população
        A = np.linalg.cholesky(cov)
        z = np.random.randn(lam, n)
        X = mean + sigma * (z @ A.T)

        # Aplica limites
        X = np.clip(X, lower_bounds, upper_bounds)

        # Avalia população
        costs = np.array([_mse_response(x, t, setpoint, plant) for x in X])
        idx_sorted = np.argsort(costs)
        X = X[idx_sorted]
        z = z[idx_sorted]
        costs = costs[idx_sorted]

        # Atualiza melhor global
        if costs[0] < best_cost:
            best_cost = costs[0]
            best_solution = X[0].copy()

        # Salva histórico da geração
        salvar_historico_evolutivo("CMA-ES", gen + 1, best_cost, np.mean(costs), np.max(costs), db_path)

        # Atualização da média
        old_mean = mean.copy()
        mean = np.dot(weights, X[:mu])

        # Caminho de evolução
        y = (mean - old_mean) / sigma
        C_half = np.linalg.cholesky(cov)
        inv_C_half = np.linalg.inv(C_half)
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_C_half @ y)

        # Atualiza sigma
        norm_p_sigma = np.linalg.norm(p_sigma)
        expected_norm = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / expected_norm - 1))

        # Atualização da covariância
        h_sigma_cond = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * expected_norm
        p_c = (1 - c_c) * p_c + (np.sqrt(c_c * (2 - c_c) * mu_eff) * y) * (1.0 if h_sigma_cond else 0.0)

        rank_mu = np.zeros((n, n))
        for k in range(mu):
            y_k = (X[k] - old_mean) / sigma
            rank_mu += weights[k] * np.outer(y_k, y_k)

        cov = (1 - c1 - c_mu) * cov + c1 * np.outer(p_c, p_c) + c_mu * rank_mu

        print(f"Geração {gen+1}/{generations} | Melhor: {best_cost:.6f} | Médio: {np.mean(costs):.6f}")

    Kp, Ki, Kd = best_solution
    print("\nParâmetros PID via CMA-ES:")
    print(f"Kp = {Kp:.4f}")
    print(f"Ki = {Ki:.4f}")
    print(f"Kd = {Kd:.4f}")
    print(f"Custo (MSE) = {best_cost:.6f}")

    return Kp, Ki, Kd
