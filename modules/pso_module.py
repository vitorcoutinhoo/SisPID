# pylint: disable="C0114, C0103, R0914, C0301, W0612"

import numpy as np
from db.db_module import salvar_historico_evolutivo
from model.model import simulate, model


def _mse_response(pid, t, setpoint, plant):
    """Calcula o MSE da resposta do sistema para um conjunto de parâmetros PID."""
    Kp, Ki, Kd = pid
    try:
        T_resp, Y_resp = simulate(plant, Kp, Ki, Kd, t, setpoint)
        mse = np.mean((Y_resp - setpoint) ** 2)
    except (ValueError, RuntimeError):
        mse = 1e6  # penaliza simulações instáveis
    return mse


def tune_pid_pso(plant=None, t=None, setpoint=1.0,
                 n_particles=20, iters=50,
                 bounds=((0,0,0), (20,2,5)),
                 db_path="db/pid_results.db"):
    """
    Implementação manual do PSO para ajuste PID com salvamento de histórico.
    """
    if plant is None:
        plant = model(59.81, 401.61)

    if t is None:
        t = np.linspace(0, 2000, 1000)

    # Limites inferior e superior de cada parâmetro
    lower_bounds, upper_bounds = np.array(bounds[0]), np.array(bounds[1])

    # Hiperparâmetros do PSO
    w = 0.7   # inércia
    c1 = 1.5  # atração para o melhor pessoal
    c2 = 1.5  # atração para o melhor global

    # Inicialização aleatória das partículas
    particles = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(n_particles, 3))
    velocities = np.zeros_like(particles)

    # Avalia fitness inicial
    fitness = np.array([_mse_response(p, t, setpoint, plant) for p in particles])

    # Melhor pessoal de cada partícula
    pbest_positions = particles.copy()
    pbest_scores = fitness.copy()

    # Melhor global
    gbest_idx = np.argmin(fitness)
    gbest_position = particles[gbest_idx].copy()
    gbest_score = fitness[gbest_idx]

    # Salva histórico inicial
    salvar_historico_evolutivo("PSO", 0, float(gbest_score), float(np.mean(fitness)), float(np.max(fitness)), db_path)

    # Loop principal do PSO
    for it in range(iters):
        for i in range(n_particles):
            r1, r2 = np.random.rand(3), np.random.rand(3)  # fatores aleatórios

            # Atualiza velocidade
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest_positions[i] - particles[i]) +
                             c2 * r2 * (gbest_position - particles[i]))

            # Atualiza posição
            particles[i] = particles[i] + velocities[i]

            # Aplica limites (clamping)
            particles[i] = np.clip(particles[i], lower_bounds, upper_bounds)

            # Avalia nova posição
            score = _mse_response(particles[i], t, setpoint, plant)

            # Atualiza melhor pessoal
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = particles[i].copy()

            # Atualiza melhor global
            if score < gbest_score:
                gbest_score = score
                gbest_position = particles[i].copy()

        # Atualiza fitness da população
        fitness = np.array([_mse_response(p, t, setpoint, plant) for p in particles])
        
        # Salva histórico da geração
        salvar_historico_evolutivo("PSO", it + 1, float(gbest_score), np.mean(fitness), np.max(fitness), db_path)
        
        print(f"Iteração {it+1}/{iters} | Melhor: {gbest_score:.6f} | Médio: {np.mean(fitness):.6f}")

    Kp, Ki, Kd = gbest_position
    print("\nParâmetros PID via PSO:")
    print(f"Kp = {Kp:.4f}")
    print(f"Ki = {Ki:.4f}")
    print(f"Kd = {Kd:.4f}")
    print(f"Custo (MSE) = {gbest_score:.6f}")

    return Kp, Ki, Kd