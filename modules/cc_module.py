# pylint: disable="C0114, C0103, R0914, C0301"

import numpy as np
import control as ctl


def sintonize(K: float, L: float, T: float):
    """
    Sintoniza os parâmetros Kp, Ki e Kd PID usando os parâmetros K, L e T
    com base nos critérios de Cohen–Coon.
    
    Parâmetros:
    - K: ganho estático.
    - L: tempo morto.
    - T: constante de tempo.

    Retorna:
    - Kp, Ki, Kd.
    """

    Kp = (1 / K) * (T / L) * (1 + 0.35 * (L / T))
    Ti = L * (30 + 3 * (L / T)) / (9 + 20 * (L / T))
    Td = (8 * L) / (100 + 3 * (L / T))

    # Conversão para Ki e Kd
    Ki = Kp / Ti
    Kd = Kp * Td

    return Kp, Ki, Kd


def cohen_coon(plant: ctl.TransferFunction, t: np.ndarray, setpoint: float = 1.0, threshold: float = 0.02):
    """
    Identifica automaticamente os parâmetros K, L, T da planta e calcula PID via Cohen-Coon.
    
    Parâmetros:
    - plant: função de transferência da planta.
    - t: vetor de tempo.
    - setpoint: valor do degrau aplicado.
    - threshold: limiar para detectar início da resposta.

    Retorna:
    - Kp, Ki, Kd calculados.
    """
    # Entrada em degrau
    u = np.ones_like(t) * setpoint

    # Resposta da planta em malha aberta
    t_out, y_out = ctl.forced_response(plant, T=t, U=u)

    # K: ganho estático
    K = (y_out[-1] - y_out[0]) / setpoint

    # L: tempo morto
    delta_y = y_out - y_out[0]
    max_delta = np.max(delta_y)
    limiar = threshold * max_delta

    idx_L = np.argmax(delta_y > limiar)
    L = t_out[idx_L]

    # T: constante de tempo
    alvo = y_out[0] + 0.632 * max_delta
    idx_T = np.argmin(np.abs(y_out - alvo))
    T_const = t_out[idx_T] - L  # T contado a partir de L

    print("\nParâmetros identificados:")
    print(f"K = {K:.4f}")
    print(f"L = {L:.4f} s")
    print(f"T = {T_const:.4f} s")

    # Cálculo PID via Cohen–Coon
    Kp, Ki, Kd = sintonize(K, L, T_const)

    print("\nParâmetros PID via Cohen–Coon:")
    print(f"Kp = {Kp:.4f}")
    print(f"Ki = {Ki:.4f}")
    print(f"Kd = {Kd:.4f}")

    return Kp, Ki, Kd
