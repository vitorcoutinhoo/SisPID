# pylint: disable="C0114, C0103, C0301"

import numpy as np
import control as ctl
from scipy.signal import find_peaks


def sintonize(K: float, L: float, T: float):
    """
    Sintoniza os parâmetros Kp, Ki e Kd PID usando os parâmetros K, L e T
    com base nos critérios de Ziegler–Nichols.

    Parâmetros:
    - K: ganho estático.
    - L: tempo morto.
    - T: constante de tempo.

    Retorna:
    - Kp, Ki, Kd.
    """

    Kp = 1.2  * (T / (K * L))
    Ti = 2 * L
    Td = 0.5 * L

    # Conversão para Ki e Kd
    Ki = Kp / Ti
    Kd = Kp * Td

    return Kp, Ki, Kd

def ziegler_nichols_1(plant: ctl.TransferFunction, T: np.ndarray, setpoint: float = 1, threshold: float = 0.02):
    """
    Função que aplica o método de Ziegler–Nichols 1 para ajuste de parâmetros PID.
    O método consiste em determinar os parâmetros K, L e T da planta e calcular os parâmetros PID (Kp, Ki, Kd)
    usando as fórmulas padrão de Ziegler–Nichols.

    Parâmetros:
    plant (TransferFunction): Função de transferência da planta.
    setpoint (float): Valor do setpoint desejado.
    T (array): Vetor de tempo para simulação.
    threshold (float): Limiar para detectar o início da resposta.

    Retorna:
    Kp (float), Ki (float), Kd (float)
    """

    # Entrada em degrau
    u = np.ones_like(T) * setpoint

    # Resposta da planta em malha aberta
    t_out, y_out = ctl.forced_response(plant, T=T, U=u)

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

    Kp, Ki, Kd = sintonize(K, L, T_const)

    print("\nParâmetros PID via Ziegler–Nichols 1:")
    print(f"Kp = {Kp:.4f}")
    print(f"Ki = {Ki:.4f}")
    print(f"Kd = {Kd:.4f}")

    return Kp, Ki, Kd


def ziegler_nichols_2(plant: ctl.TransferFunction, T: np.ndarray, setpoint: float = 1):
    """
    Função que aplica o método de Ziegler–Nichols para ajuste de parâmetros PID.
    O método consiste em varrer o ganho proporcional (Kp) e observar a resposta do sistema.
    A função calcula os parâmetros PID (Kp, Ki, Kd) usando as fórmulas padrão de Ziegler–Nichols.

    Parâmetros:
    setpoint (float): Valor do setpoint desejado.
    plant (TransferFunction): Função de transferência da planta.
    T (array): Vetor de tempo para simulação.

    Retorna:
    Kp_zn (float), Ki_zn (float), Kd_zn (float)
    """
    kp_range = np.linspace(0.1, 10000, 10000)
    Ku = None
    Tu = None

    for kp in kp_range:
        ki = 1
        kd = 0

        pid = ctl.tf([kd, kp, ki], [1, 0])
        sys = ctl.feedback(pid * plant, 1)
        u = np.ones_like(T) * setpoint

        t, y = ctl.forced_response(sys, T=T, U=u)
        peaks, _ = find_peaks(y)

        if len(peaks) >= 5:
            Ku = kp
            peak_times = t[peaks]
            periods = np.diff(peak_times)
            Tu = np.mean(periods)
            break

    if Ku is None or Tu is None:
        print("Não foi possível determinar Ku e Tu. Tente ajustar o intervalo de Kp.")
        return None, None, None

    # Fórmulas padrão de Ziegler–Nichols
    Kp_zn = 0.6 * Ku
    Ti_zn = 0.5 * Tu
    Td_zn = 0.125 * Tu

    Ki_zn = Kp_zn / Ti_zn
    Kd_zn = Kp_zn * Td_zn

    print("\nParâmetros PID via Ziegler–Nichols:")
    print(f"Kp = {Kp_zn:.4f}")
    print(f"Ki = {Ki_zn:.4f}")
    print(f"Kd = {Kd_zn:.4f}")

    return Kp_zn, Ki_zn, Kd_zn
