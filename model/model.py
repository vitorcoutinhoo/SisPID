# pylint: disable="C0114, C0103, R0903, C0301, R0913, R0917"

import control as ctl
import numpy as np


def model(gterm: float, t: float):
    """
    Função que define o modelo da planta (estufa elétrica)
    e retorna a função de transferência.

    A planta é modelada como um sistema de primeira ordem com ganho K e constante de tempo tau.
    A função de transferência é dada por plant(s) = K / (tau * s + 1), onde:

    - K é o ganho térmico (°C/W)
    - tau é a constante de tempo (s)

    A função de transferência é criada usando a biblioteca control.
    A função retorna a função de transferência da planta.

    Parâmetros:
    gterm (float): Ganho térmico da planta (°C/W).
    t (float): Constante de tempo da planta (s).

    Retorna:
    plant (TransferFunction): Função de transferência da planta. dada por K / (tau * s + 1).
    """

    # Parâmetros da planta (estufa elétrica)
    K = gterm  # Ganho térmico (°C/W)
    tau = t  # Constante de tempo (s)
    plant = ctl.tf([K], [tau, 1])  # Função de transferência da planta

    return plant


def simulate(plant: ctl.TransferFunction, Kp: float, Ki: float, Kd: float, T: np.ndarray, setpoint: float = 1):
    """
    Função que simula a resposta do sistema a um degrau unitário.

    A função usa a biblioteca control para simular a resposta do sistema
    e retorna o tempo e a resposta do sistema.

    Parâmetros:
    plant (TransferFunction): Função de transferência da planta.
    Kp (float): Ganho proporcional do controlador PID.
    Ki (float): Ganho integral do controlador PID.
    Kd (float): Ganho derivativo do controlador PID.
    T (array): Vetor de tempo para simulação.
    setpoint (float): Valor do setpoint desejado (default é 1).

    Retorna:
    t (array): Vetor de tempo.
    y (array): Resposta do sistema.
    """

    pid = ctl.tf([Kd, Kp, Ki], [1, 0])  # Formula do PID
    sys = ctl.feedback(pid * plant, 1)  # Sistema em malha fechada
    u = setpoint * np.ones_like(T)  # Sinal de entrada (setpoint)
    t, y = ctl.forced_response(sys, T=T, U=u)  # Resposta do sistema

    return t, y
