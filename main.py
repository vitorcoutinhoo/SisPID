# pylint: disable="C0114, C0103, R0914, C0301, W0612"

import sys
import numpy as np
from model.model import model, simulate

# Importar métodos de sintonia
from modules.zn_module import ziegler_nichols_1
from modules.cc_module import cohen_coon
from modules.pso_module import tune_pid_pso
from modules.ga_module import tune_pid_ga
from modules.de_module import tune_pid_de
from modules.cma_module import tune_pid_cma

# Importar funções do DB
from db.db_module import (
    init_database, 
    salvar_resultado, 
    comparar_metodos,
    testar_robustez,
    comparar_robustez
)


def main(modo="completo"):
    """
    Executa sintonia PID com testes de robustez integrados.
    
    Modos:
        - completo: Sintonia + Robustez
        - sintonia: Apenas sintonia nominal
        - robustez: Apenas testes de robustez (requer sintonias prévias)
    """
    
    # Inicializa banco
    init_database("db/pid_results.db")
    
    # Modelo Térmico da Estufa Elétrica
    Kterm = 59.81
    tau = 401.61
    plant = model(Kterm, tau)

    # Parâmetros de Simulação
    t = np.linspace(0, 2*tau, 1000)
    setpoint = 80.0

    # Métodos disponíveis
    metodos = {
        "ZN1": lambda: ziegler_nichols_1(plant, t, setpoint),
        "CC": lambda: cohen_coon(plant, t, setpoint),
        "GA": lambda: tune_pid_ga(plant, t, setpoint, db_path="db/pid_results.db"),
        "PSO": lambda: tune_pid_pso(plant, t, setpoint, db_path="db/pid_results.db"),
        "DE": lambda: tune_pid_de(plant, t, setpoint, db_path="db/pid_results.db"),
        "CMA-ES": lambda: tune_pid_cma(plant, t, setpoint, db_path="db/pid_results.db")
    }
    
    if modo in ["completo", "sintonia"]:
        print("\n" + "="*70)
        print("FASE 1: SINTONIA DE CONTROLADORES PID (CONDIÇÃO NOMINAL)")
        print("="*70)

        pid_params = {}  # Armazena parâmetros para testes de robustez
        
        x = 15
        # Executa cada método x vezes
        for iteration in range(1, x + 1):
            for name, func in metodos.items():
                print(f"\n{'='*70}")
                print(f"MÉTODO: {name} - Iteração {iteration}/{x}")
                print(f"{'='*70}")

                try:
                    # Executa sintonia
                    kp, ki, kd = func()
                    pid_params[name] = (kp, ki, kd)
                    
                    # Simula resposta
                    tresp, yresp = simulate(plant, kp, ki, kd, t, setpoint)
                    
                    # Salva resultado final na tabela 'resultados'
                    salvar_resultado(name, kp, ki, kd, tresp, yresp, setpoint, plant, 
                        db_name="db/pid_results.db")
                    
                except Exception as e:
                    print(f"ERRO ao executar {name}: {str(e)}")

        # Mostra comparação
        print("\n" + "="*70)
        print("RESUMO - SINTONIA NOMINAL")
        print("="*70)
        comparar_metodos(db_name="db/pid_results.db")

    if modo in ["completo", "robustez"]:
        print("\n" + "="*70)
        print("FASE 2: TESTES DE ROBUSTEZ EM MÚLTIPLOS CENÁRIOS")
        print("="*70)
        
        # Se estamos no modo "robustez", buscar parâmetros do banco
        if modo == "robustez":
            import sqlite3
            conn = sqlite3.connect("db/pid_results.db")
            cursor = conn.cursor()
            
            pid_params = {}
            for metodo in metodos.keys():
                cursor.execute("""
                    SELECT Kp, Ki, Kd FROM resultados 
                    WHERE metodo = ?
                    ORDER BY data_hora DESC LIMIT 1
                """, (metodo,))
                
                result = cursor.fetchone()
                if result:
                    pid_params[metodo] = result
            
            conn.close()
            
            if not pid_params:
                print("⚠ Nenhuma sintonia encontrada! Execute primeiro o modo 'sintonia' ou 'completo'")
                return
        
        # Executar testes de robustez para cada método
        for metodo, (kp, ki, kd) in pid_params.items():
            try:
                testar_robustez(metodo, kp, ki, kd, t, setpoint, db_path="db/pid_results.db")
            except Exception as e:
                print(f"ERRO ao testar robustez de {metodo}: {e}")
        
        # Comparação final de robustez
        print("\n" + "="*70)
        print("RESUMO - COMPARAÇÃO DE ROBUSTEZ")
        print("="*70)
        comparar_robustez(db_path="db/pid_results.db")

    print("\n✓ Execução concluída!")


if __name__ == "__main__":
    # Executa modo completo por padrão
    main(modo="completo")
    
    # Abre GUI
    from GUI.gui import main as gui_main
    gui_main()