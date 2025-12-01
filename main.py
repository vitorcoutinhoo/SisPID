# pylint: disable="C0114, C0103, R0914, C0301, W0612"

import sys
import sqlite3
import numpy as np
from model.model import model, simulate

# Importar métodos de sintonia
from modules.zn_module import ziegler_nichols_1
from modules.cc_module import cohen_coon
from modules.pso_module import tune_pid_pso
from modules.ga_module import tune_pid_ga
from modules.de_module import tune_pid_de
from modules.cma_module import tune_pid_cma
from modules.statistics_module import teste_friedman, imprimir_resultado_friedman, gerar_resumo_estatistico

# Importar funções do DB
from db.db_module import (
    init_database, 
    salvar_resultado, 
    comparar_metodos,
    testar_robustez,
    comparar_robustez
)


def executar_sintonia(k_term, tau, setpoint, t_final, n_pontos, 
                     metodos_selecionados, iteracoes=15, 
                     executar_robustez=True, db_path="db/pid_results.db"):
    """
    Executa sintonia PID com os parâmetros fornecidos.
    
    Esta função foi refatorada para ser chamada tanto pela linha de comando
    quanto pela interface gráfica.
    
    Args:
        k_term: Ganho térmico (°C/W)
        tau: Constante de tempo (s)
        setpoint: Temperatura desejada (°C)
        t_final: Tempo final de simulação (s)
        n_pontos: Número de pontos da simulação
        metodos_selecionados: Dict com nome->função dos métodos
        iteracoes: Número de iterações por método
        executar_robustez: Se True, executa análise de robustez
        db_path: Caminho do banco de dados
    
    Returns:
        pid_params: Dict com parâmetros PID de cada método
    """
    
    # Criar modelo da planta
    plant = model(k_term, tau)
    t = np.linspace(0, t_final, n_pontos)
    
    print("\n" + "="*70)
    print("FASE 1: SINTONIA DE CONTROLADORES PID")
    print("="*70)
    print(f"Planta: K_Term={k_term} °C/W, τ={tau} s")
    print(f"Setpoint: {setpoint}°C, Tempo: {t_final}s, Pontos: {n_pontos}")
    print(f"Métodos: {', '.join(metodos_selecionados.keys())}")
    print(f"Iterações: {iteracoes}")
    print("="*70)
    
    pid_params = {}
    
    # Executar cada método
    for iteration in range(1, iteracoes + 1):
        for name, func in metodos_selecionados.items():
            print(f"\n{'='*70}")
            print(f"MÉTODO: {name} - Iteração {iteration}/{iteracoes}")
            print(f"{'='*70}")
            
            try:
                # Executar sintonia
                if name in ['ZN1', 'CC']:
                    kp, ki, kd = func(plant, t, setpoint)
                else:
                    kp, ki, kd = func(plant, t, setpoint, db_path=db_path)
                
                pid_params[name] = (kp, ki, kd)
                
                # Simular resposta
                tresp, yresp = simulate(plant, kp, ki, kd, t, setpoint)
                
                # Salvar resultado
                salvar_resultado(name, kp, ki, kd, tresp, yresp, setpoint, 
                               plant, db_name=db_path)
                
            except Exception as e:
                print(f"ERRO ao executar {name}: {str(e)}")
    
    # Mostrar comparação
    print("\n" + "="*70)
    print("RESUMO - SINTONIA NOMINAL")
    print("="*70)
    comparar_metodos(db_name=db_path)
    
    # Análise de robustez (se solicitado)
    if executar_robustez and pid_params:
        print("\n" + "="*70)
        print("FASE 2: TESTES DE ROBUSTEZ EM MÚLTIPLOS CENÁRIOS")
        print("="*70)
        
        for metodo, (kp, ki, kd) in pid_params.items():
            try:
                testar_robustez(metodo, kp, ki, kd, t, k_term, tau, setpoint, db_path=db_path)
            except Exception as e:
                print(f"ERRO ao testar robustez de {metodo}: {e}")
        
        # Comparação final de robustez
        print("\n" + "="*70)
        print("RESUMO - COMPARAÇÃO DE ROBUSTEZ")
        print("="*70)

        metricas = ["mse", "overshoot", "tempo_acomodacao"]
        for metrica in metricas:
            print(f"\n{'─'*70}")
            print(f"MÉTRICA: {metrica.upper()}")
            print(f"{'─'*70}")
            resultado = teste_friedman(db_path, metrica)
            if resultado:
                imprimir_resultado_friedman(resultado)

        comparar_robustez(db_path=db_path)
    
    print("\n" + "="*70)
    print("FASE 3: ANÁLISE ESTATÍSTICA (TESTE DE FRIEDMAN)")
    print("="*70)
    resultado_friedman = teste_friedman(db_path, "mse")
    if resultado_friedman:
        imprimir_resultado_friedman(resultado_friedman)
    
    print("\n✓ Execução concluída!")
    return pid_params


def main_cli():
    """
    Modo de linha de comando (CLI) - Executa com parâmetros padrão.
    Mantido para compatibilidade e testes rápidos.
    """
    
    # Inicializa banco
    init_database("db/pid_results.db")
    
    # Parâmetros padrão (modelo original)
    k_term = 59.81
    tau = 401.61
    setpoint = 80.0
    t_final = 2 * tau  # 803.22s
    n_pontos = 1000
    
    # Todos os métodos
    metodos = {
        "ZN1": ziegler_nichols_1,
        "CC": cohen_coon,
        "GA": tune_pid_ga,
        "PSO": tune_pid_pso,
        "DE": tune_pid_de,
        "CMA-ES": tune_pid_cma
    }
    
    # Executar
    executar_sintonia(
        k_term=k_term,
        tau=tau,
        setpoint=setpoint,
        t_final=t_final,
        n_pontos=n_pontos,
        metodos_selecionados=metodos,
        iteracoes=15,
        executar_robustez=True,
        db_path="db/pid_results.db"
    )


def main_gui():
    """
    Modo de interface gráfica (GUI) - Abre a interface.
    """
    # Inicializa banco
    init_database("db/pid_results.db")
    
    print("\n" + "="*70)
    print("SISTEMA DE ANÁLISE COMPARATIVA DE SINTONIA PID")
    print("="*70)
    print("Iniciando interface gráfica...")
    print("Use a aba 'Configuração & Execução' para definir parâmetros")
    print("="*70 + "\n")
    
    # Abre GUI
    from GUI.gui import main as gui_main
    gui_main()
    
def print_PID_params(path="db/pid_results.db"):
    """Obtém os parâmetros PID médios de cada método."""
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT metodo, AVG(Kp) as kp_avg, AVG(Ki) as ki_avg, AVG(Kd) as kd_avg
            FROM resultados
            GROUP BY metodo
            ORDER BY metodo
        """)
        
        resultados = cursor.fetchall()
        conn.close()
        
        return resultados
        
    except Exception as e:
        print(f"Erro ao obter parâmetros PID: {e}")
        return []
    

if __name__ == "__main__":
    # Verifica argumentos de linha de comando
    if len(sys.argv) > 1:
        if sys.argv[1] == "--cli":
            # Modo CLI (linha de comando)
            main_cli()
        elif sys.argv[1] == "--gui":
            # Modo GUI (interface gráfica)
            main_gui()
        elif sys.argv[1] == "--help":
            print("\nUso:")
            print("  python main.py          → Abre interface gráfica (padrão)")
            print("  python main.py --gui    → Abre interface gráfica")
            print("  python main.py --cli    → Executa via linha de comando")
            print("  python main.py --help   → Mostra esta ajuda\n")
        else:
            print(f"Argumento inválido: {sys.argv[1]}")
            print("Use --help para ver opções disponíveis")
    else:
        # Comportamento padrão: abrir GUI
        x = print_PID_params()
        print(x)
        main_gui()
        