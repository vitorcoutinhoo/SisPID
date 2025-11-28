# pylint: disable="C0114, C0103, C0301"

import os
import sqlite3
import numpy as np
from datetime import datetime
import control as ctl


def init_database(db_path="db/pid_results.db"):
    """Cria o banco de dados com histÃ³rico evolutivo e robustez."""
    
    if os.path.exists(db_path):
        print(f"Banco '{db_path}' já existe")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tabela principal 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resultados (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            metodo TEXT,
            Kp REAL,
            Ki REAL,
            Kd REAL,
            mse REAL,
            overshoot REAL,
            tempo_acomodacao REAL,
            margem_ganho REAL,
            margem_fase REAL
        )
    """)
    
    # Histórico evolutivo
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historico_evolutivo (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            metodo TEXT,
            geracao INTEGER,
            melhor_fitness REAL,
            fitness_medio REAL,
            pior_fitness REAL
        )
    """)
    
    # NOVA: Tabela de testes de robustez
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS robustez (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            metodo TEXT,
            cenario TEXT,
            k_term REAL,
            tau REAL,
            mse REAL,
            overshoot REAL,
            tempo_acomodacao REAL,
            variacao_mse REAL,
            descricao TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✓ Banco '{db_path}' criado com sucesso")
    
    return True


def calcular_metricas(t, y, setpoint=1.0):
    """Calcula apenas as métricas essenciais."""
    
    # MSE
    mse = float(np.mean((y - setpoint) ** 2))
    
    # Overshoot (%)
    pico = np.max(y)
    overshoot = float(max(0, ((pico - setpoint) / setpoint) * 100))
    
    # Tempo de acomodação (2%)
    faixa = 0.02 * setpoint
    indices = np.where(np.abs(y - setpoint) <= faixa)[0]
    tempo_acomodacao = t[indices[0]] if len(indices) > 0 else t[-1]
    
    return {
        'mse': mse,
        'overshoot': overshoot,
        'tempo_acomodacao': tempo_acomodacao
    }


def calcular_robustez(Kp, Ki, Kd, plant):
    """
    Calcula métricas de robustez (margens de ganho e fase).
    """
    try:
        # Cria PID
        pid_tf = ctl.tf([Kd, Kp, Ki], [1, 0])
        
        # Sistema em malha aberta
        sys_ma = pid_tf * plant
        
        # Calcula margens
        gm, pm, wgc, wpc = ctl.margin(sys_ma)
        
        # Converte ganho para dB
        if np.isinf(gm) or gm > 1e6:
            gm_db = 999.99
        elif gm > 0:
            gm_db = 20 * np.log10(gm)
        else:
            gm_db = None
        
        # Converte fase para graus
        if pm is not None:
            pm_deg = pm if pm < 360 else pm % 360
        else:
            pm_deg = None
        
        return {
            'margem_ganho': gm_db,
            'margem_fase': pm_deg
        }
    except Exception as e:
        print(f"Não foi possível calcular robustez: {e}")
        return {
            'margem_ganho': None,
            'margem_fase': None
        }


def salvar_resultado(metodo, Kp, Ki, Kd, t, y, setpoint, plant, db_name="pid_results.db"):
    """Salva resultado no banco com métricas de desempenho e robustez."""
    
    # Calcula métricas de desempenho
    metricas = calcular_metricas(t, y, setpoint)
    
    # Calcula métricas de robustez
    robustez = calcular_robustez(Kp, Ki, Kd, plant)
    
    # Salva no banco
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO resultados 
        (data_hora, metodo, Kp, Ki, Kd, mse, overshoot, tempo_acomodacao, 
         margem_ganho, margem_fase)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        metodo,
        Kp, Ki, Kd,
        metricas['mse'],
        metricas['overshoot'],
        metricas['tempo_acomodacao'],
        robustez['margem_ganho'],
        robustez['margem_fase']
    ))
    
    conn.commit()
    conn.close()
    
    # Imprime resumo
    print(f"\n✓ Resultado salvo:")
    print(f"  MSE: {metricas['mse']:.6f}")
    print(f"  Overshoot: {metricas['overshoot']:.2f}%")
    print(f"  Tempo acomodação: {metricas['tempo_acomodacao']:.2f}s")
    if robustez['margem_ganho']:
        if robustez['margem_ganho'] > 900:
            print(f"  Margem de ganho: ∞ (infinita)")
        else:
            print(f"  Margem de ganho: {robustez['margem_ganho']:.2f} dB")
    if robustez['margem_fase']:
        print(f"  Margem de fase: {robustez['margem_fase']:.2f}°")


def comparar_metodos(db_name="pid_results.db"):
    """Mostra comparação simples entre métodos."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT metodo, 
               AVG(mse) as mse_avg,
               AVG(overshoot) as overshoot_avg,
               AVG(tempo_acomodacao) as ts_avg,
               AVG(margem_ganho) as mg_avg,
               AVG(margem_fase) as mf_avg,
               COUNT(*) as n
        FROM resultados
        GROUP BY metodo
        ORDER BY mse_avg
    """)
    
    resultados = cursor.fetchall()
    conn.close()
    
    if not resultados:
        print("\n⚠ Nenhum resultado encontrado no banco")
        return
    
    print("\n" + "="*90)
    print("COMPARAÇÃO DE MÉTODOS (média dos testes)")
    print("="*90)
    print(f"{'Método':<10} {'MSE':<12} {'Overshoot':<12} {'Ts (s)':<10} {'MG (dB)':<10} {'MF (°)':<10} {'N':<5}")
    print("-"*90)
    
    for row in resultados:
        metodo, mse, overshoot, ts, mg, mf, n = row
        if mg and mg > 900:
            mg_str = "∞"
        elif mg:
            mg_str = f"{mg:.2f}"
        else:
            mg_str = "N/A"
        
        mf_str = f"{mf:.2f}" if mf else "N/A"
        print(f"{metodo:<10} {mse:<12.6f} {overshoot:<12.2f} {ts:<10.2f} {mg_str:<10} {mf_str:<10} {n:<5}")
    
    print("="*90)
    print(f"\n✓ Melhor método (menor MSE): {resultados[0][0]}")
    
    # Análise de robustez
    print("\nANÁLISE DE ROBUSTEZ:")
    metodos_robustos = [r for r in resultados if r[4] and r[5]]
    if metodos_robustos:
        melhor_mg = max(metodos_robustos, key=lambda x: x[4] if x[4] else -999)
        melhor_mf = max(metodos_robustos, key=lambda x: x[5] if x[5] else -999)
        
        if melhor_mg[4] > 900:
            print(f"  Maior margem de ganho: {melhor_mg[0]} (∞ dB)")
        else:
            print(f"  Maior margem de ganho: {melhor_mg[0]} ({melhor_mg[4]:.2f} dB)")
        
        print(f"  Maior margem de fase: {melhor_mf[0]} ({melhor_mf[5]:.2f}°)")
        
        print("\n  Interpretação:")
        if melhor_mg[4] > 6 or melhor_mg[4] > 900:
            print("  ✓ Margem de ganho adequada (MG > 6 dB)")
        else:
            print("  ✗ Margem de ganho baixa (MG < 6 dB)")
        
        if melhor_mf[5] > 45:
            print("  ✓ Margem de fase adequada (MF > 45°)")
        else:
            print("  ✗ Margem de fase baixa (MF < 45°)")
    else:
        print("  Dados de robustez não disponíveis")


def salvar_historico_evolutivo(metodo, geracao, melhor_fitness, fitness_medio, pior_fitness, db_path="db/pid_results.db"):
    """Salva histórico de uma geração no banco de dados."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO historico_evolutivo 
            (data_hora, metodo, geracao, melhor_fitness, fitness_medio, pior_fitness)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metodo,
            geracao,
            melhor_fitness,
            fitness_medio,
            pior_fitness
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Erro ao salvar histórico: {e}")


# ============================================================================
# FUNÇÕES DE ROBUSTEZ INTEGRADAS
# ============================================================================

CENARIOS_ROBUSTEZ = {
    "Nominal": {"K_term": 59.81, "tau": 401.61, "desc": "Condições nominais"},
    "C1": {"K_term": 53.83, "tau": 401.61, "desc": "Degradação aquecedor (-10%)"},
    "C2": {"K_term": 65.79, "tau": 401.61, "desc": "Aquecedor eficiente (+10%)"},
    "C3": {"K_term": 59.81, "tau": 361.45, "desc": "Menor capacidade térmica (-10%)"},
    "C4": {"K_term": 59.81, "tau": 441.77, "desc": "Maior capacidade térmica (+10%)"},
    "C5": {"K_term": 53.83, "tau": 441.77, "desc": "Cenário combinado (pior caso)"}
}


def testar_robustez(metodo, Kp, Ki, Kd, t_sim, setpoint=80.0, db_path="db/pid_results.db"):
    """
    Testa robustez de um controlador PID em múltiplos cenários.
    
    Parâmetros:
        metodo: Nome do método
        Kp, Ki, Kd: Parâmetros PID sintonizados
        t_sim: Vetor de tempo
        setpoint: Valor de referência
        db_path: Caminho do banco de dados
    """
    from model.model import model, simulate
    
    print(f"\n{'='*70}")
    print(f"TESTE DE ROBUSTEZ: {metodo}")
    print(f"{'='*70}")
    print(f"{'Cenário':<10} {'MSE':<12} {'Variação':<12} {'Overshoot':<12}")
    print(f"{'-'*70}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    mse_nominal = None
    resultados = []
    
    for cenario, params in CENARIOS_ROBUSTEZ.items():
        # Criar planta para este cenário
        plant_cenario = model(params["K_term"], params["tau"])
        
        # Simular resposta
        t_resp, y_resp = simulate(plant_cenario, Kp, Ki, Kd, t_sim, setpoint)
        
        # Calcular métricas
        metricas = calcular_metricas(t_resp, y_resp, setpoint)
        mse = metricas['mse']
        
        # Guardar MSE nominal
        if cenario == "Nominal":
            mse_nominal = mse
        
        # Calcular variação percentual
        if mse_nominal and cenario != "Nominal":
            variacao = ((mse - mse_nominal) / mse_nominal) * 100
        else:
            variacao = 0.0
        
        # Salvar no banco
        cursor.execute("""
            INSERT INTO robustez 
            (data_hora, metodo, cenario, k_term, tau, mse, overshoot, tempo_acomodacao, variacao_mse, descricao)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metodo, cenario,
            params["K_term"], params["tau"],
            mse, metricas['overshoot'], metricas['tempo_acomodacao'],
            variacao, params["desc"]
        ))
        
        resultados.append((cenario, mse, variacao, metricas['overshoot']))
        
        # Exibir resultado
        var_str = f"{variacao:+.2f}%" if cenario != "Nominal" else "---"
        print(f"{cenario:<10} {mse:<12.6f} {var_str:<12} {metricas['overshoot']:<12.2f}")
    
    conn.commit()
    conn.close()
    
    # Análise
    variacoes = [r[2] for r in resultados if r[0] != "Nominal"]
    var_media = np.mean([abs(v) for v in variacoes])
    
    print(f"\n{'='*70}")
    print(f"📊 ANÁLISE DE ROBUSTEZ - {metodo}")
    print(f"{'='*70}")
    print(f"Variação média de MSE: {var_media:.2f}%")
    
    if var_media < 5:
        print("✓ Robustez EXCELENTE (< 5%)")
    elif var_media < 15:
        print("✓ Robustez BOA (< 15%)")
    elif var_media < 30:
        print("⚠ Robustez REGULAR (< 30%)")
    else:
        print("✗ Robustez BAIXA (> 30%)")


def comparar_robustez(db_path="db/pid_results.db"):
    """Compara robustez entre métodos testados."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT metodo, 
               AVG(ABS(variacao_mse)) as var_media,
               MAX(ABS(variacao_mse)) as var_max
        FROM robustez
        WHERE cenario != 'Nominal'
        GROUP BY metodo
        ORDER BY var_media ASC
    """)
    
    resultados = cursor.fetchall()
    conn.close()
    
    if not resultados:
        print("\n⚠ Nenhum teste de robustez encontrado!")
        return
    
    print(f"\n{'='*70}")
    print(f"COMPARAÇÃO DE ROBUSTEZ ENTRE MÉTODOS")
    print(f"{'='*70}")
    print(f"{'Método':<12} {'Var. Média (%)':<18} {'Var. Máx (%)':<15}")
    print(f"{'-'*70}")
    
    for row in resultados:
        metodo, var_media, var_max = row
        print(f"{metodo:<12} {var_media:<18.2f} {var_max:<15.2f}")
    
    print(f"{'='*70}")
    print(f"\n✓ Método MAIS ROBUSTO: {resultados[0][0]}")
    print(f"  Variação média: {resultados[0][1]:.2f}%")