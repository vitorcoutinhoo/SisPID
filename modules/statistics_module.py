# pylint: disable="C0114, C0103, C0301"

"""
Módulo de análise estatística para comparação de métodos de sintonia PID.
Implementa o teste de Friedman para verificar significância estatística.
"""

import sqlite3
import numpy as np
import itertools
from scipy import stats

def teste_friedman(db_path="db/pid_results.db", metrica="mse"):
    """
    Executa o teste de Friedman para comparar múltiplos métodos.
    
    O teste de Friedman é um teste não-paramétrico usado para detectar
    diferenças em tratamentos através de múltiplas tentativas de teste.
    
    Args:
        db_path: Caminho do banco de dados
        metrica: Métrica a ser analisada ('mse', 'overshoot', 'tempo_acomodacao')
    
    Returns:
        dict com resultados do teste:
            - statistic: Estatística χ² de Friedman
            - pvalue: p-valor do teste
            - rankings: Ranking médio de cada método
            - n_metodos: Número de métodos comparados
            - n_iteracoes: Número de iterações por método
            - significativo: Boolean indicando se p < 0.05
    """
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Buscar métodos disponíveis
        cursor.execute("SELECT DISTINCT metodo FROM resultados ORDER BY metodo")
        metodos = [row[0] for row in cursor.fetchall()]
        
        if len(metodos) < 3:
            print(f"AVISO: Apenas {len(metodos)} métodos encontrados.")
            print("   O teste de Friedman requer pelo menos 3 métodos para comparação.")
            conn.close()
            return None
        
        # Buscar dados de cada método
        dados_metodos = []
        min_iteracoes = float('inf')
        
        for metodo in metodos:
            query = f"""
                SELECT {metrica}
                FROM resultados
                WHERE metodo = ?
                ORDER BY data_hora DESC
            """
            cursor.execute(query, (metodo,))
            valores = [row[0] for row in cursor.fetchall()]
            
            if len(valores) < min_iteracoes:
                min_iteracoes = len(valores)
            
            dados_metodos.append(valores)
        
        conn.close()
        
        # Verificar se há iterações suficientes
        if min_iteracoes < 3:
            print(f"AVISO: Apenas {min_iteracoes} iterações encontradas.")
            print("   Recomenda-se pelo menos 5 iterações para análise estatística confiável.")
            print("   O teste será executado, mas os resultados podem ter baixa confiabilidade.")
        
        # Truncar todos os arrays para o tamanho mínimo
        dados_truncados = [arr[:min_iteracoes] for arr in dados_metodos]
        
        # Executar teste de Friedman
        # Cada linha = uma iteração, cada coluna = um método
        dados_array = np.array(dados_truncados).T
        
        statistic, pvalue = stats.friedmanchisquare(*dados_truncados)
        
        # Calcular rankings médios
        # Para cada iteração, ranquear os métodos (1 = melhor)
        rankings = []
        for i in range(min_iteracoes):
            valores_iteracao = dados_array[i, :]
            ranks = stats.rankdata(valores_iteracao)  # Menor valor = menor rank
            rankings.append(ranks)
        
        rankings_array = np.array(rankings)
        ranking_medio = np.mean(rankings_array, axis=0)
        
        # Criar dicionário de rankings por método
        rankings_dict = {}
        for idx, metodo in enumerate(metodos):
            rankings_dict[metodo] = ranking_medio[idx]
        
        # Ordenar por ranking
        rankings_ordenados = sorted(rankings_dict.items(), key=lambda x: x[1])
        
        resultado = {
            'statistic': statistic,
            'pvalue': pvalue,
            'rankings': rankings_ordenados,
            'rankings_dict': rankings_dict,
            'n_metodos': len(metodos),
            'n_iteracoes': min_iteracoes,
            'significativo': pvalue < 0.05,
            'metrica': metrica.upper()
        }
        
        return resultado
        
    except Exception as e:
        print(f"Erro ao executar teste de Friedman: {e}")
        return None


def imprimir_resultado_friedman(resultado):
    """
    Imprime os resultados do teste de Friedman de forma formatada.
    
    Args:
        resultado: Dicionário retornado pela função teste_friedman()
    """
    
    if resultado is None:
        print("\nNão foi possível executar o teste estatístico.")
        return
    
    print("\n" + "="*70)
    print("TESTE DE FRIEDMAN - ANÁLISE DE SIGNIFICÂNCIA ESTATÍSTICA")
    print("="*70)
    
    print(f"\nCONFIGURAÇÃO DO TESTE:")
    print(f"   Métrica analisada: {resultado['metrica']}")
    print(f"   Métodos comparados: {resultado['n_metodos']}")
    print(f"   Iterações por método: {resultado['n_iteracoes']}")
    print(f"   Total de amostras: {resultado['n_metodos'] * resultado['n_iteracoes']}")
    
    print(f"\nRESULTADOS:")
    print(f"   Estatística χ² (Friedman): {resultado['statistic']:.4f}")
    print(f"   p-valor: {resultado['pvalue']:.6f}")
    
    print(f"\nINTERPRETAÇÃO:")
    if resultado['significativo']:
        print("   ✓ As diferenças entre os métodos são estatisticamente SIGNIFICATIVAS")
        print("     (p < 0.05 → Rejeita H₀)")
        print("     → Há evidências de que os métodos têm desempenhos diferentes")
    else:
        print("   ✗ As diferenças entre os métodos NÃO são estatisticamente significativas")
        print("     (p ≥ 0.05 → Não rejeita H₀)")
        print("     → Não há evidências suficientes de diferenças de desempenho")

    print(f"\nRANKING MÉDIO (quanto menor, melhor):")
    print(f"   {'Posição':<10} {'Método':<15} {'Ranking Médio':<15}")
    print(f"   {'-'*40}")
    
    for idx, (metodo, rank) in enumerate(resultado['rankings'], 1):
        emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
        print(f"   {emoji} {idx}º{'':<6} {metodo:<15} {rank:<15.2f}")
    
    print("\n" + "="*70)
    
    # Avisos sobre confiabilidade
    if resultado['n_iteracoes'] < 5:
        print("\nAVISO: Número de iterações baixo (< 5)")
        print("   Recomenda-se executar mais iterações para maior confiabilidade estatística.")
    
    if resultado['n_metodos'] < 4:
        print("\nAVISO: Poucos métodos comparados (< 4)")
        print("   Quanto mais métodos, mais robusto é o teste estatístico.")


def analise_completa(db_path="db/pid_results.db"):
    """
    Executa análise estatística completa com múltiplas métricas.
    
    Args:
        db_path: Caminho do banco de dados
    
    Returns:
        dict com resultados para cada métrica
    """
    
    print("\n" + "="*70)
    print("ANÁLISE ESTATÍSTICA COMPLETA - MÚLTIPLAS MÉTRICAS")
    print("="*70)
    
    metricas = ['mse', 'overshoot', 'tempo_acomodacao']
    resultados = {}
    
    for metrica in metricas:
        print(f"\n{'='*70}")
        print(f"MÉTRICA: {metrica.upper()}")
        print(f"{'='*70}")
        
        resultado = teste_friedman(db_path, metrica)
        
        if resultado:
            imprimir_resultado_friedman(resultado)
            resultados[metrica] = resultado
        else:
            print(f"\nNão foi possível analisar a métrica {metrica}")
    
    return resultados


def gerar_resumo_estatistico(db_path="db/pid_results.db"):
    """
    Gera um resumo consolidado da análise estatística.
    
    Args:
        db_path: Caminho do banco de dados
    
    Returns:
        str com resumo formatado
    """
    
    resultado = teste_friedman(db_path, "mse")
    
    if resultado is None:
        return "Dados insuficientes para análise estatística."
    
    resumo = []
    resumo.append("╔═══════════════════════════════════════════════════╗")
    resumo.append("║     RESUMO DA ANÁLISE ESTATÍSTICA (MSE)           ║")
    resumo.append("╚═══════════════════════════════════════════════════╝")
    resumo.append("")
    resumo.append(f"Teste: Friedman (χ² = {resultado['statistic']:.4f})")
    resumo.append(f"p-valor: {resultado['pvalue']:.6f}")
    resumo.append("")
    
    if resultado['significativo']:
        resumo.append("✓ SIGNIFICATIVO (p < 0.05)")
        resumo.append("  → Diferenças entre métodos são estatisticamente")
        resumo.append("    relevantes e não ocorreram por acaso")
    else:
        resumo.append("✗ NÃO SIGNIFICATIVO (p ≥ 0.05)")
        resumo.append("  → Não há evidências estatísticas de diferenças")
        resumo.append("    significativas entre os métodos")
    
    resumo.append("")
    resumo.append("🏆 TOP 3 MÉTODOS:")
    for idx, (metodo, rank) in enumerate(resultado['rankings'][:3], 1):
        emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
        resumo.append(f"  {emoji} {metodo} (rank: {rank:.2f})")
    
    resumo.append("")
    resumo.append(f"Amostra: {resultado['n_metodos']} métodos × {resultado['n_iteracoes']} iterações")
    resumo.append("─" * 51)
    
    return "\n".join(resumo)


# Função auxiliar para integração com GUI
def obter_dados_para_grafico(db_path="db/pid_results.db", metrica="mse"):
    """
    Obtém dados formatados para plotagem de gráficos.
    
    Args:
        db_path: Caminho do banco de dados
        metrica: Métrica a ser analisada
    
    Returns:
        dict com dados prontos para visualização
    """
    
    resultado = teste_friedman(db_path, metrica)
    
    if resultado is None:
        return None
    
    metodos = [m for m, _ in resultado['rankings']]
    rankings = [r for _, r in resultado['rankings']]
    
    return {
        'metodos': metodos,
        'rankings': rankings,
        'pvalue': resultado['pvalue'],
        'significativo': resultado['significativo'],
        'statistic': resultado['statistic']
    }

def posthoc_nemenyi(rankings_dict, n_iteracoes):
    """
    Executa o pós-teste de Nemenyi após o teste de Friedman.
    
    Args:
        rankings_dict: dict -> {metodo: ranking_medio}
        n_iteracoes: int -> número de conjuntos (bloques) usados no Friedman
        
    Retorna:
        Lista de comparações:
            [
                (metodoA, metodoB, diff, CD, significativo)
            ]
    """
    
    metodos = list(rankings_dict.keys())
    k = len(metodos)              # número de métodos
    N = n_iteracoes               # número de blocos (iterações)
    
    # Ordenar rankings
    ordered = sorted(rankings_dict.items(), key=lambda x: x[1])
    
    # Função q_alpha para Nemenyi
    # aproximação pela distribuição studentized range
    q_alpha = {
        0.10: 2.291,  # valores aproximados
        0.05: 2.569,
        0.01: 3.291
    }
    
    alpha = 0.05  # nível de significância
    q = q_alpha[alpha]
    
    # Diferença crítica
    CD = q * np.sqrt(k * (k + 1) / (6 * N))
    
    resultados = []
    
    # Comparações pareadas
    for (m1, r1), (m2, r2) in itertools.combinations(ordered, 2):
        diff = abs(r1 - r2)
        significativo = diff > CD
        resultados.append((m1, m2, diff, CD, significativo))
    
    return resultados


def imprimir_posthoc_nemenyi(resultados):
    """
    Imprime os resultados do pós-teste de Nemenyi.
    """
    print("\n" + "="*70)
    print("PÓS-TESTE DE NEMENYI (α = 0.05)")
    print("="*70)
    
    print(f"{'Método A':<15} {'Método B':<15} {'Diferença':<12} {'CD':<10} {'Significativo'}")
    print("-"*70)
    
    for m1, m2, diff, CD, sig in resultados:
        flag = "✓" if sig else "✗"
        print(f"{m1:<15} {m2:<15} {diff:<12.4f} {CD:<10.4f} {flag}")
    
    print("-"*70)
    
    total_sig = sum(1 for r in resultados if r[4])
    print(f"\nTotal de pares com diferença significativa: {total_sig}")

if __name__ == "__main__":
    # Teste do módulo
    print("\nTESTE DO MÓDULO DE ESTATÍSTICA")
    print("="*70)
    
    for metrica in ["mse", "overshoot", "tempo_acomodacao"]:
        resultado = teste_friedman("db/pid_results.db", metrica)
        
        if resultado:
            imprimir_resultado_friedman(resultado)
            
            posthoc = posthoc_nemenyi(resultado['rankings_dict'], resultado['n_iteracoes'])
            if posthoc:
                imprimir_posthoc_nemenyi(posthoc)
        else:
            print("\nNão foi possível executar o teste.")
            print("   Certifique-se de que há dados suficientes no banco.")

