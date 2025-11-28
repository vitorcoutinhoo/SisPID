# pylint: disable="C0114, C0103, C0301"

"""
Interface gráfica simples para visualizar resultados PID.
"""

from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class PIDResultsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Resultados PID - Análise")
        self.root.geometry("1200x700")
        
        self.db_name = "db/pid_results.db"
        self.primeira_carga = True  # Flag para controlar primeira carga
        
        # Configurar layout
        self.setup_ui()
        self.carregar_dados()
    
    def setup_ui(self):
        """Configura a interface."""
        
        # Frame superior - Tabela
        frame_tabela = ttk.LabelFrame(self.root, text="Comparação de Métodos", padding=10)
        frame_tabela.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollbar para tabela
        scroll_y = ttk.Scrollbar(frame_tabela, orient=tk.VERTICAL)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview (tabela)
        colunas = ("Método", "MSE", "Overshoot (%)", "Ts (s)", "MG (dB)", "MF (°)", "Testes")
        self.tree = ttk.Treeview(frame_tabela, columns=colunas, show="headings", 
                                  yscrollcommand=scroll_y.set, height=8)
        scroll_y.config(command=self.tree.yview)
        
        # Configurar colunas
        larguras = [100, 120, 120, 100, 100, 100, 80]
        for col, largura in zip(colunas, larguras):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=largura, anchor=tk.CENTER)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Frame inferior - Gráficos e análise
        frame_inferior = tk.Frame(self.root)
        frame_inferior.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Frame esquerdo - Análise
        frame_analise = ttk.LabelFrame(frame_inferior, text="Análise", padding=10)
        frame_analise.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.texto_analise = tk.Text(frame_analise, height=10, width=40, 
                                      font=("Courier", 10), bg="#f0f0f0")
        self.texto_analise.pack(fill=tk.BOTH, expand=True)
        
        # Frame direito - Botões e gráfico
        frame_direita = tk.Frame(frame_inferior)
        frame_direita.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Frame de botões - LINHA 1
        frame_botoes1 = ttk.LabelFrame(frame_direita, text="Visualizações", padding=10)
        frame_botoes1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(frame_botoes1, text="📊 Gráfico MSE", 
                   command=self.plot_mse).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botoes1, text="📈 Gráfico Overshoot", 
                   command=self.plot_overshoot).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(frame_botoes1, text="❌ Sair", 
                   command=self.root.quit).pack(side=tk.RIGHT, padx=5)
        ttk.Button(frame_botoes1, text="🔄 Atualizar", 
                   command=lambda: self.carregar_dados(plotar_grafico=False)).pack(side=tk.RIGHT, padx=5)
        
        # Frame de botões - LINHA 2
        frame_botoes2 = ttk.LabelFrame(frame_direita, text="Análises Avançadas", padding=10)
        frame_botoes2.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(frame_botoes2, text="⚡ Respostas Temporais", 
                   command=self.plot_respostas_temporais).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botoes2, text="📉 Regime Permanente", 
                   command=self.plot_regime_permanente).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botoes2, text="🧬 Evolução Métodos", 
                   command=self.plot_evolucao_metodos).pack(side=tk.LEFT, padx=5)
        
        # Frame para gráfico pequeno
        self.frame_grafico = ttk.LabelFrame(frame_direita, text="Comparação Visual", padding=5)
        self.frame_grafico.pack(fill=tk.BOTH, expand=True)
    
    def carregar_dados(self, plotar_grafico=None):
        """Carrega dados do banco e atualiza interface."""
        try:
            conn = sqlite3.connect(self.db_name)
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
            
            # Limpa tabela
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Preenche tabela
            if resultados:
                for row in resultados:
                    metodo, mse, overshoot, ts, mg, mf, n = row
                    
                    if mg and mg > 900:
                        mg_str = "∞"
                    elif mg:
                        mg_str = f"{mg:.2f}"
                    else:
                        mg_str = "N/A"
                    
                    mf_str = f"{mf:.2f}" if mf else "N/A"
                    
                    self.tree.insert("", tk.END, values=(
                        metodo,
                        f"{mse:.6f}",
                        f"{overshoot:.2f}",
                        f"{ts:.2f}",
                        mg_str,
                        mf_str,
                        n
                    ))
                
                # Atualiza análise
                self.atualizar_analise(resultados)
                
                # Plota gráfico inicial apenas na primeira vez
                plotar_grafico = (plotar_grafico is None and self.primeira_carga)
                
                if plotar_grafico:
                    self.plot_comparacao(resultados)
                #     self.primeira_carga = False
            else:
                messagebox.showinfo("Info", "Nenhum resultado encontrado no banco!")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar dados: {e}")
    
    def atualizar_analise(self, resultados):
        """Atualiza texto de análise."""
        self.texto_analise.delete(1.0, tk.END)
        
        if not resultados:
            return
        
        texto =  "╔═══════════════════════════════════════╗\n"
        texto += "║         ANÁLISE DE RESULTADOS         ║\n"
        texto += "╚═══════════════════════════════════════╝\n\n"
        
        # Melhor MSE
        melhor_mse = min(resultados, key=lambda x: x[1])
        texto += f"🏆 MELHOR DESEMPENHO (MSE):\n"
        texto += f"   → {melhor_mse[0]}\n"
        texto += f"   MSE: {melhor_mse[1]:.6f}\n\n"
        
        # Menor Overshoot
        menor_os = min(resultados, key=lambda x: x[2])
        texto += f"📉 MENOR OVERSHOOT:\n"
        texto += f"   → {menor_os[0]}\n"
        texto += f"   Overshoot: {menor_os[2]:.2f}%\n\n"
        
        # Robustez
        texto += "🛡️ ROBUSTEZ:\n"
        metodos_robustos = [r for r in resultados if r[4] and r[5]]
        if metodos_robustos:
            melhor_mg = max(metodos_robustos, key=lambda x: x[4] if x[4] < 900 else 0)
            melhor_mf = max(metodos_robustos, key=lambda x: x[5])
            
            if melhor_mg[4] > 900:
                texto += f"   Maior MG: Todos (∞ dB)\n"
            else:
                texto += f"   Maior MG: {melhor_mg[0]} ({melhor_mg[4]:.2f} dB)\n"
            
            texto += f"   Maior MF: {melhor_mf[0]} ({melhor_mf[5]:.2f}°)\n"
        else:
            texto += "   Dados não disponíveis\n"
        
        texto += "\n" + "─" * 40 + "\n"
        texto += f"📊 Total de métodos: {len(resultados)}\n"
        texto += f"🔬 Total de testes: {sum(r[6] for r in resultados)}"
        
        self.texto_analise.insert(1.0, texto)
    
    def plot_comparacao(self, resultados):
        plt.close('all')
        """Plota gráfico de barras comparativo."""
        # Limpa gráfico anterior
        for widget in self.frame_grafico.winfo_children():
            widget.destroy()
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
        
        metodos = [r[0] for r in resultados]
        mse_values = [r[1] for r in resultados]
        
        cores = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(metodos))]
        
        bars = ax.bar(metodos, mse_values, color=cores, alpha=0.8, edgecolor='black')
        
        # Destaca o melhor
        bars[0].set_color('#27ae60')
        bars[0].set_linewidth(2)
        
        ax.set_ylabel('MSE', fontweight='bold')
        ax.set_title('Comparação de Desempenho (MSE)', fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Adicionar à interface
        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_mse(self):
        plt.close('all')
        """Plota gráfico detalhado de MSE."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT metodo, AVG(mse) as mse_avg
                FROM resultados
                GROUP BY metodo
                ORDER BY mse_avg
            """)
            
            resultados = cursor.fetchall()
            conn.close()
            
            if not resultados:
                messagebox.showinfo("Info", "Sem dados para plotar!")
                return
            
            # Nova janela
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metodos = [r[0] for r in resultados]
            mse_values = [r[1] for r in resultados]
            
            cores = plt.cm.viridis(np.linspace(0.3, 0.9, len(metodos)))
            bars = ax.bar(metodos, mse_values, color=cores, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.6f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_ylabel('MSE (Erro Quadrático Médio)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Método de Sintonia', fontsize=12, fontweight='bold')
            ax.set_title('Comparação de Desempenho - MSE', fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar gráfico: {e}")
    
    def plot_overshoot(self):
        plt.close('all')
        """Plota gráfico detalhado de Overshoot."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT metodo, AVG(overshoot) as os_avg, AVG(tempo_acomodacao) as ts_avg
                FROM resultados
                GROUP BY metodo
                ORDER BY os_avg
            """)
            
            resultados = cursor.fetchall()
            conn.close()
            
            if not resultados:
                messagebox.showinfo("Info", "Sem dados para plotar!")
                return
            
            # Nova janela com subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            metodos = [r[0] for r in resultados]
            os_values = [r[1] for r in resultados]
            ts_values = [r[2] for r in resultados]
            
            # Gráfico Overshoot
            cores1 = plt.cm.Reds(np.linspace(0.4, 0.8, len(metodos)))
            bars1 = ax1.bar(metodos, os_values, color=cores1, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax1.set_ylabel('Overshoot (%)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Método', fontsize=12, fontweight='bold')
            ax1.set_title('Overshoot Médio', fontsize=13, fontweight='bold', pad=15)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            ax1.set_axisbelow(True)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Gráfico Tempo de Acomodação
            cores2 = plt.cm.Blues(np.linspace(0.4, 0.8, len(metodos)))
            bars2 = ax2.bar(metodos, ts_values, color=cores2, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}s',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax2.set_ylabel('Tempo de Acomodação (s)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Método', fontsize=12, fontweight='bold')
            ax2.set_title('Tempo de Acomodação Médio', fontsize=13, fontweight='bold', pad=15)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            ax2.set_axisbelow(True)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar gráfico: {e}")
    
    def plot_regime_permanente(self):
        plt.close('all')
        """Plota gráfico de regime permanente para todos os métodos."""
        try:
            import control as ctl
            
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Buscar todos os métodos salvos
            cursor.execute('SELECT DISTINCT metodo FROM resultados')
            metodos = [row[0] for row in cursor.fetchall()]
            
            if not metodos:
                messagebox.showinfo("Info", "Nenhum método encontrado no banco!")
                conn.close()
                return
            
            # Configurar o gráfico
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Cores para cada método
            cores = {
                'CC': 'blue',
                'CMA-ES': 'orange', 
                'DE': 'green',
                'GA': 'cyan',
                'PSO': 'red',
                'ZN1': 'purple',
                'ZN2': 'magenta'
            }
            
            # Definir o tempo de início do regime permanente
            tempo_inicio_regime = 160  # segundos
            
            # Para cada método, simular a resposta
            for metodo in metodos:
                # Buscar os parâmetros PID do método
                cursor.execute('''
                    SELECT Kp, Ki, Kd 
                    FROM resultados 
                    WHERE metodo = ?
                    ORDER BY data_hora DESC
                    LIMIT 1
                ''', (metodo,))
                
                resultado = cursor.fetchone()
                
                if resultado:
                    Kp, Ki, Kd = resultado
                    
                    # Definir a planta - G(s) = 59.81/(401.61s + 1)
                    Kterm = 59.81
                    tau = 401.61
                    plant = ctl.tf([Kterm], [tau, 1])
                    
                    # Criar controlador PID
                    pid_tf = ctl.tf([Kd, Kp, Ki], [1, 0])
                    
                    # Sistema em malha fechada
                    sys_mf = ctl.feedback(pid_tf * plant, 1)
                    
                    # Simular resposta ao degrau
                    t = np.linspace(0, 200, 2000)  # 0 a 200s
                    t_out, y_out = ctl.step_response(sys_mf, t)
                    
                    # Filtrar apenas regime permanente
                    mask = t_out >= tempo_inicio_regime
                    tempos_regime = t_out[mask]
                    temp_regime = y_out[mask] * 80  # Escalar para setpoint de 80°C
                    
                    # Plotar
                    cor = cores.get(metodo, 'gray')
                    ax.plot(tempos_regime, temp_regime, label=metodo, color=cor, linewidth=2)
            
            conn.close()
            
            # Linha do setpoint
            ax.axhline(y=80, color='black', linestyle='--', linewidth=2, label='Setpoint (80°C)')
            
            # Adicionar banda de ±2%
            banda_percentual = 0.02
            ax.axhline(y=80*(1+banda_percentual), color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(y=80*(1-banda_percentual), color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.fill_between([tempo_inicio_regime, 200], 
                            80*(1-banda_percentual), 80*(1+banda_percentual), 
                            color='green', alpha=0.1, label='Banda ±2%')
            
            # Configurações do gráfico
            ax.set_xlabel('Tempo (s)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Temperatura (°C)', fontsize=12, fontweight='bold')
            ax.set_title(f'Resposta em Regime Permanente (após {tempo_inicio_regime} s)', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([75, 85])
            
            plt.tight_layout()
            plt.show()
        
            print(f"\n Gráfico gerado mostrando o regime permanente após {tempo_inicio_regime}s")
            
        except ImportError:
            messagebox.showerror("Erro", "Biblioteca 'control' não encontrada! Instale com: pip install control")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar gráfico de regime permanente: {e}")


    def plot_respostas_temporais(self):
        plt.close('all')
        """Gera gráfico das respostas temporais com foco no transitório inicial."""
        try:
            import control as ctl
            
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Buscar dados de todos os métodos
            cursor.execute('''
                SELECT metodo, Kp, Ki, Kd, overshoot
                FROM resultados
                GROUP BY metodo
                ORDER BY metodo
            ''')
            
            resultados = cursor.fetchall()
            conn.close()
            
            if not resultados:
                messagebox.showinfo("Info", "Nenhum método encontrado no banco!")
                return
            
            # Preparar dados
            dados = {}
            for metodo, kp, ki, kd, overshoot in resultados:
                dados[metodo] = {
                    'Kp': kp,
                    'Ki': ki,
                    'Kd': kd,
                    'Overshoot': overshoot
                }
            
            # Cores para cada método
            CORES = {
                'ZN1': '#1f77b4',
                'ZN2': '#ff1493',
                'CC': '#ff7f0e',
                'GA': '#2ca02c',
                'PSO': '#d62728',
                'DE': '#9467bd',
                'CMA-ES': '#8c564b'
            }
            
            # Parâmetros
            Kterm = 59.81
            tau = 401.61
            plant = ctl.tf([Kterm], [tau, 1])
            t = np.linspace(0, 2*tau, 1000)
            setpoint = 80.0
            
            # Criar figura com 2 subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # SUBPLOT 1: Resposta completa
            for metodo in dados.keys():
                d = dados[metodo]
                
                # Criar controlador PID
                pid_tf = ctl.tf([d['Kd'], d['Kp'], d['Ki']], [1, 0])
                
                # Sistema em malha fechada
                sys_mf = ctl.feedback(pid_tf * plant, 1)
                
                # Simular resposta
                t_out, y_out = ctl.step_response(sys_mf, t)
                y = y_out * setpoint
                
                cor = CORES.get(metodo, 'gray')
                ax1.plot(t_out, y, color=cor, linewidth=2.5, 
                        label=f"{metodo} (OS: {d['Overshoot']:.1f}%)", alpha=0.85)
            
            ax1.axhline(setpoint, color='red', linestyle='--', linewidth=2, 
                       label='Setpoint (80°C)', alpha=0.7)
            ax1.set_xlabel('Tempo (s)', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Temperatura (°C)', fontweight='bold', fontsize=12)
            ax1.set_title('(a) Resposta Completa', fontweight='bold', fontsize=13, pad=15)
            ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 2*tau)
            
            # SUBPLOT 2: Zoom no transitório (primeiros 20%)
            t_max_zoom = (2*tau) * 0.2  # ~160s
            
            for metodo in dados.keys():
                d = dados[metodo]
                
                # Criar controlador PID
                pid_tf = ctl.tf([d['Kd'], d['Kp'], d['Ki']], [1, 0])
                
                # Sistema em malha fechada
                sys_mf = ctl.feedback(pid_tf * plant, 1)
                
                # Simular resposta
                t_out, y_out = ctl.step_response(sys_mf, t)
                y = y_out * setpoint
                
                # Filtrar apenas primeiros 20%
                mask = t_out <= t_max_zoom
                t_zoom = t_out[mask]
                y_zoom = y[mask]
                
                cor = CORES.get(metodo, 'gray')
                ax2.plot(t_zoom, y_zoom, color=cor, linewidth=2.5, 
                        label=f"{metodo}", alpha=0.85)
                
                # Marcar pico
                idx_pico = np.argmax(y_zoom)
                ax2.plot(t_zoom[idx_pico], y_zoom[idx_pico], 'o', 
                        color=cor, markersize=7, 
                        markeredgecolor='black', markeredgewidth=1)
            
            ax2.axhline(setpoint, color='red', linestyle='--', linewidth=2, 
                       label='Setpoint', alpha=0.7)
            ax2.set_xlabel('Tempo (s)', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Temperatura (°C)', fontweight='bold', fontsize=12)
            ax2.set_title(f'(b) Zoom no Transitório (0-{t_max_zoom:.0f}s)', 
                         fontweight='bold', fontsize=13, pad=15)
            ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, t_max_zoom)
            
            # Título geral
            fig.suptitle('Comparação das Respostas Temporais de Todos os Métodos', 
                         fontweight='bold', fontsize=14, y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
            
            print(" Gráfico de respostas temporais gerado")
            print(f"  - Subplot (a): Resposta completa (0-{2*tau:.0f}s)")
            print(f"  - Subplot (b): Zoom no transitório (0-{t_max_zoom:.0f}s) com marcação dos picos")
            
        except ImportError:
            messagebox.showerror("Erro", "Biblioteca 'control' não encontrada! Instale com: pip install control")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar gráfico de respostas temporais: {e}")
        
    def plot_evolucao_metodos(self):
        plt.close('all')
        """Plota evolução dos métodos evolutivos ao longo das gerações."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Buscar métodos evolutivos disponíveis
            cursor.execute("""
                SELECT DISTINCT metodo 
                FROM historico_evolutivo
                ORDER BY metodo
            """)
            
            metodos = [row[0] for row in cursor.fetchall()]
            
            if not metodos:
                messagebox.showinfo("Info", "Nenhum histórico evolutivo encontrado!")
                conn.close()
                return
            
            # Criar figura com mais espaço
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Cores modernas e distintas
            cores = {
                'PSO': '#e74c3c',      # Vermelho vibrante
                'GA': '#2ecc71',       # Verde esmeralda
                'DE': '#9b59b6',       # Roxo
                'CMA-ES': '#f39c12'    # Laranja
            }
            
            # Símbolos diferentes para cada método
            markers = {
                'PSO': 'o',
                'GA': 's',
                'DE': '^',
                'CMA-ES': 'D'
            }
            
            # ===== SUBPLOT 1: Convergência (Melhor Fitness) =====
            
            for metodo in metodos:
                cursor.execute("""
                    SELECT geracao, melhor_fitness
                    FROM historico_evolutivo
                    WHERE metodo = ?
                    ORDER BY geracao
                """, (metodo,))
                
                dados = cursor.fetchall()
                geracoes = [d[0] for d in dados]
                fitness = [d[1] for d in dados]
                
                cor = cores.get(metodo, 'gray')
                marker = markers.get(metodo, 'o')
                
                ax1.plot(geracoes, fitness, color=cor, linewidth=3, 
                        label=metodo, marker=marker, markersize=6, 
                        markevery=max(1, len(geracoes)//10),
                        alpha=0.9, markeredgecolor='white', markeredgewidth=1.5)
            
            ax1.set_xlabel('Geração', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Melhor Fitness (MSE)', fontweight='bold', fontsize=12)
            ax1.set_title('(a) Convergência - Evolução do Melhor Indivíduo', 
                         fontweight='bold', fontsize=13, pad=12)
            ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
            ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax1.set_yscale('linear')
            ax1.set_facecolor('#f8f9fa')
            
            # ===== SUBPLOT 2: Fitness Médio =====
            
            for metodo in metodos:
                cursor.execute("""
                    SELECT geracao, fitness_medio
                    FROM historico_evolutivo
                    WHERE metodo = ?
                    ORDER BY geracao
                """, (metodo,))
                
                dados = cursor.fetchall()
                geracoes = [d[0] for d in dados]
                fitness = [d[1] for d in dados]
                
                cor = cores.get(metodo, 'gray')
                marker = markers.get(metodo, 'o')
                
                ax2.plot(geracoes, fitness, color=cor, linewidth=2.5, 
                        label=metodo, marker=marker, markersize=5,
                        markevery=max(1, len(geracoes)//10),
                        alpha=0.85, markeredgecolor='white', markeredgewidth=1)
            
            ax2.set_xlabel('Geração', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Fitness Médio (MSE)', fontweight='bold', fontsize=12)
            ax2.set_title('(b) Fitness Médio da População', 
                         fontweight='bold', fontsize=13, pad=12)
            ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
            ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax2.set_yscale('linear')
            ax2.set_facecolor('#f8f9fa')
            
            conn.close()
            
            # Título geral com estilo
            fig.suptitle('Evolução dos Algoritmos Evolutivos ao Longo das Gerações', 
                         fontweight='bold', fontsize=16, y=0.995)           
            plt.show()
            
            print("\n Gráfico de evolução gerado com sucesso!")
            print(f"    Métodos analisados: {', '.join(metodos)}")
            
        except ImportError:
            messagebox.showerror("Erro", "Biblioteca necessária não encontrada!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar gráfico de evolução: {e}")


def main():
    root = tk.Tk()
    app = PIDResultsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()