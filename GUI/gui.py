# pylint: disable="C0114, C0103, C0301"

"""
Interface gráfica para visualizar resultados PID com análise de robustez.
"""

from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from modules.zn_module import ziegler_nichols_1
from modules.cc_module import cohen_coon
from modules.pso_module import tune_pid_pso
from modules.ga_module import tune_pid_ga
from modules.de_module import tune_pid_de
from modules.cma_module import tune_pid_cma
from modules.statistics_module import teste_friedman, gerar_resumo_estatistico, obter_dados_para_grafico
from main import print_PID_params


class PIDResultsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Resultados PID - Análise Completa")
        self.root.geometry("1400x800")
        
        self.db_name = "db/pid_results.db"
        self.primeira_carga = True
        
        # Configurar layout
        self.setup_ui()
        self.carregar_dados()
    
    def setup_ui(self):
        """Configura a interface."""
    
        # Notebook (abas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ABA 0: Configuração e Execução (NOVA)
        self.aba_config = ttk.Frame(self.notebook)
        self.notebook.add(self.aba_config, text="⚙️ Configuração & Execução")
        self.setup_aba_config()
        
        # ABA 1: Desempenho Nominal
        self.aba_nominal = ttk.Frame(self.notebook)
        self.notebook.add(self.aba_nominal, text="📊 Desempenho Nominal")
        self.setup_aba_nominal()
        
        # ABA 2: Análise de Robustez
        self.aba_robustez = ttk.Frame(self.notebook)
        self.notebook.add(self.aba_robustez, text="🛡️ Robustez Paramétrica")
        self.setup_aba_robustez()
        
        # ABA 3: Visualizações Avançadas
        self.aba_graficos = ttk.Frame(self.notebook)
        self.notebook.add(self.aba_graficos, text="📈 Gráficos Avançados")
        self.setup_aba_graficos()

        # ABA 4: Análise Estatística 
        self.aba_estatistica = ttk.Frame(self.notebook)
        self.notebook.add(self.aba_estatistica, text="📊 Análise Estatística")
        self.setup_aba_estatistica()

    def setup_aba_config(self):
        """Configura aba de configuração e execução."""
        
        # ===== SEÇÃO 1: PARÂMETROS DA PLANTA =====
        frame_planta = ttk.LabelFrame(self.aba_config, text="Parâmetros da Planta Térmica (Equação 4)", padding=15)
        frame_planta.pack(fill=tk.X, padx=10, pady=10)
        
        # Informação sobre a equação
        info_text = "G(s) = K_Term / (τ·s + 1)"
        ttk.Label(frame_planta, text=info_text, font=("Courier", 11, "bold"), 
                foreground="blue").grid(row=0, column=0, columnspan=4, pady=(0, 10))
        
        # K_Term
        ttk.Label(frame_planta, text="K_Term (°C/W):", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_k_term = ttk.Entry(frame_planta, width=15, font=("Arial", 10))
        self.entry_k_term.insert(0, "59.81")
        self.entry_k_term.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(frame_planta, text="Ganho térmico da estufa", 
                foreground="gray").grid(row=1, column=2, sticky=tk.W, padx=5)
        
        # τ (tau)
        ttk.Label(frame_planta, text="τ (s):", font=("Arial", 10, "bold")).grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_tau = ttk.Entry(frame_planta, width=15, font=("Arial", 10))
        self.entry_tau.insert(0, "401.61")
        self.entry_tau.grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(frame_planta, text="Constante de tempo (inércia térmica)", 
                foreground="gray").grid(row=2, column=2, sticky=tk.W, padx=5)
        
        # Botão para carregar perfis pré-definidos
        ttk.Button(frame_planta, text="📋 Perfis Pré-definidos", 
                command=self.mostrar_perfis).grid(row=1, column=3, rowspan=2, padx=10)
        
        # ===== SEÇÃO 2: PARÂMETROS DE SIMULAÇÃO =====
        frame_sim = ttk.LabelFrame(self.aba_config, text="Parâmetros de Simulação", padding=15)
        frame_sim.pack(fill=tk.X, padx=10, pady=10)
        
        # Setpoint
        ttk.Label(frame_sim, text="Setpoint (°C):", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_setpoint = ttk.Entry(frame_sim, width=15, font=("Arial", 10))
        self.entry_setpoint.insert(0, "80.0")
        self.entry_setpoint.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(frame_sim, text="Temperatura desejada", 
                foreground="gray").grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # Tempo de simulação
        ttk.Label(frame_sim, text="Tempo Final (s):", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_tempo_final = ttk.Entry(frame_sim, width=15, font=("Arial", 10))
        self.entry_tempo_final.insert(0, "803.22")
        self.entry_tempo_final.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(frame_sim, text="Duração da simulação (padrão: 2τ)", 
                foreground="gray").grid(row=1, column=2, sticky=tk.W, padx=5)
        
        # Número de pontos
        ttk.Label(frame_sim, text="Pontos:", font=("Arial", 10, "bold")).grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_pontos = ttk.Entry(frame_sim, width=15, font=("Arial", 10))
        self.entry_pontos.insert(0, "1000")
        self.entry_pontos.grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(frame_sim, text="Resolução da simulação", 
                foreground="gray").grid(row=2, column=2, sticky=tk.W, padx=5)
        
        # Checkbox para calcular automaticamente 2τ
        self.var_auto_tempo = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_sim, text="Calcular automaticamente como 2τ", 
                        variable=self.var_auto_tempo,
                        command=self.atualizar_tempo_automatico).grid(
                            row=3, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # ===== SEÇÃO 3: SELEÇÃO DE MÉTODOS =====
        frame_metodos = ttk.LabelFrame(self.aba_config, text="Métodos de Sintonia", padding=15)
        frame_metodos.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(frame_metodos, text="Selecione os métodos a executar:", 
                font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Checkboxes para cada método
        self.var_zn1 = tk.BooleanVar(value=True)
        self.var_cc = tk.BooleanVar(value=True)
        self.var_ga = tk.BooleanVar(value=True)
        self.var_pso = tk.BooleanVar(value=True)
        self.var_de = tk.BooleanVar(value=True)
        self.var_cma = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(frame_metodos, text="Ziegler-Nichols (Curva de Reação)", 
                        variable=self.var_zn1).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frame_metodos, text="Cohen-Coon", 
                        variable=self.var_cc).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frame_metodos, text="Algoritmo Genético (GA)", 
                        variable=self.var_ga).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frame_metodos, text="PSO (Enxame de Partículas)", 
                        variable=self.var_pso).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frame_metodos, text="Evolução Diferencial (DE)", 
                        variable=self.var_de).grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(frame_metodos, text="CMA-ES", 
                        variable=self.var_cma).grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        
        # Botões de seleção rápida
        frame_botoes_sel = ttk.Frame(frame_metodos)
        frame_botoes_sel.grid(row=3, column=0, columnspan=3, pady=10)
        ttk.Button(frame_botoes_sel, text="Selecionar Todos", 
                command=self.selecionar_todos).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botoes_sel, text="Desselecionar Todos", 
                command=self.desselecionar_todos).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botoes_sel, text="Apenas Heurísticos", 
                command=self.apenas_heuristicos).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botoes_sel, text="Apenas Evolutivos", 
                command=self.apenas_evolutivos).pack(side=tk.LEFT, padx=5)
        
        # ===== SEÇÃO 4: CONFIGURAÇÕES DE EXECUÇÃO =====
        frame_exec = ttk.LabelFrame(self.aba_config, text="Configurações de Execução", padding=15)
        frame_exec.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(frame_exec, text="Número de iterações:", 
                font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_iteracoes = ttk.Entry(frame_exec, width=10, font=("Arial", 10))
        self.entry_iteracoes.insert(0, "15")
        self.entry_iteracoes.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(frame_exec, text="Cada método será executado N vezes", 
                foreground="gray").grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # Checkbox para análise de robustez
        self.var_robustez = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_exec, text="Executar análise de robustez (variações paramétricas)", 
                        variable=self.var_robustez).grid(row=1, column=0, columnspan=3, 
                                                sticky=tk.W, padx=5, pady=5)
        
        # ===== SEÇÃO 5: BOTÕES DE AÇÃO =====
        frame_acoes = ttk.Frame(self.aba_config)
        frame_acoes.pack(fill=tk.X, padx=10, pady=20)
        
        # Botão principal de execução
        self.btn_executar = ttk.Button(frame_acoes, text="EXECUTAR SIMULAÇÕES", 
                                        command=self.executar_simulacoes,
                                        style="Accent.TButton")
        self.btn_executar.pack(side=tk.LEFT, padx=5, ipadx=20, ipady=10)

        # Limpa banco de dados
        self.btn_limpar_db = ttk.Button(frame_acoes, text="Limpar Banco", 
                                         command=self.limpar_banco_dados)
        self.btn_limpar_db.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=10)
        
        # Barra de progresso
        self.progress = ttk.Progressbar(frame_acoes, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Label de status
        self.label_status = ttk.Label(frame_acoes, text="Pronto para executar", 
                                    font=("Arial", 10))
        self.label_status.pack(side=tk.LEFT, padx=5)
        
        # ===== SEÇÃO 6: LOG DE EXECUÇÃO =====
        frame_log = ttk.LabelFrame(self.aba_config, text="Log de Execução", padding=10)
        frame_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbar para log
        scroll_log = ttk.Scrollbar(frame_log, orient=tk.VERTICAL)
        scroll_log.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.text_log = tk.Text(frame_log, height=10, font=("Courier", 9), 
                                bg="#1e1e1e", fg="#00ff00", 
                                yscrollcommand=scroll_log.set)
        scroll_log.config(command=self.text_log.yview)
        self.text_log.pack(fill=tk.BOTH, expand=True)
        
        # Mensagem inicial
        self.log("="*70)
        self.log("SISTEMA DE ANÁLISE COMPARATIVA DE MÉTODOS DE SINTONIA PID")
        self.log("="*70)
        self.log("Configure os parâmetros acima e clique em EXECUTAR SIMULAÇÕES")
        self.log("")

    def log(self, mensagem):
        """Adiciona mensagem ao log."""
        self.text_log.insert(tk.END, f"{mensagem}\n")
        self.text_log.see(tk.END)
        self.text_log.update()

    def atualizar_tempo_automatico(self):
        """Atualiza tempo final baseado em 2τ se checkbox marcado."""
        if self.var_auto_tempo.get():
            try:
                tau = float(self.entry_tau.get())
                tempo_final = 2 * tau
                self.entry_tempo_final.delete(0, tk.END)
                self.entry_tempo_final.insert(0, f"{tempo_final:.2f}")
                self.entry_tempo_final.config(state='disabled')
            except ValueError:
                pass
        else:
            self.entry_tempo_final.config(state='normal')

    def mostrar_perfis(self):
        """Mostra janela com perfis pré-definidos de plantas."""
        perfis = {
            "Estufa Padrão": {"K_Term": 59.81, "tau": 401.61},
            "Estufa Rápida": {"K_Term": 80.0, "tau": 250.0},
            "Estufa Lenta": {"K_Term": 45.0, "tau": 600.0},
            "Forno Industrial": {"K_Term": 120.0, "tau": 180.0},
            "Incubadora": {"K_Term": 35.0, "tau": 300.0},
        }
        
        janela = tk.Toplevel(self.root)
        janela.title("Perfis Pré-definidos")
        janela.geometry("400x300")
        
        ttk.Label(janela, text="Selecione um perfil:", font=("Arial", 11, "bold")).pack(pady=10)
        
        for nome, params in perfis.items():
            btn = ttk.Button(janela, text=f"{nome} (K={params['K_Term']}, τ={params['tau']})",
                            command=lambda p=params: self.carregar_perfil(p, janela))
            btn.pack(fill=tk.X, padx=20, pady=5)

    def carregar_perfil(self, params, janela):
        """Carrega perfil selecionado."""
        self.entry_k_term.delete(0, tk.END)
        self.entry_k_term.insert(0, str(params['K_Term']))
        self.entry_tau.delete(0, tk.END)
        self.entry_tau.insert(0, str(params['tau']))
        self.atualizar_tempo_automatico()
        janela.destroy()
        self.log(f"✓ Perfil carregado: K_Term={params['K_Term']}, τ={params['tau']}")

    def selecionar_todos(self):
        """Seleciona todos os métodos."""
        for var in [self.var_zn1, self.var_cc, self.var_ga, self.var_pso, self.var_de, self.var_cma]:
            var.set(True)

    def desselecionar_todos(self):
        """Desseleciona todos os métodos."""
        for var in [self.var_zn1, self.var_cc, self.var_ga, self.var_pso, self.var_de, self.var_cma]:
            var.set(False)

    def apenas_heuristicos(self):
        """Seleciona apenas métodos heurísticos."""
        self.var_zn1.set(True)
        self.var_cc.set(True)
        self.var_ga.set(False)
        self.var_pso.set(False)
        self.var_de.set(False)
        self.var_cma.set(False)

    def apenas_evolutivos(self):
        """Seleciona apenas métodos evolutivos."""
        self.var_zn1.set(False)
        self.var_cc.set(False)
        self.var_ga.set(True)
        self.var_pso.set(True)
        self.var_de.set(True)
        self.var_cma.set(True)

    def executar_simulacoes(self):
        """Executa simulações com os parâmetros configurados."""
        try:
            # Validar entradas
            k_term = float(self.entry_k_term.get())
            tau = float(self.entry_tau.get())
            setpoint = float(self.entry_setpoint.get())
            t_final = float(self.entry_tempo_final.get())
            n_pontos = int(self.entry_pontos.get())
            iteracoes = int(self.entry_iteracoes.get())

            self.k_term_atual = k_term
            self.tau_atual = tau
            self.setpoint_atual = setpoint
            self.t_final_atual = t_final
            self.n_pontos_atual = n_pontos
            
            if k_term <= 0 or tau <= 0:
                messagebox.showerror("Erro", "K_Term e τ devem ser positivos!")
                return
            
            if n_pontos < 100:
                messagebox.showerror("Erro", "Número de pontos deve ser >= 100!")
                return
            
            metodos_selecionados = {}
            if self.var_zn1.get():
                metodos_selecionados['ZN1'] = ziegler_nichols_1
            if self.var_cc.get():
                metodos_selecionados['CC'] = cohen_coon
            if self.var_ga.get():
                metodos_selecionados['GA'] = tune_pid_ga
            if self.var_pso.get():
                metodos_selecionados['PSO'] = tune_pid_pso
            if self.var_de.get():
                metodos_selecionados['DE'] = tune_pid_de
            if self.var_cma.get():
                metodos_selecionados['CMA-ES'] = tune_pid_cma
            
            if not metodos_selecionados:
                messagebox.showwarning("Aviso", "Selecione pelo menos um método!")
                return
            
            # Confirmar execução
            msg = f"Executar {len(metodos_selecionados)} métodos, {iteracoes} iterações cada?\n\n"
            msg += f"Planta: K_Term={k_term}, τ={tau}\n"
            msg += f"Simulação: {t_final}s, {n_pontos} pontos, setpoint={setpoint}°C"
            
            if not messagebox.askyesno("Confirmar Execução", msg):
                return
            
            # Desabilitar botão e iniciar progresso
            self.btn_executar.config(state='disabled')
            self.progress.start(10)
            self.label_status.config(text="Executando...")
            
            self.log("\n" + "="*70)
            self.log(f"NOVA EXECUÇÃO INICIADA")
            self.log(f"Planta: K_Term={k_term} °C/W, τ={tau} s")
            self.log(f"Setpoint: {setpoint}°C, Tempo: {t_final}s, Pontos: {n_pontos}")
            self.log(f"Métodos: {', '.join(metodos_selecionados.keys())}")
            self.log(f"Iterações por método: {iteracoes}")
            self.log("="*70)
            
            from main import executar_sintonia
            
            try:
                pid_params = executar_sintonia(
                    k_term=k_term,
                    tau=tau,
                    setpoint=setpoint,
                    t_final=t_final,
                    n_pontos=n_pontos,
                    metodos_selecionados=metodos_selecionados,
                    iteracoes=iteracoes,
                    executar_robustez=self.var_robustez.get(),
                    db_path=self.db_name
                )
                
                # Finalizar
                self.progress.stop()
                self.btn_executar.config(state='normal')
                self.label_status.config(text="✓ Concluído!")
                
                self.log(f"\n{'='*70}")
                self.log(f"✓ EXECUÇÃO CONCLUÍDA COM SUCESSO")
                self.log(f"{'='*70}\n")
                
                # Atualizar dados nas outras abas
                self.carregar_dados()
                
                messagebox.showinfo("Sucesso", 
                                f"Simulações concluídas!\n\n"
                                f"Métodos executados: {len(metodos_selecionados)}\n"
                                f"Iterações por método: {iteracoes}\n"
                                f"Total de simulações: {len(metodos_selecionados) * iteracoes}\n\n"
                                f"Veja os resultados nas outras abas.")
            
            except Exception as e:
                self.log(f"\n✗ ERRO DURANTE EXECUÇÃO: {str(e)}")
                raise
            
        except ValueError as e:
            messagebox.showerror("Erro de Validação", 
                            f"Valores inválidos nos parâmetros:\n{str(e)}")
            self.btn_executar.config(state='normal')
            self.progress.stop()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro durante execução:\n{str(e)}")
            self.btn_executar.config(state='normal')
            self.progress.stop()
            self.label_status.config(text="✗ Erro na execução")

    def setup_aba_nominal(self):
        """Configura aba de desempenho nominal."""
        
        # Frame superior - Tabela
        frame_tabela = ttk.LabelFrame(self.aba_nominal, text="Comparação de Métodos - Condições Nominais", padding=10)
        frame_tabela.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollbar
        scroll_y = ttk.Scrollbar(frame_tabela, orient=tk.VERTICAL)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview
        colunas = ("Método", "MSE", "Overshoot (%)", "Ts (s)", "MG (dB)", "MF (°)", "Testes")
        self.tree_nominal = ttk.Treeview(frame_tabela, columns=colunas, show="headings", 
                                  yscrollcommand=scroll_y.set, height=8)
        scroll_y.config(command=self.tree_nominal.yview)
        
        larguras = [100, 120, 120, 100, 100, 100, 80]
        for col, largura in zip(colunas, larguras):
            self.tree_nominal.heading(col, text=col)
            self.tree_nominal.column(col, width=largura, anchor=tk.CENTER)
        
        self.tree_nominal.pack(fill=tk.BOTH, expand=True)
        
        # Frame inferior
        frame_inferior = tk.Frame(self.aba_nominal)
        frame_inferior.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Frame para Parâmetros PID
        frame_params = ttk.LabelFrame(frame_inferior, text="Parâmetros PID", padding=10)
        frame_params.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.texto_params = tk.Text(frame_params, height=10, width=40, 
                                    font=("Courier", 10), bg="#f0f0f0")
        self.texto_params.pack(fill=tk.BOTH, expand=True)
        
        # Frame esquerdo - Análise
        frame_analise = ttk.LabelFrame(frame_inferior, text="Análise", padding=10)
        frame_analise.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.texto_analise = tk.Text(frame_analise, height=10, width=40, 
                                      font=("Courier", 10), bg="#f0f0f0")
        self.texto_analise.pack(fill=tk.BOTH, expand=True)
        
        # Frame direito
        frame_direita = tk.Frame(frame_inferior)
        frame_direita.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Botões
        frame_botoes = ttk.LabelFrame(frame_direita, text="Visualizações", padding=10)
        frame_botoes.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(frame_botoes, text="📊 Gráfico MSE", 
                   command=self.plot_mse).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botoes, text="📈 Gráfico Overshoot", 
                   command=self.plot_overshoot).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botoes, text="🔄 Atualizar", 
                   command=lambda: self.carregar_dados(plotar_grafico=False)).pack(side=tk.LEFT, padx=5)
        
        # Frame para gráfico
        self.frame_grafico_nominal = ttk.LabelFrame(frame_direita, text="Comparação Visual", padding=5)
        self.frame_grafico_nominal.pack(fill=tk.BOTH, expand=True)
    
    def setup_aba_robustez(self):
        """Configura aba de robustez paramétrica."""
        
        # Frame superior - Controles
        frame_controles = ttk.LabelFrame(self.aba_robustez, text="Controles", padding=10)
        frame_controles.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(frame_controles, text="Selecione o método:").pack(side=tk.LEFT, padx=5)
        
        self.combo_metodo = ttk.Combobox(frame_controles, state="readonly", width=15)
        self.combo_metodo.pack(side=tk.LEFT, padx=5)
        self.combo_metodo.bind("<<ComboboxSelected>>", lambda e: self.atualizar_robustez())
        
        ttk.Button(frame_controles, text="🔄 Atualizar", 
                   command=self.atualizar_robustez).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_controles, text="📊 Comparar Todos", 
                   command=self.plot_comparacao_robustez).pack(side=tk.LEFT, padx=5)
        
        # Frame do meio - Tabela de cenários
        frame_tabela_rob = ttk.LabelFrame(self.aba_robustez, text="Resultados por Cenário", padding=10)
        frame_tabela_rob.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollbar
        scroll_rob = ttk.Scrollbar(frame_tabela_rob, orient=tk.VERTICAL)
        scroll_rob.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview
        colunas_rob = ("Cenário", "K_Term", "τ", "MSE", "Δ MSE (%)", "Overshoot (%)", "Ts (s)", "Descrição")
        self.tree_robustez = ttk.Treeview(frame_tabela_rob, columns=colunas_rob, show="headings",
                                          yscrollcommand=scroll_rob.set, height=8)
        scroll_rob.config(command=self.tree_robustez.yview)
        
        larguras_rob = [80, 90, 90, 120, 100, 110, 90, 250]
        for col, largura in zip(colunas_rob, larguras_rob):
            self.tree_robustez.heading(col, text=col)
            self.tree_robustez.column(col, width=largura, anchor=tk.CENTER)
        
        self.tree_robustez.pack(fill=tk.BOTH, expand=True)
        
        # Frame inferior - Análise de robustez
        frame_analise_rob = ttk.LabelFrame(self.aba_robustez, text="Análise de Robustez", padding=10)
        frame_analise_rob.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.texto_robustez = tk.Text(frame_analise_rob, height=8, 
                                       font=("Courier", 10), bg="#f0f0f0")
        self.texto_robustez.pack(fill=tk.BOTH, expand=True)
    
    def setup_aba_estatistica(self):
        """Configura aba de análise estatística."""
        
        # Frame superior - Controles
        frame_controles = ttk.LabelFrame(self.aba_estatistica, text="Teste de Friedman", padding=10)
        frame_controles.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(frame_controles, text="Selecione a métrica:").pack(side=tk.LEFT, padx=5)
        
        self.combo_metrica = ttk.Combobox(frame_controles, 
                                        values=["mse", "overshoot", "tempo_acomodacao"],
                                        state="readonly", width=20)
        self.combo_metrica.set("mse")
        self.combo_metrica.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(frame_controles, text="🔄 Executar Teste", 
                command=self.executar_teste_estatistico).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_controles, text="📊 Visualizar Ranking", 
                command=self.plot_ranking_estatistico).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_controles, text="🔬 Pós-teste Nemenyi", 
                command=self.executar_posthoc_nemenyi).pack(side=tk.LEFT, padx=5)
        
        # Frame do meio - Resultados textuais
        frame_resultados = ttk.LabelFrame(self.aba_estatistica, text="Resultados do Teste", padding=10)
        frame_resultados.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.texto_estatistica = tk.Text(frame_resultados, height=15, 
                                        font=("Courier", 10), bg="#f0f0f0")
        self.texto_estatistica.pack(fill=tk.BOTH, expand=True)
        
        # Frame inferior - Informações
        frame_info = ttk.LabelFrame(self.aba_estatistica, text="Sobre o Teste", padding=10)
        frame_info.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        info_texto = """
            📊 TESTE DE FRIEDMAN - ANÁLISE NÃO-PARAMÉTRICA

            O teste de Friedman compara múltiplos métodos simultaneamente verificando
            se as diferenças de desempenho são estatisticamente significativas.

            - Hipótese Nula (H₀): Todos os métodos têm desempenho equivalente
            - Hipótese Alternativa (H₁): Pelo menos um método difere significativamente

            INTERPRETAÇÃO:
            ✓ p < 0.05 → Diferenças são SIGNIFICATIVAS (rejeita H₀)
            ✗ p ≥ 0.05 → Diferenças NÃO são significativas (não rejeita H₀)

            RANKING: Quanto menor o valor, melhor o método

            ───────────────────────────────────────────────────────────

            🔬 PÓS-TESTE DE NEMENYI

            Quando Friedman é significativo, o pós-teste identifica QUAIS pares
            de métodos diferem significativamente entre si.

            - Calcula Diferença Crítica (CD)
            - Compara todos os pares de métodos
            - |Ranking_A - Ranking_B| > CD → Diferença significativa

            Use o botão "🔬 Pós-teste Nemenyi" após executar o Teste de Friedman.
        """
        
        texto_info = tk.Text(frame_info, height=12, font=("Arial", 9), 
                            bg="#f0f0f0", wrap=tk.WORD)
        texto_info.insert(1.0, info_texto)
        texto_info.config(state=tk.DISABLED)
        texto_info.pack(fill=tk.BOTH, expand=True)

    def executar_posthoc_nemenyi(self):
        """Executa pós-teste de Nemenyi após Friedman."""
        from modules.statistics_module import posthoc_nemenyi
        
        metrica = self.combo_metrica.get()
        
        self.texto_estatistica.delete(1.0, tk.END)
        self.texto_estatistica.insert(tk.END, "⏳ Executando pós-teste de Nemenyi...\n\n")
        self.texto_estatistica.update()
        
        try:
            # Primeiro executar Friedman
            resultado = teste_friedman(self.db_name, metrica)
            
            if resultado is None:
                self.texto_estatistica.delete(1.0, tk.END)
                self.texto_estatistica.insert(tk.END, 
                    "❌ Dados insuficientes para análise estatística.\n\n"
                    "Execute o Teste de Friedman primeiro!")
                return
            
            if not resultado['significativo']:
                self.texto_estatistica.delete(1.0, tk.END)
                self.texto_estatistica.insert(tk.END,
                    "⚠️ AVISO: Teste de Friedman NÃO foi significativo!\n\n"
                    f"p-valor = {resultado['pvalue']:.6f} (≥ 0.05)\n\n"
                    "O pós-teste de Nemenyi só é recomendado quando\n"
                    "o teste de Friedman indica diferenças significativas.\n\n"
                    "Deseja continuar mesmo assim?")
                
                if not messagebox.askyesno("Continuar?", 
                    "Friedman não foi significativo. Continuar com Nemenyi?"):
                    self.executar_teste_estatistico()  # Volta para Friedman
                    return
            
            # Executar pós-teste
            posthoc = posthoc_nemenyi(resultado['rankings_dict'], resultado['n_iteracoes'])
            
            if not posthoc:
                self.texto_estatistica.delete(1.0, tk.END)
                self.texto_estatistica.insert(tk.END, "❌ Erro ao executar pós-teste.")
                return
            
            # Formatar saída
            self.texto_estatistica.delete(1.0, tk.END)
            
            texto = f"{'='*60}\n"
            texto += f"PÓS-TESTE DE NEMENYI - {resultado['metrica']}\n"
            texto += f"{'='*60}\n\n"
            
            texto += f"📊 PRÉ-REQUISITO (Friedman):\n"
            texto += f"   χ² = {resultado['statistic']:.4f}\n"
            texto += f"   p-valor = {resultado['pvalue']:.6f}\n"
            if resultado['significativo']:
                texto += f"   ✓ SIGNIFICATIVO - Pós-teste é válido\n\n"
            else:
                texto += f"   ✗ NÃO SIGNIFICATIVO - Pós-teste não recomendado\n\n"
            
            # Calcular CD
            k = resultado['n_metodos']
            N = resultado['n_iteracoes']
            q_alpha = 2.569  # α = 0.05
            CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
            
            texto += f"🔬 DIFERENÇA CRÍTICA (CD):\n"
            texto += f"   CD = {CD:.4f}\n"
            texto += f"   (α = 0.05, k = {k} métodos, N = {N} blocos)\n\n"
            
            texto += f"📋 COMPARAÇÕES PAREADAS:\n"
            texto += f"   {'Método A':<15} {'Método B':<15} {'Diferença':<12} {'Significativo'}\n"
            texto += f"   {'-'*60}\n"
            
            sig_count = 0
            for m1, m2, diff, cd, sig in posthoc:
                flag = "✓" if sig else "✗"
                if sig:
                    sig_count += 1
                texto += f"   {m1:<15} {m2:<15} {diff:<12.4f} {flag}\n"
            
            texto += f"   {'-'*60}\n"
            texto += f"\n🎯 RESUMO:\n"
            texto += f"   Total de comparações: {len(posthoc)}\n"
            texto += f"   Diferenças significativas: {sig_count}\n"
            texto += f"   Diferenças não significativas: {len(posthoc) - sig_count}\n\n"
            
            texto += f"💡 INTERPRETAÇÃO:\n"
            texto += f"   Se |Ranking_A - Ranking_B| > {CD:.4f}:\n"
            texto += f"      → Métodos têm desempenho SIGNIFICATIVAMENTE diferente\n"
            texto += f"   Caso contrário:\n"
            texto += f"      → Métodos têm desempenho estatisticamente equivalente\n\n"
            
            texto += f"{'='*60}\n"
            
            self.texto_estatistica.insert(tk.END, texto)
        
        except Exception as e:
            self.texto_estatistica.delete(1.0, tk.END)
            self.texto_estatistica.insert(tk.END, f"❌ Erro ao executar pós-teste:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def executar_teste_estatistico(self):
        """Executa o teste de Friedman e exibe resultados."""
        metrica = self.combo_metrica.get()
        
        self.texto_estatistica.delete(1.0, tk.END)
        self.texto_estatistica.insert(tk.END, "⏳ Executando teste de Friedman...\n\n")
        self.texto_estatistica.update()
        
        try:
            resultado = teste_friedman(self.db_name, metrica)
            
            if resultado is None:
                self.texto_estatistica.delete(1.0, tk.END)
                self.texto_estatistica.insert(tk.END, 
                    "❌ Dados insuficientes para análise estatística.\n\n"
                    "Requisitos mínimos:\n"
                    "• Pelo menos 3 métodos\n"
                    "• Pelo menos 3 iterações por método\n\n"
                    "Recomendado: 5+ iterações para maior confiabilidade")
                return
            
            # Formatar saída
            self.texto_estatistica.delete(1.0, tk.END)
            
            texto = f"{'='*60}\n"
            texto += f"TESTE DE FRIEDMAN - {resultado['metrica']}\n"
            texto += f"{'='*60}\n\n"
            
            texto += f"📊 CONFIGURAÇÃO:\n"
            texto += f"   Métodos comparados: {resultado['n_metodos']}\n"
            texto += f"   Iterações por método: {resultado['n_iteracoes']}\n"
            texto += f"   Total de amostras: {resultado['n_metodos'] * resultado['n_iteracoes']}\n\n"
            
            texto += f"📈 RESULTADOS:\n"
            texto += f"   Estatística χ²: {resultado['statistic']:.4f}\n"
            texto += f"   p-valor: {resultado['pvalue']:.6f}\n\n"
            
            texto += f"🎯 INTERPRETAÇÃO:\n"
            if resultado['significativo']:
                texto += "   ✓ SIGNIFICATIVO (p < 0.05)\n"
                texto += "   → As diferenças entre métodos são estatisticamente\n"
                texto += "     relevantes e não ocorreram por acaso.\n"
                texto += "   → Rejeita H₀ (hipótese nula)\n"
            else:
                texto += "   ✗ NÃO SIGNIFICATIVO (p ≥ 0.05)\n"
                texto += "   → Não há evidências estatísticas suficientes\n"
                texto += "     de diferenças entre os métodos.\n"
                texto += "   → Não rejeita H₀ (hipótese nula)\n"
            
            texto += f"\n🏆 RANKING MÉDIO (quanto menor, melhor):\n"
            texto += f"   {'Pos':<5} {'Método':<15} {'Ranking':<10}\n"
            texto += f"   {'-'*35}\n"
            
            for idx, (metodo, rank) in enumerate(resultado['rankings'], 1):
                emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
                texto += f"   {emoji} {idx}º{'':<2} {metodo:<15} {rank:.2f}\n"
            
            texto += f"\n{'='*60}\n"
            
            if resultado['n_iteracoes'] < 5:
                texto += "\n⚠️  AVISO: Poucas iterações (< 5)\n"
                texto += "   Execute mais iterações para maior confiabilidade.\n"
            
            self.texto_estatistica.insert(tk.END, texto)
            
        except Exception as e:
            self.texto_estatistica.delete(1.0, tk.END)
            self.texto_estatistica.insert(tk.END, f"❌ Erro ao executar teste:\n{str(e)}")

    def plot_ranking_estatistico(self):
        """Plota gráfico de ranking com significância estatística."""
        import matplotlib.pyplot as plt
        
        metrica = self.combo_metrica.get()
        dados = obter_dados_para_grafico(self.db_name, metrica)
        
        if dados is None:
            messagebox.showinfo("Info", "Dados insuficientes para gráfico!")
            return
        
        plt.close('all')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metodos = dados['metodos']
        rankings = dados['rankings']
        
        cores = ['#27ae60' if i == 0 else '#3498db' if i < 3 else '#95a5a6' 
                for i in range(len(metodos))]
        
        bars = ax.barh(metodos, rankings, color=cores, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, rank in zip(bars, rankings):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{rank:.2f}',
                    ha='left', va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Ranking Médio (menor = melhor)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Método', fontsize=12, fontweight='bold')
        
        # Título com informação de significância
        sig_text = "SIGNIFICATIVO" if dados['significativo'] else "NÃO SIGNIFICATIVO"
        cor_sig = "green" if dados['significativo'] else "red"
        
        ax.set_title(f'Ranking Estatístico - {metrica.upper()}\n'
                    f'Teste de Friedman: χ²={dados["statistic"]:.2f}, '
                    f'p={dados["pvalue"]:.4f} ({sig_text})',
                    fontsize=13, fontweight='bold', pad=15, color=cor_sig)
        
        ax.invert_xaxis()  # Menor ranking à direita
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.show()

    def setup_aba_graficos(self):
        """Configura aba de gráficos avançados."""
        
        frame_botoes = ttk.LabelFrame(self.aba_graficos, text="Visualizações Avançadas", padding=10)
        frame_botoes.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(frame_botoes, text="⚡ Respostas Temporais", 
                   command=self.plot_respostas_temporais).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botoes, text="📉 Regime Permanente", 
                   command=self.plot_regime_permanente).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botoes, text="🧬 Evolução Métodos", 
                   command=self.plot_evolucao_metodos).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botoes, text="🎯 Cenário Pior Caso", 
                   command=self.plot_cenario_pior_caso).pack(side=tk.LEFT, padx=5)
        
        # Frame para descrição
        frame_desc = ttk.LabelFrame(self.aba_graficos, text="Informações", padding=10)
        frame_desc.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        texto_info = """
╔═══════════════════════════════════════════════════════════════════════╗
║                    VISUALIZAÇÕES AVANÇADAS DISPONÍVEIS                ║
╚═══════════════════════════════════════════════════════════════════════╝

📊 Respostas Temporais
   Visualiza resposta completa e zoom no transitório para todos os métodos

📉 Regime Permanente  
   Analisa comportamento após estabilização (foco em precisão)

🧬 Evolução dos Métodos
   Mostra convergência dos algoritmos evolutivos por geração

🎯 Cenário C5 (Pior Caso)
   Compara desempenho no cenário mais adverso (K_Term -10%, τ +10%)

═══════════════════════════════════════════════════════════════════════
        """
        
        texto_widget = tk.Text(frame_desc, font=("Courier", 10), bg="#f0f0f0", height=20)
        texto_widget.insert(1.0, texto_info)
        texto_widget.config(state=tk.DISABLED)
        texto_widget.pack(fill=tk.BOTH, expand=True)
    
    def carregar_dados(self, plotar_grafico=None):
        """Carrega dados do banco e atualiza interface."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Carregar dados nominais
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
            
            # Atualizar combo de métodos
            cursor.execute("SELECT DISTINCT metodo FROM robustez ORDER BY metodo")
            metodos = [row[0] for row in cursor.fetchall()]
            self.combo_metodo['values'] = metodos
            if metodos:
                self.combo_metodo.current(0)
            
            conn.close()
            
            # Limpa tabela nominal
            for item in self.tree_nominal.get_children():
                self.tree_nominal.delete(item)
            
            # Preenche tabela nominal
            if resultados:
                for row in resultados:
                    metodo, mse, overshoot, ts, mg, mf, n = row
                    
                    mg_str = "∞" if (mg and mg > 900) else (f"{mg:.2f}" if mg else "N/A")
                    mf_str = f"{mf:.2f}" if mf else "N/A"
                    
                    self.tree_nominal.insert("", tk.END, values=(
                        metodo,
                        f"{mse:.6f}",
                        f"{overshoot:.2f}",
                        f"{ts:.2f}",
                        mg_str,
                        mf_str,
                        n
                    ))
                
                self.atualizar_analise(resultados)
                self.atualizar_parametros_pid()
                
                plotar_grafico = (plotar_grafico is None and self.primeira_carga)
                if plotar_grafico:
                    self.plot_comparacao_nominal(resultados)
            else:
                messagebox.showinfo("Info", "Nenhum resultado encontrado no banco!")
            
            # Carregar estatísticas 
            if hasattr(self, 'aba_estatistica'):
                self.executar_teste_estatistico()

            # Carregar dados de robustez
            self.atualizar_robustez()
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar dados: {e}")
    
    def atualizar_analise(self, resultados):
        """Atualiza texto de análise."""
        self.texto_analise.delete(1.0, tk.END)
        
        if not resultados:
            return
        
        texto =  "╔═══════════════════════════════════════════╗\n"
        texto += "║      ANÁLISE DE RESULTADOS NOMINAIS       ║\n"
        texto += "╚═══════════════════════════════════════════╝\n\n"
        
        melhor_mse = min(resultados, key=lambda x: x[1])
        texto += f"🏆 MELHOR DESEMPENHO (MSE):\n"
        texto += f"   → {melhor_mse[0]}\n"
        texto += f"   MSE: {melhor_mse[1]:.6f}\n\n"
        
        menor_os = min(resultados, key=lambda x: x[2])
        texto += f"📉 MENOR OVERSHOOT:\n"
        texto += f"   → {menor_os[0]}\n"
        texto += f"   Overshoot: {menor_os[2]:.2f}%\n\n"
        
        texto += "🛡️ ROBUSTEZ (Margens Clássicas):\n"
        metodos_robustos = [r for r in resultados if r[4] and r[5]]
        if metodos_robustos:
            melhor_mf = max(metodos_robustos, key=lambda x: x[5])
            texto += f"   Maior MG: Todos (∞ dB)\n"
            texto += f"   Maior MF: {melhor_mf[0]} ({melhor_mf[5]:.2f}°)\n"
        else:
            texto += "   Dados não disponíveis\n"
        
        texto += "\n" + "─" * 43 + "\n"
        texto += f"📊 Total de métodos: {len(resultados)}\n"
        texto += f"🔬 Total de testes: {sum(r[6] for r in resultados)}\n\n"
        texto += "💡 Use a aba 'Robustez Paramétrica'\n"
        texto += "   para análise sob variações!"
        
        self.texto_analise.insert(1.0, texto)
    
    def atualizar_robustez(self):
        """Atualiza dados de robustez para o método selecionado."""
        metodo = self.combo_metodo.get()
        if not metodo:
            return
        
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT cenario, k_term, tau, mse, variacao_mse, overshoot, 
                       tempo_acomodacao, descricao
                FROM robustez
                WHERE metodo = ?
                ORDER BY 
                    CASE cenario
                        WHEN 'Nominal' THEN 0
                        WHEN 'C1' THEN 1
                        WHEN 'C2' THEN 2
                        WHEN 'C3' THEN 3
                        WHEN 'C4' THEN 4
                        WHEN 'C5' THEN 5
                        ELSE 6
                    END
            """, (metodo,))
            
            dados = cursor.fetchall()
            conn.close()
            
            # Limpa tabela
            for item in self.tree_robustez.get_children():
                self.tree_robustez.delete(item)
            
            # Preenche tabela
            if dados:
                for row in dados:
                    cenario, k_term, tau, mse, var_mse, os, ts, desc = row
                    
                    # Formatação especial para Nominal
                    if cenario == "Nominal":
                        var_str = "---"
                        # Destaque visual
                        self.tree_robustez.insert("", tk.END, values=(
                            cenario, f"{k_term:.2f}", f"{tau:.2f}", 
                            f"{mse:.6f}", var_str, f"{os:.2f}", f"{ts:.2f}", desc
                        ), tags=('nominal',))
                    else:
                        var_str = f"{var_mse:+.2f}"
                        self.tree_robustez.insert("", tk.END, values=(
                            cenario, f"{k_term:.2f}", f"{tau:.2f}",
                            f"{mse:.6f}", var_str, f"{os:.2f}", f"{ts:.2f}", desc
                        ))
                
                # Configurar tag para destaque
                self.tree_robustez.tag_configure('nominal', background='#e8f5e9')
                
                # Atualizar análise de robustez
                self.atualizar_analise_robustez(metodo, dados)
            else:
                messagebox.showinfo("Info", f"Nenhum teste de robustez encontrado para {metodo}")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar robustez: {e}")
    
    def atualizar_analise_robustez(self, metodo, dados):
        """Atualiza análise textual de robustez."""
        self.texto_robustez.delete(1.0, tk.END)
        
        if not dados:
            return
        
        # Filtrar dados (excluir nominal)
        dados_var = [d for d in dados if d[0] != "Nominal"]
        
        if not dados_var:
            self.texto_robustez.insert(1.0, "Nenhum cenário de variação encontrado")
            return
        
        # Calcular estatísticas
        variacoes = [abs(d[4]) for d in dados_var]
        var_media = np.mean(variacoes)
        var_max = max(variacoes)
        
        # Encontrar pior cenário
        pior_cenario = max(dados_var, key=lambda x: abs(x[4]))
        
        # Análise de estabilidade (todos mantiveram MSE finito?)
        todos_estaveis = all(d[3] < 1e6 for d in dados)
        
        texto =  f"╔═══════════════════════════════════════════════════╗\n"
        texto += f"║         ANÁLISE DE ROBUSTEZ - {metodo:<14}      ║\n"
        texto += f"╚═══════════════════════════════════════════════════╝\n\n"
        
        texto += f"📊 ESTATÍSTICAS DE DEGRADAÇÃO:\n"
        texto += f"   Variação média: {var_media:.2f}%\n"
        texto += f"   Variação máxima: {var_max:.2f}%\n\n"
        
        texto += f"⚠️  PIOR CENÁRIO:\n"
        texto += f"   Cenário: {pior_cenario[0]} - {pior_cenario[7]}\n"
        texto += f"   Degradação: {pior_cenario[4]:+.2f}%\n"
        texto += f"   MSE: {pior_cenario[3]:.6f}\n\n"
        
        texto += f"🎯 ESTABILIDADE:\n"
        if todos_estaveis:
            texto += f"   ✓ Estável em TODOS os cenários\n\n"
        else:
            texto += f"   ✗ Instável em algum cenário\n\n"
        
        texto += f"🏆 CLASSIFICAÇÃO:\n"
        if var_media < 5:
            texto += f"   ✓ EXCELENTE (< 5%)\n"
            cor = "green"
        elif var_media < 15:
            texto += f"   ✓ BOA (< 15%)\n"
            cor = "blue"
        elif var_media < 30:
            texto += f"   ⚠ REGULAR (< 30%)\n"
            cor = "orange"
        else:
            texto += f"   ✗ BAIXA (> 30%)\n"
            cor = "red"
        
        texto += f"\n{'─'*51}\n"
        texto += f"💡 Critério: Δ_MSE < 50% em 80% dos casos\n"
        texto += f"             Δ_MSE_max < 100%"
        
        self.texto_robustez.insert(1.0, texto)
    
    def plot_comparacao_nominal(self, resultados):
        """Plota gráfico de barras comparativo nominal."""
        plt.close('all')
        
        for widget in self.frame_grafico_nominal.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
        
        metodos = [r[0] for r in resultados]
        mse_values = [r[1] for r in resultados]
        
        cores = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(metodos))]
        bars = ax.bar(metodos, mse_values, color=cores, alpha=0.8, edgecolor='black')
        bars[0].set_color('#27ae60')
        bars[0].set_linewidth(2)
        
        ax.set_ylabel('MSE', fontweight='bold')
        ax.set_title('Comparação de Desempenho (Nominal)', fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico_nominal)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_comparacao_robustez(self):
        """Compara robustez entre todos os métodos."""
        plt.close('all')
        
        try:
            conn = sqlite3.connect(self.db_name)
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
                messagebox.showinfo("Info", "Nenhum teste de robustez encontrado!")
                return
            
            # Criar figura
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            metodos = [r[0] for r in resultados]
            var_media = [r[1] for r in resultados]
            var_max = [r[2] for r in resultados]
            
            # Subplot 1: Variação Média
            cores1 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(metodos)))
            bars1 = ax1.bar(metodos, var_media, color=cores1, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax1.axhline(y=5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excelente (5%)')
            ax1.axhline(y=15, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Bom (15%)')
            ax1.axhline(y=30, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Regular (30%)')
            
            ax1.set_ylabel('Degradação Média do MSE (%)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Método', fontsize=12, fontweight='bold')
            ax1.set_title('(a) Variação Média de Desempenho', fontsize=13, fontweight='bold', pad=15)
            ax1.legend(loc='upper left', fontsize=9)
            ax1.grid(axis='y', alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Subplot 2: Variação Máxima
            cores2 = plt.cm.Reds(np.linspace(0.4, 0.9, len(metodos)))
            bars2 = ax2.bar(metodos, var_max, color=cores2, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax2.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Limite Aceitável (50%)')
            ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Limite Crítico (100%)')
            
            ax2.set_ylabel('Degradação Máxima do MSE (%)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Método', fontsize=12, fontweight='bold')
            ax2.set_title('(b) Pior Caso (Cenário C5)', fontsize=13, fontweight='bold', pad=15)
            ax2.legend(loc='upper left', fontsize=9)
            ax2.grid(axis='y', alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            fig.suptitle('Comparação de Robustez Paramétrica entre Métodos', 
                         fontweight='bold', fontsize=14, y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
            
            print("\n✓ Gráfico de comparação de robustez gerado")
            print(f"  Método MAIS robusto: {resultados[0][0]} (Δ média: {resultados[0][1]:.2f}%)")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar gráfico de robustez: {e}")
    
    def plot_cenario_pior_caso(self):
        """Plota comparação no REAL pior cenário (maior variação de MSE)."""
        plt.close('all')
        
        try:
            import control as ctl
            
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # IDENTIFICAR QUAL É O PIOR CENÁRIO
            cursor.execute("""
                SELECT cenario, AVG(ABS(variacao_mse)) as degradacao_media
                FROM robustez
                WHERE cenario != 'Nominal'
                GROUP BY cenario
                ORDER BY degradacao_media DESC
                LIMIT 1
            """)
            
            resultado = cursor.fetchone()
            
            if not resultado:
                messagebox.showinfo("Info", "Nenhum teste de robustez encontrado!")
                conn.close()
                return
            
            pior_cenario, degradacao = resultado
            
            print(f"\n🎯 Pior cenário identificado: {pior_cenario} (Δ_MSE = {degradacao:.2f}%)")
            
            # Buscar parâmetros do pior cenário
            cursor.execute("""
                SELECT DISTINCT k_term, tau, descricao
                FROM robustez
                WHERE cenario = ?
                LIMIT 1
            """, (pior_cenario,))
            
            k_term_pior, tau_pior, descricao_pior = cursor.fetchone()
            
            # Buscar métodos disponíveis
            cursor.execute('SELECT DISTINCT metodo FROM resultados ORDER BY metodo')
            metodos = [row[0] for row in cursor.fetchall()]
            
            if not metodos:
                messagebox.showinfo("Info", "Nenhum método encontrado!")
                conn.close()
                return
            
            # Parâmetros nominais
            k_term_nominal = getattr(self, 'k_term_atual', 59.81)
            tau_nominal = getattr(self, 'tau_atual', 401.61)
            setpoint = getattr(self, 'setpoint_atual', 80.0)
            t_max = getattr(self, 't_final_atual', 803.22)
            
            # Criar plantas
            plant_nominal = ctl.tf([k_term_nominal], [tau_nominal, 1])
            plant_pior = ctl.tf([k_term_pior], [tau_pior, 1])
            
            # Criar figura
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            cores = {
                'ZN1': '#1f77b4', 'CC': '#ff7f0e', 'GA': '#2ca02c',
                'PSO': '#d62728', 'DE': '#9467bd', 'CMA-ES': '#8c564b'
            }
            
            t = np.linspace(0, t_max, 1000)
            
            # SUBPLOT 1: Comparação Nominal vs Pior Caso
            for metodo in metodos:
                cursor.execute("""
                    SELECT Kp, Ki, Kd 
                    FROM resultados 
                    WHERE metodo = ?
                    ORDER BY data_hora DESC LIMIT 1
                """, (metodo,))
                
                resultado = cursor.fetchone()
                if resultado:
                    kp, ki, kd = resultado
                    
                    pid_tf = ctl.tf([kd, kp, ki], [1, 0])
                    
                    # Resposta nominal (linha sólida)
                    sys_nominal = ctl.feedback(pid_tf * plant_nominal, 1)
                    t_out, y_out = ctl.step_response(sys_nominal, t)
                    y_nominal = y_out * setpoint
                    
                    # Resposta pior caso (linha tracejada)
                    sys_pior = ctl.feedback(pid_tf * plant_pior, 1)
                    t_out, y_out = ctl.step_response(sys_pior, t)
                    y_pior = y_out * setpoint
                    
                    cor = cores.get(metodo, 'gray')
                    ax1.plot(t_out, y_nominal, color=cor, linewidth=2, 
                            label=f"{metodo} (Nominal)", alpha=0.7)
                    ax1.plot(t_out, y_pior, color=cor, linewidth=2.5, 
                            linestyle='--', label=f"{metodo} ({pior_cenario})", alpha=0.9)
            
            ax1.axhline(setpoint, color='red', linestyle=':', linewidth=2, 
                    label='Setpoint', alpha=0.7)
            ax1.set_xlabel('Tempo (s)', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Temperatura (°C)', fontweight='bold', fontsize=12)
            ax1.set_title(f'(a) Nominal vs {pior_cenario}\n{descricao_pior}', 
                        fontweight='bold', fontsize=12, pad=15)
            ax1.legend(loc='lower right', fontsize=8, ncol=2)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, t_max)
            
            # SUBPLOT 2: Degradação de MSE
            cursor.execute("""
                SELECT metodo, variacao_mse
                FROM robustez
                WHERE cenario = ?
                ORDER BY ABS(variacao_mse) DESC
            """, (pior_cenario,))
            
            dados_pior = cursor.fetchall()
            
            if dados_pior:
                metodos_pior = [d[0] for d in dados_pior]
                degradacoes = [d[1] for d in dados_pior]
                
                cores_bars = [cores.get(m, 'gray') for m in metodos_pior]
                bars = ax2.bar(metodos_pior, degradacoes, color=cores_bars, 
                            alpha=0.8, edgecolor='black', linewidth=1.5)
                
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:+.2f}%',
                            ha='center', va='bottom' if height > 0 else 'top',
                            fontsize=10, fontweight='bold')
                
                ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
                ax2.axhline(y=50, color='orange', linestyle='--', linewidth=2, 
                        alpha=0.5, label='Limite Aceitável (50%)')
                ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, 
                        alpha=0.5, label='Crítico (100%)')
                
                ax2.set_ylabel('Degradação do MSE (%)', fontweight='bold', fontsize=12)
                ax2.set_xlabel('Método', fontweight='bold', fontsize=12)
                ax2.set_title(f'(b) Impacto no Desempenho\nCenário {pior_cenario}', 
                            fontweight='bold', fontsize=12, pad=15)
                ax2.legend(loc='upper left', fontsize=10)
                ax2.grid(axis='y', alpha=0.3)
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            conn.close()
            
            fig.suptitle(f'Análise do Pior Cenário: {pior_cenario} - {descricao_pior}\n'
                        f'K_term: {k_term_pior:.2f} ({(k_term_pior/k_term_nominal-1)*100:+.1f}%), '
                        f'τ: {tau_pior:.2f} ({(tau_pior/tau_nominal-1)*100:+.1f}%)', 
                        fontweight='bold', fontsize=13, y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
            
            print(f"✓ Gráfico do pior cenário ({pior_cenario}) gerado")
            
        except ImportError:
            messagebox.showerror("Erro", "Biblioteca 'control' não encontrada!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar gráfico: {e}")
    
    def plot_mse(self):
        """Plota gráfico detalhado de MSE."""
        plt.close('all')
        
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
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metodos = [r[0] for r in resultados]
            mse_values = [r[1] for r in resultados]
            
            cores = plt.cm.viridis(np.linspace(0.3, 0.9, len(metodos)))
            bars = ax.bar(metodos, mse_values, color=cores, alpha=0.8, 
                         edgecolor='black', linewidth=1.5)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.6f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_ylabel('MSE (Erro Quadrático Médio)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Método de Sintonia', fontsize=12, fontweight='bold')
            ax.set_title('Comparação de Desempenho - MSE', fontsize=14, 
                        fontweight='bold', pad=20)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar gráfico: {e}")
    
    def plot_overshoot(self):
        """Plota gráfico detalhado de Overshoot."""
        plt.close('all')
        
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
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            metodos = [r[0] for r in resultados]
            os_values = [r[1] for r in resultados]
            ts_values = [r[2] for r in resultados]
            
            # Gráfico Overshoot
            cores1 = plt.cm.Reds(np.linspace(0.4, 0.8, len(metodos)))
            bars1 = ax1.bar(metodos, os_values, color=cores1, alpha=0.8, 
                           edgecolor='black', linewidth=1.5)
            
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
            bars2 = ax2.bar(metodos, ts_values, color=cores2, alpha=0.8, 
                           edgecolor='black', linewidth=1.5)
            
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
    
    def plot_respostas_temporais(self):
        """Gera gráfico das respostas temporais com foco no transitório inicial."""
        plt.close('all')
        
        try:
            import control as ctl
            
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
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
            
            dados = {}
            for metodo, kp, ki, kd, overshoot in resultados:
                dados[metodo] = {'Kp': kp, 'Ki': ki, 'Kd': kd, 'Overshoot': overshoot}
            
            CORES = {
                'ZN1': '#1f77b4', 'CC': '#ff7f0e', 'GA': '#2ca02c',
                'PSO': '#d62728', 'DE': '#9467bd', 'CMA-ES': '#8c564b'
            }
            
            Kterm = getattr(self, 'k_term_atual', 59.81)
            tau = getattr(self, 'tau_atual', 401.61)
            setpoint = getattr(self, 'setpoint_atual', 80.0)
            t_max = getattr(self, 't_final_atual', 2*tau)
            
            plant = ctl.tf([Kterm], [tau, 1])
            t = np.linspace(0, t_max, 1000)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # SUBPLOT 1: Resposta completa
            for metodo in dados.keys():
                d = dados[metodo]
                pid_tf = ctl.tf([d['Kd'], d['Kp'], d['Ki']], [1, 0])
                sys_mf = ctl.feedback(pid_tf * plant, 1)
                t_out, y_out = ctl.step_response(sys_mf, t)
                y = y_out * setpoint
                
                cor = CORES.get(metodo, 'gray')
                ax1.plot(t_out, y, color=cor, linewidth=2.5, 
                        label=f"{metodo} (OS: {d['Overshoot']:.1f}%)", alpha=0.85)
            
            ax1.axhline(setpoint, color='red', linestyle='--', linewidth=2, 
                       label=f"Setpoint ({setpoint}°C)", alpha=0.7)
            ax1.set_xlabel('Tempo (s)', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Temperatura (°C)', fontweight='bold', fontsize=12)
            ax1.set_title('(a) Resposta Completa', fontweight='bold', fontsize=13, pad=15)
            ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 2*tau)
            
            # SUBPLOT 2: Zoom no transitório
            t_max_zoom = (2*tau) * 0.2
            
            for metodo in dados.keys():
                d = dados[metodo]
                pid_tf = ctl.tf([d['Kd'], d['Kp'], d['Ki']], [1, 0])
                sys_mf = ctl.feedback(pid_tf * plant, 1)
                t_out, y_out = ctl.step_response(sys_mf, t)
                y = y_out * setpoint
                
                mask = t_out <= t_max_zoom
                t_zoom = t_out[mask]
                y_zoom = y[mask]
                
                cor = CORES.get(metodo, 'gray')
                ax2.plot(t_zoom, y_zoom, color=cor, linewidth=2.5, 
                        label=f"{metodo}", alpha=0.85)
                
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
            
            fig.suptitle('Comparação das Respostas Temporais de Todos os Métodos', 
                         fontweight='bold', fontsize=14, y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
            
            print("\n✓ Gráfico de respostas temporais gerado")
            
        except ImportError:
            messagebox.showerror("Erro", "Biblioteca 'control' não encontrada!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar gráfico de respostas temporais: {e}")
    
    def plot_regime_permanente(self):
        """Plota gráfico de regime permanente para todos os métodos."""
        plt.close('all')
        
        try:
            import control as ctl
            
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('SELECT DISTINCT metodo FROM resultados')
            metodos = [row[0] for row in cursor.fetchall()]
            
            if not metodos:
                messagebox.showinfo("Info", "Nenhum método encontrado no banco!")
                conn.close()
                return
            
            # Parâmetros configurados pelo usuário
            Kterm = getattr(self, 'k_term_atual', 59.81)
            tau = getattr(self, 'tau_atual', 401.61)
            setpoint = getattr(self, 'setpoint_atual', 80.0)
            t_final = getattr(self, 't_final_atual', 2 * tau)
            
            # Tempo de início do regime (20% do tempo total)
            tempo_inicio_regime = int(t_final * 0.2)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            cores = {
                'CC': 'blue', 'CMA-ES': 'orange', 'DE': 'green',
                'GA': 'cyan', 'PSO': 'red', 'ZN1': 'purple'
            }
            
            plant = ctl.tf([Kterm], [tau, 1])
            
            # Vetor de tempo mais longo para capturar regime permanente
            t = np.linspace(0, t_final, int(t_final * 2.5))
            
            for metodo in metodos:
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
                    
                    pid_tf = ctl.tf([Kd, Kp, Ki], [1, 0])
                    sys_mf = ctl.feedback(pid_tf * plant, 1)
                    
                    t_out, y_out = ctl.step_response(sys_mf, t)
                    
                    # Filtrar apenas regime permanente
                    mask = t_out >= tempo_inicio_regime
                    tempos_regime = t_out[mask]
                    temp_regime = y_out[mask] * setpoint
                    
                    cor = cores.get(metodo, 'gray')
                    ax.plot(tempos_regime, temp_regime, label=metodo, color=cor, linewidth=2)
            
            conn.close()
            
            # Linha do setpoint
            ax.axhline(y=setpoint, color='black', linestyle='--', linewidth=2, 
                    label=f'Setpoint ({setpoint}°C)')
            
            # Banda de ±2%
            banda_percentual = 0.02
            y_superior = setpoint * (1 + banda_percentual)
            y_inferior = setpoint * (1 - banda_percentual)
            
            ax.axhline(y=y_superior, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(y=y_inferior, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.fill_between([tempo_inicio_regime, t_final], 
                            y_inferior, y_superior, 
                            color='green', alpha=0.1, label='Banda ±2%')
            
            ax.set_xlabel('Tempo (s)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Temperatura (°C)', fontsize=12, fontweight='bold')
            ax.set_title(f'Resposta em Regime Permanente (após {tempo_inicio_regime}s)', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Limites dinâmicos do eixo Y (±10% do setpoint)
            margin = setpoint * 0.1
            ax.set_ylim([setpoint - margin, setpoint + margin])
            
            # Limites do eixo X
            ax.set_xlim([tempo_inicio_regime, t_final])
            
            plt.tight_layout()
            plt.show()
            
            print(f"\n✓ Gráfico de regime permanente gerado")
            print(f"  Tempo de análise: {tempo_inicio_regime}s até {t_final}s")
            print(f"  Setpoint: {setpoint}°C")
            print(f"  Banda: ±{banda_percentual*100}% ({y_inferior:.2f}°C a {y_superior:.2f}°C)")
            
        except ImportError:
            messagebox.showerror("Erro", "Biblioteca 'control' não encontrada!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar gráfico de regime permanente: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_evolucao_metodos(self):
        """Plota evolução dos métodos evolutivos ao longo das gerações."""
        plt.close('all')
        
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
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
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            cores = {
                'PSO': '#e74c3c', 'GA': '#2ecc71',
                'DE': '#9b59b6', 'CMA-ES': '#f39c12'
            }
            
            markers = {'PSO': 'o', 'GA': 's', 'DE': '^', 'CMA-ES': 'D'}
            
            # SUBPLOT 1: Convergência (Melhor Fitness)
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
            ax1.set_facecolor('#f8f9fa')
            
            # SUBPLOT 2: Fitness Médio
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
            ax2.set_facecolor('#f8f9fa')
            
            conn.close()
            
            fig.suptitle('Evolução dos Algoritmos Evolutivos ao Longo das Gerações', 
                         fontweight='bold', fontsize=16, y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            plt.show()
            
            print("\n✓ Gráfico de evolução gerado com sucesso!")
            print(f"  Métodos analisados: {', '.join(metodos)}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar gráfico de evolução: {e}")
    
    def limpar_banco_dados(self):
        """Limpa todas as tabelas do banco de dados."""
        try:
            # Confirmar ação
            msg = "⚠️ ATENÇÃO: Esta ação irá deletar TODOS os dados do banco!\n\n"
            msg += "Tabelas afetadas:\n"
            msg += "  • resultados\n"
            msg += "  • robustez\n"
            msg += "  • historico_evolutivo\n\n"
            msg += "Esta ação NÃO pode ser desfeita!\n\n"
            msg += "Deseja continuar?"
            
            if not messagebox.askyesno("Confirmar Limpeza", msg, icon='warning'):
                return
            
            # Segunda confirmação
            if not messagebox.askyesno("Confirmação Final", 
                                    "Tem certeza ABSOLUTA?\n\nTodos os dados serão perdidos!",
                                    icon='warning'):
                return
            
            # Limpar banco
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM resultados")
            cursor.execute("DELETE FROM robustez")
            cursor.execute("DELETE FROM historico_evolutivo")
            
            conn.commit()
            
            # Contar registros deletados
            total_deletados = cursor.rowcount
            conn.close()
            
            # Limpar interface
            for item in self.tree_nominal.get_children():
                self.tree_nominal.delete(item)
            
            for item in self.tree_robustez.get_children():
                self.tree_robustez.delete(item)
            
            self.texto_analise.delete(1.0, tk.END)
            self.texto_robustez.delete(1.0, tk.END)
            
            self.combo_metodo['values'] = []
            
            # Log
            self.log("\n" + "="*70)
            self.log("🗑️ BANCO DE DADOS LIMPO COM SUCESSO")
            self.log(f"   Todas as tabelas foram esvaziadas")
            self.log("="*70 + "\n")
            
            messagebox.showinfo("Sucesso", 
                            "✓ Banco de dados limpo com sucesso!\n\n"
                            "Todas as tabelas foram esvaziadas.")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao limpar banco de dados:\n{str(e)}")
            self.log(f"\n✗ ERRO ao limpar banco: {str(e)}")

    def atualizar_parametros_pid(self):
        """Atualiza exibição dos parâmetros PID."""
        self.texto_params.delete(1.0, tk.END)
        
        parametros = print_PID_params(self.db_name)
        
        if not parametros:
            self.texto_params.insert(1.0, "Nenhum parâmetro disponível")
            return
        
        texto =  "╔═══════════════════════════════════════╗\n"
        texto += "║    PARÂMETROS PID SINTONIZADOS        ║\n"
        texto += "╚═══════════════════════════════════════╝\n\n"
        
        for metodo, kp, ki, kd in parametros:
            texto += f"📌 {metodo}\n"
            texto += f"   Kp: {kp:>8.4f}\n"
            texto += f"   Ki: {ki:>8.4f}\n"
            texto += f"   Kd: {kd:>8.4f}\n"
            texto += "   " + "─"*30 + "\n\n"
        
        texto += f"Total de métodos: {len(parametros)}\n"
        
        self.texto_params.insert(1.0, texto)

def main():
    root = tk.Tk()
    app = PIDResultsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()