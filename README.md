## SisPID

---
#### - Vídeos

1. Vídeo mostrando o funcionamento e a estrutura do projeto.
    ***Funcionamento do sistema**: [Sistema funcionando - Youtube](https://youtu.be/Ez8kzoGIM5U)*

<br>

2. Vídeo mostrando funcionamento do sistema com outros valores.
    ***Outros casos de teste**: [Funcionando com outros valores - Youtube](https://youtu.be/nXDChXy6mfE)*

---

#### - Estrutura do sistema

    📦 Sistema de Análise Comparativa (Sintonia de Controladores PID)
    │
    ├── 📄 main.py                              # Ponto de entrada da aplicação
    │
    ├── 📁 db/                                 # Camada de Banco de Dados
    │   ├── db_module.py                        # Gerenciamento de BD e recuparação de dados
    │   └── pid_results.db                      # BD SQLite com os resultados obtidos
    │
    ├── 📁 GUI/                                # Interface Gráfica do Usuário
    │   └── gui.py                              # Interface visual (tkinter)
    │
    ├── 📁 model/                              # Camada de Modelo de Dados
    │   └── model.py                            # Modelo da planta termica e função de simulação
    │
    └── 📁 modules/                            # Módulos dos Algoritmos
        │
        ├── 🧬 Algoritmos Evolutivos
        │   ├── ga_module.py                    # Genetic Algorithm (Algoritmo Genético)
        │   ├── pso_module.py                   # Particle Swarm Optimization (Enxame de Partículas)
        │   ├── cma_module.py                   # CMA-ES (Covariance Matrix Adaptation)
        │   └── de_module.py                    # Differential Evolution (Evolução Diferencial)
        │
        ├── 📐 Métodos Heurísticos Clássicos
        │   ├── zn_module.py                    # Ziegler-Nichols (método de sintonia heurístico clássico)
        │   └── cc_module.py                    # Cohen-Coon (método de sintonia hrurístico clássico)
        │
        └── 📊 Análise Estatística
            └── statistics_module.py            # Métricas e análise estatística