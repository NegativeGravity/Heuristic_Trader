# Advanced Evolutionary Trading System: NSGA-II, CPCV, and Fractional Differentiation

An institutional-grade algorithmic trading framework designed to discover, optimize, and validate robust trading strategies using Multi-Objective Genetic Algorithms (NSGA-II) and advanced financial econometrics.

## 🚀 Key Features

* **Fractional Differentiation (FracDiff):** Implements Fixed Width Window Fractional Differentiation to transform non-stationary price series into stationary features while preserving maximum historical memory.
* **NSGA-II & Island Model:** Uses a multi-objective genetic algorithm to balance Net Profit against Maximum Drawdown. The Island Model prevents premature convergence by maintaining diverse sub-populations.
* **Combinatorial Purged Cross-Validation (CPCV):** A rigorous validation framework that eliminates data leakage through purging and embargoing, testing strategies across multiple combinatorial paths.
* **High-Performance Backtesting:** The core signal generation and backtesting engines are optimized with **Numba (JIT compilation)** for near-C execution speeds.
* **Consensus Ensemble Logic:** Executes trades based on a "Team" of Pareto-optimal specialists, requiring at least 50% agreement and using dynamic risk scaling (1% to 5%).
* **Feature Clustering:** Groups technical indicators using Hierarchical Clustering (Ward’s Method) based on predictive Rank IC to ensure strategy diversity.

---

## 🏗️ Architecture & Workflow

The system operates in three distinct phases to ensure the robustness of the discovered strategies:

### Phase 1: CPCV Robustness Check

The dataset is split into  bins. The system generates all possible combinations of training and testing paths, applying purging and embargoes to prevent "look-ahead" bias.

### Phase 2: Production Training

The elite candidates identified during CPCV are used to seed a final evolutionary run on the full training set to produce a diverse population of "specialists".

### Phase 3: Ensemble Validation

The Pareto Rank-0 individuals form an ensemble team. This team is tested on held-out data, where trades are only executed if a consensus is reached.

---

## 🛠️ Installation

```bash
pip install numpy pandas numba scipy matplotlib

```

*Note: This environment is designed for Google Colab or local Python environments with high-performance computing capabilities.*

---

## ⚙️ Configuration

The system is highly modular. You can adjust the parameters within the `Config` dataclass:

| Parameter | Default Value | Description |
| --- | --- | --- |
| `risk_per_trade` | 0.01 (1%) | Base risk per trade. |
| `reward_risk_ratio` | 2.0 | Fixed RR ratio for all candidates. |
| `n_islands` | 4 | Number of independent genetic sub-populations. |
| `frac_diff_d` | 0.35 | Differentiation order for stationarity. |
| `n_bins` | 6 | Number of blocks for CPCV splits. |

---

## 📊 Core Modules

* **`preprocess_with_frac_diff`**: Cleans data and applies the FFD (Fixed-Width Window) algorithm.
* **`backtest_numba_stats`**: Individual strategy evaluation engine.
* **`get_cpcv_splits`**: Generates purged/embargoed train-test indices.
* **`evolve_islands`**: Manages the life cycle of the genetic algorithm across islands.

---

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading financial markets involves significant risk. The authors are not responsible for any financial losses incurred through the use of this code.

---

