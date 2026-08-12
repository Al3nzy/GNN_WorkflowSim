# 🦗 WorkflowSim — GNN-Enhanced & Locust-Inspired Workflow Scheduling

<p align="center">
  <img src="https://img.shields.io/badge/Java-11%2B-orange?logo=java" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/WorkflowSim-1.1.0-blue" />
  <img src="https://img.shields.io/badge/CloudSim-3.0-blue" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" />
</p>

<p align="center">
  <b>Structure-Aware Graph Neural Networks & Density-Adaptive Swarm Optimization for Cloud Workflow Scheduling</b><br>
  <i>Dr. Mohammed Alaa Ala'anzy — SDU University, Kazakhstan</i>
</p>

---

> *Desert locusts don't follow a timer; they respond to crowding. LIWSA brings this mechanism into cloud workflow scheduling, while Graph Neural Networks (GNN) learn structural DAG patterns to guide the swarm search.*

---

## 🚀 Overview

This repository provides a unified framework combining **LIWSA** (Density-Adaptive Locust Swarm Optimisation)[cite: 2] and its **GNN-based Warm-Start Extension**[cite: 3, 4] for cloud workflow scheduling inside [WorkflowSim 1.1.0](https://github.com/WorkflowSim/WorkflowSim-1.0)[cite: 2] and [CloudSim 3.0](https://github.com/Cloudslab/cloudsim)[cite: 2].

The system combines:
1. **Python GNN Training Pipeline**: Learns task execution metrics from DAG structural topology and exports standalone model weights[cite: 3, 4].
2. **Pure-Java WorkflowSim Extension**: Loads exported model weights without heavyweight native ML dependencies to initialize Pareto swarm optimization[cite: 2, 4].
3. **Multi-Algorithm Benchmarking**: Integrated baseline comparisons including HEFT, Min-Min, MLEAO, LIWSA, LIWSA-ML, and LIWSA-GNN[cite: 2, 4].

---

## 🧠 Core Algorithms

### 1. LIWSA (Locust-Inspired Workflow Scheduling Algorithm)
A multi-objective swarm optimization algorithm where candidate schedules switch between solitary and gregarious phases based on local crowding density $\rho_i$[cite: 2]. It produces a true Pareto front across **Makespan** and **Execution Cost** without needing scalar weights or objective normalization[cite: 2].

### 2. LIWSA-ML (OLS Warm-Start)
Injects ordinary least-squares (OLS) regression predictions directly into the initial swarm population based on topological features, task fan-in/out, and VM processing speeds[cite: 2].

### 3. LIWSA-GNN (Graph Neural Network Warm-Start)
Extends swarm initialization using a Graph Neural Network (GNN) trained on workflow DAG structures[cite: 3, 4]. The GNN embeds DAG nodes and message-passing dependencies to predict task metrics across varying workflow scales[cite: 4].

---

## 📁 Repository Structure

```text
├── config/dax/                         # Benchmark scientific DAG inputs (.xml)
├── sources/org/workflowsim/            # Java WorkflowSim Core & Planning
│   ├── WorkflowPlanner.java            # Main planner dispatcher
│   ├── planning/
│   │   ├── LIWSAGNNPlanningAlgorithm.java # GNN-guided planner implementation
│   │   ├── LIWSAPlanningAlgorithm.java   # Core LIWSA implementation
│   │   ├── LIWSAMLPlanningAlgorithm.java # OLS-boosted LIWSA implementation
│   │   ├── HEFTPlanningAlgorithmExample1.java
│   │   ├── MLEAOPlanningAlgorithmExample.java
│   │   ├── ParetoMetrics.java          # Hypervolume metrics calculator
│   │   ├── ResultsCsvWriter.java       # Standardized CSV exporter
│   │   └── RunMetricsCalculator.java   # Performance metrics evaluator
├── gnn_weights.txt                     # Exported GNN model weights for Java runtime
├── build_dataset.py                    # Constructs PyTorch base datasets
├── build_family_datasets.py            # Generates family-specific datasets
├── build_augmented_dataset.py          # Data augmentation via synthetic DAGs
├── model.py                            # PyTorch GNN architecture definition
├── decoder.py                          # Schedule decoding logic
├── dax_parser.py                       # Pegasus DAX XML parser
├── train_baseline.py                   # Structure-blind baseline training
├── train_production.py                 # Production GNN model training
├── run_one_seed.py                     # Single-seed execution runner
├── run_family_multiseed.py             # Multi-seed held-out family evaluation
├── export_weights.py                   # Exports PyTorch weights to gnn_weights.txt
├── verify_plain_forward.py             # Validates Java-side forward pass math
└── gnn_benchmark_results.csv           # Benchmark execution results
```[cite: 1, 2, 3, 4]

---

## 🔄 Execution Workflow

```mermaid
flowchart LR
    A[Pegasus DAX Files] --> B[Python Dataset Builder]
    B --> C[PyTorch GNN Training]
    C --> D[Weight Exporter]
    D -->|gnn_weights.txt| E[Java WorkflowSim Planner]
    E --> F[LIWSA Swarm Optimization]
    F --> G[Pareto-Optimal Execution Results]
```[cite: 3, 4]

---

## 📊 Experimental Benchmark Results (LIWSA-GNN)

Below are the experimental results for **LIWSA-GNN** across 15 scientific workflow instances from the Pegasus Workflow Gallery (averaged across 5 random seeds):

| Workflow | Tasks | Avg Makespan (s) | Avg Cost ($) | Avg Sim Wallclock (s) |
| :--- | :---: | :---: | :---: | :---: |
| **CyberShake_30** | 30 | 394.60 | 815.78 | 0.082 |
| **CyberShake_50** | 50 | 559.57 | 1,631.05 | 0.150 |
| **CyberShake_100** | 100 | 966.36 | 3,350.46 | 0.326 |
| **Epigenomics_24** | 24 | 4,015.80 | 8,384.21 | 0.082 |
| **Epigenomics_46** | 46 | 7,695.44 | 19,942.00 | 0.154 |
| **Epigenomics_100** | 100 | 55,835.70 | 199,899.00 | 0.346 |
| **Inspiral_30** | 30 | 853.61 | 3,141.33 | 0.084 |
| **Inspiral_50** | 50 | 1,352.08 | 5,623.66 | 0.144 |
| **Inspiral_100** | 100 | 2,896.56 | 9,930.78 | 0.310 |
| **Montage_25** | 25 | 47.30 | 105.73 | 0.242 |
| **Montage_50** | 50 | 84.01 | 242.82 | 0.172 |
| **Montage_100** | 100 | 136.15 | 536.31 | 0.346 |
| **Sipht_30** | 30 | 2,207.63 | 2,498.77 | 0.212 |
| **Sipht_60** | 60 | 2,332.87 | 5,256.17 | 0.406 |
| **Sipht_100** | 100 | 2,438.34 | 8,268.49 | 0.674 |

---

## 🛠️ Execution & Getting Started

### Step 1: Python Pipeline (Dataset & GNN Model Export)

To build datasets, train the GNN model, and export weights for Java consumption, run the following scripts in sequence from the root directory[cite: 1, 4]:

```bash
# 1. Build training and testing datasets from DAX inputs
python build_dataset.py

# 2. Train single-seed model baseline
python run_one_seed.py 1

# 3. Construct family-specific holdout datasets
python build_family_datasets.py

# 4. Evaluate multi-seed generalization across held-out families
python run_family_multiseed.py

# 5. Train production model on full dataset
python train_production.py

# 6. Export trained model weights to plain-text format for Java
python export_weights.py

# 7. Verify mathematical equivalence between PyTorch and Java reference implementation
python verify_plain_forward.py
```[cite: 1, 4]

Step 6 generates `gnn_weights.txt` in the root directory[cite: 4].

---

### Step 2: Java Patch & Integration

1. **Update Enum** in `sources/org/workflowsim/utils/Parameters.java`:
   ```java
   public enum PlanningAlgorithm {
       INVALID, RANDOM, HEFT, DHEFT, LIWSA, MLEAO, LIWSAML, LIWSAGNN
   }
   ```[cite: 4]

2. **Register Planner** in `sources/org/workflowsim/WorkflowPlanner.java`:
   ```java
   case LIWSAGNN:
       planner = new LIWSAGNNPlanningAlgorithm();
       break;
   ```[cite: 4]

3. **Deploy Model Weights**: Ensure `gnn_weights.txt` is located in the working directory from which your JVM command is run[cite: 4].

---

### Step 3: Run WorkflowSim Experiments

Compile the Java planner extensions and execute the benchmarks[cite: 2]:

```bash
# Compile Java planner sources
javac -cp ".:workflowsim.jar:lib/*" sources/org/workflowsim/planning/LIWSAGNNPlanningAlgorithm.java

# Run a single GNN-guided simulation run
java -cp ".:workflowsim.jar:lib/*" org.workflowsim.examples.planning.LIWSAGNNPlanningAlgorithmExample

# Run the full comparative benchmark
java -cp ".:workflowsim.jar:lib/*" org.workflowsim.examples.planning.LIWSABenchmarkExample
```[cite: 2]

---

## 📚 Scope & Generalization Notes

* **Scale Generalization**: The GNN predictor is validated for generalizing across **problem sizes** within the workflow families included in training (e.g., predicting on 1000-task workflows when trained on smaller instances)[cite: 4].
* **Cross-Family Transfer**: Cross-family transfer to completely unseen DAG topologies requires synthetic data augmentation during pre-training[cite: 4].

---

## 📄 License

Copyright 2025–2026 SDU University, Kazakhstan[cite: 2].  
Licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)[cite: 2].