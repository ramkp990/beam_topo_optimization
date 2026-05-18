# beam_shape_optimization

# Beam Topology Optimization with GNN Surrogate

A GNN-based surrogate model for structural topology optimization, trained on FEM simulation data generated with FEniCS. Includes an LLM-guided dataset generation pipeline (Azure OpenAI) that produces diverse boundary condition configurations, reducing reliance on a single fixed problem setup.

> This repository contains code developed as part of the M.Sc. thesis:
> **"Machine Learning and Surrogate-Based Optimization for Engineering Design"**
> Hamburg University of Technology (TUHH)

---

## Overview

Topology optimization finds the optimal material distribution inside a structure given load and boundary conditions. Classical methods (e.g., SIMP) require dozens to hundreds of FEM solves per design iteration — computationally expensive at scale.

This project trains a GNN surrogate that predicts the next density field from the current one, replacing FEM solves during optimization. The mesh is represented as a graph where each element is a node connected to its 4-neighbors, enabling the GNN to learn local structural relationships.

---

## Architecture

```
FEniCS Simulation (SIMP method)
    │
    ▼
Dataset: (density_t, energy_t) ──► next_density_t+1
    │
    ▼
Graph Construction (4-neighbor grid)
    │
    ▼
GCNConv GNN (2-layer)
    ├── Input: [density, strain_energy] per element
    └── Output: predicted next density field
```

**Two datasets are combined for training:**
- `topopt_dataset.npz` — standard SIMP runs with fixed boundary conditions
- `topopt_dataset_llm.npz` — LLM-guided runs with varied load positions, magnitudes, and boundary configurations (generated via Azure OpenAI)

---

## LLM-Guided Dataset Generation

A key contribution of this work is using an LLM (Azure OpenAI) to propose diverse structural problem configurations — varying load positions, magnitudes, support placements, and material fractions. This augments the training set beyond what a single fixed problem setup can provide, improving the surrogate's generalization.

Pipeline: `final_pb_create_dataset_llm.py`
- Queries Azure OpenAI for problem parameter suggestions
- Runs FEniCS FEM solver with each configuration
- Stores density and strain energy fields per iteration

---

## File Structure

```
gnn.py                          # GNN model definition and training (GCNConv)
gnn_predict.py                  # Surrogate inference integrated with FEniCS solver
final_pb_create_dataset_std.py  # Standard dataset generation (fixed boundary conditions)
final_pb_create_dataset_llm.py  # LLM-guided diverse dataset generation
beam_third_area_constant.py     # Beam optimization: constant cross-sectional area
beam_third_weight_minimization.py # Beam optimization: weight minimization (Aluminum)
final_pb_images.py              # Visualization of density fields across iterations
final_density.png               # Example optimized density field output
```

---

## Tech Stack

- **PyTorch** + **PyTorch Geometric** — GNN model (GCNConv)
- **FEniCS** — finite element analysis and SIMP topology optimization
- **Azure OpenAI** — LLM-guided problem configuration generation
- **NumPy / SciPy** — numerical operations and differential evolution
- **Matplotlib** — density field visualization

---

## Setup

```bash
# FEniCS installation (recommended via Docker or conda)
# https://fenicsproject.org/download/

pip install torch torch-geometric numpy scipy matplotlib openai

# Generate standard dataset
python final_pb_create_dataset_std.py

# Generate LLM-augmented dataset (requires Azure OpenAI credentials)
python final_pb_create_dataset_llm.py

# Train GNN surrogate
python gnn.py

# Run surrogate-assisted optimization
python gnn_predict.py
```

> **Azure OpenAI credentials** must be configured via `keyring` or environment variables before running the LLM pipeline.

---

## Example Output

The file `final_density.png` shows a converged density field from the surrogate-assisted optimization — darker regions indicate material, lighter regions indicate void.
