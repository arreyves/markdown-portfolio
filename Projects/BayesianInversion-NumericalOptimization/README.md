# High-Dimensional Bayesian Inversion & Monte Carlo Numerical Optimization Engine

**Role:** Doctoral Researcher & Numerical Computing Lead  
**Stack:** Python 3, NumPy, SciPy (`scipy.optimize`, `scipy.stats`), PyTorch, Bash, Linux/WSL  
**Context:** Computational Hydroinformatics & Subsurface Parameter Estimation  

## Problem Statement
Calibrating non-linear partial differential equations (Richards' equation for unsaturated porous media flow) involves tightly coupled hydraulic parameters ($K_s$, $\alpha$, $n$, $\theta_r$, $\theta_s$) spanning multiple orders of magnitude ($10^{-6}$ to $10^{2}$). Standard gradient descent fails due to non-smooth objective surfaces, severe ill-conditioning, and local minima.

## Technical Architecture & Numerical Engineering
* **Vectorized Array Computing (`NumPy`):** Eliminated nested Python loops across multi-dimensional spatio-temporal simulation outputs using memory-efficient NumPy views, broadcasting rules, and pre-allocated buffers—cutting post-processing memory overhead and execution latency.
* **Ill-Conditioned Parameter Optimization (`SciPy`):** Re-parameterized hydraulic variables into log-transformed, normalized search spaces to stabilize finite-difference Jacobian approximations and Hessian curvature during `scipy.optimize` routines.
* **Large-Scale Parallel Ensembles:** Orchestrated 10,000-run Monte Carlo Latin Hypercube Sampling (LHS) workflows in Linux/WSL environments to map posterior parameter uncertainty via the Method of Anchored Distributions (MAD).
* **Tensor Prototyping & Stability Verification (`PyTorch`):** Built surrogate neural approximations and custom loss pipelines using PyTorch autograd, enforcing numerical stability via log-space formulations, gradient norm clipping, deterministic seeding, and `torch.allclose` regression checks.

## Key Outcomes & Impact
* **Quantified Predictive Uncertainty:** Established rigorous 95% confidence bounds for subsurface contaminant transport and aquifer vulnerability under noisy, sparse boundary observations.
* **55% Pipeline Throughput Gain:** Applied these vectorized data-triage and numerical validation patterns to production cloud environmental telemetry pipelines at AbbaTek Group.
