# Artifacts for paper: *Entanglement-Assisted Quantum Differential Privacy*

This repository contains the code and data of the experiments in Sec. VI. We use Python 3.10.9 with `qiskit` 2.3.0, `qiskit-aer` 0.17.2, `qiskit-ibm-runtime` 0.52.0, `numpy`, `scipy` and `matplotlib` (see `requirements.txt`). Physical experiments were run on the `ibm_fez` QPU of the IBM Quantum platform, and simulations on a laptop with an Intel Core i7-8750H CPU and 16 GB memory. Our code is available via the anonymous link [https://anonymous.4open.science/r/EA-QDP_artifacts/](https://anonymous.4open.science/r/EA-QDP_artifacts/) (this site).

The components of the anonymous repository are as follows:

+ `README.md`: This readme file.
+ `circuits.py`: Circuits of all experiments, calibration of the mechanism parameters, privacy parameters and accuracy bounds.
+ `do_hardware_exp.py`: Submits the circuits to `ibm_fez` and retrieves the measurement counts. Fill in your own IBM Quantum credentials in `save_account()` before use.
+ `do_simulation.py`: Noiseless simulation of all configurations with the Qiskit Aer statevector simulator.
+ `plot_figures.py`: Computes the accuracies from the measurement counts and plots Fig. 2 (bottom), Fig. 4 and Fig. 5 into `figures/`; every plotted point is also listed in `results_summary.csv`.
+ `inspect_circuits.ipynb`: Jupyter notebook that shows, for any configuration, the circuit, the compiled circuit executed on `ibm_fez`, its measurement counts, and the circuit used in simulation.
+ `data/`: Measurement counts of every configuration shown in the paper, on hardware (`data/hardware/`) and in simulation (`data/simulation/`).
+ `qpy/`: The circuits executed on `ibm_fez`, before and after compilation, in Qiskit's QPY format.

The three experiments are named `motivating` (Sec. VI-A, Fig. 2), `attack` (Sec. VI-B, Fig. 4) and `defense` (Sec. VI-C, Fig. 5). Their configurations are `XZ_NR`, `XZ_EA` for $\mathcal E_{XZ}$ in the motivating example; `XZ_BV`, `XZ_EA` for $\mathcal E_{XZ,q}^{\otimes n}$ and `DP_BV`, `DP_EA` for $\mathcal D_p^{\otimes n}$; and `DP_GLOBAL_EA` for $\mathcal D_p^{\rm glob}$. Here `NR` and `BV` denote the adversary without a reference system (measuring the returned qubit directly, or running the Bernstein-Vazirani algorithm), and `EA` denotes the entanglement attack.

## Reproducing the results

+ **Figures:** `python plot_figures.py`.
+ **Circuits:** open `inspect_circuits.ipynb` and run all cells.
+ **Simulation:** `python do_simulation.py [experiment ...] [--n N ...]`, e.g. `python do_simulation.py attack --n 1 2 3`. It reports whether the counts are identical to those in `data/simulation/`. All configurations with $n\le5$ take about 20 seconds; the full run takes about one hour.
+ **Hardware:** after `python do_hardware_exp.py save-account`, run `python do_hardware_exp.py submit <experiment> [--config ...] [--n ...]` and then `python do_hardware_exp.py collect <experiment>`. By default the compiled circuits in `qpy/` are submitted; see `python do_hardware_exp.py -h` for other options. Hardware results vary with the device noise at the time of execution.
