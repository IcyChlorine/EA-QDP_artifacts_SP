"""Noiseless simulation of the experiments in Sec. VI with the Qiskit Aer statevector simulator.

Usage:
    python do_simulation.py                       # all experiments, all n (n <= 10)
    python do_simulation.py attack --n 1 2 3      # selected experiments / sizes
    python do_simulation.py --out my_dir          # output directory (default: reproduced/)

For every configuration, 2,048 shots are split evenly over its secret strings
(circuits.secret_strings), as on hardware. The results are written in the same
format as data/simulation/*.json and compared count by count with the archived
simulation data. With the same seeds and package versions (requirements.txt) the
counts are reproduced exactly.

Circuits:
* motivating, entanglement attack (XZ_EA): the hardware logical circuit itself
  (E_XZ realized with two environment qubits), transpiled for the Aer simulator.
* motivating, no reference (XZ_NR): one system qubit with the Pauli channel of E_XZ.
* attack, D_p^{(x)n} in defense: circuits.simulation_circuit with the Pauli
  error of one copy of the mechanism on each system qubit, i.e. one Pauli
  branch is sampled independently per qubit and per shot (Pauli trajectories).
* D_p^glob in defense: one keep/replace decision per shot shared by all
  system qubits. The number of kept shots is drawn as Binomial(shots, p); the
  kept shots run without error and the replaced shots run with an independent
  uniform Pauli (I, X, Y, Z) on each system qubit, which replaces the register
  by I/2^n.

Run time on a laptop (Intel i7-8750H, 16 GB): seconds for n <= 6; the n = 10
entanglement-attack configurations (20 qubits) take several minutes each.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error

import circuits as C

HERE = Path(__file__).resolve().parent
# seed_simulator = SEED_BASE[experiment] + 10 n + SEED_OFFSET[configuration]
SEED_BASE = {"attack": 20261113, "defense": 20261114}
SEED_OFFSET = {"attack": {"XZ_BV": 0, "XZ_EA": 1, "DP_EA": 2, "DP_BV": 3},
               "defense": {"DP_EA": 0, "DP_GLOBAL_EA": 1}}
MOTIVATING_SEED = {"XZ_EA": 20261113, "XZ_NR": 314159265}
SEED_TRANSPILER = 41


def simulator():
    return AerSimulator(method="statevector", precision="double",
                        max_parallel_experiments=1, max_parallel_shots=1)


def to_key(word, configuration):
    """Aer count key -> '<reference bits>|<system bits>' with character i = qubit pair i."""
    parts = word.split()
    if configuration.endswith("_BV"):
        return "|" + parts[0][::-1]
    flag, guess = parts                      # registers: guess <- R, flag <- S
    return guess[::-1] + "|" + flag[::-1]


def run_motivating(configuration):
    secrets = C.secrets_for("motivating", 1)
    shots = C.shots_per_secret("motivating", 1)
    seed = MOTIVATING_SEED[configuration]
    sim = AerSimulator(method="statevector", precision="double")
    if configuration == "XZ_EA":
        circuits = transpile([C.hardware_circuit("motivating", "XZ_EA", x) for x in secrets], backend=sim,
                             optimization_level=1, seed_transpiler=SEED_TRANSPILER)
        seeds = {"seed_transpiler": SEED_TRANSPILER, "seed_simulator": seed}
        key = lambda w: f"{w[1]}|{w[0]}"           # bell[1]bell[0] -> R|S
    else:
        error = pauli_error(C.pauli_branches("XZ_NR", C.mechanism_parameter("motivating", "XZ_NR", 1))).to_instruction()
        circuits = [C.motivating_no_reference_simulation_circuit(x, error) for x in secrets]
        seeds = {"seed_simulator": seed}
        key = lambda w: f"|{w}"
    result = sim.run(circuits, shots=shots, seed_simulator=seed).result()
    runs = []
    for i, secret in enumerate(secrets):
        counts = {key(w): c for w, c in result.get_counts(i).items()}
        runs.append({"secret": secret, "shots": shots, "counts": dict(sorted(counts.items()))})
    return runs, seeds


def run_product(experiment, configuration, n, seed):
    """Mechanism applied independently to each system qubit (one Aer run for all secret strings)."""
    secrets, shots = C.secret_strings(n), C.shots_per_secret(experiment, n)
    x = C.mechanism_parameter(experiment, configuration, n)
    error = pauli_error(C.pauli_branches(configuration, x)).to_instruction()
    circuits = [C.simulation_circuit(secret, configuration, error) for secret in secrets]
    result = simulator().run(circuits, shots=shots, seed_simulator=seed).result()
    runs = []
    for i, secret in enumerate(secrets):
        counts = {to_key(w, configuration): c for w, c in result.get_counts(i).items()}
        runs.append({"secret": secret, "shots": shots, "counts": dict(sorted(counts.items()))})
    return runs


def run_global(n, seed):
    """D_p^glob: one keep/replace decision per shot, shared by all system qubits."""
    secrets, shots = C.secret_strings(n), C.shots_per_secret("defense", n)
    p = C.mechanism_parameter("defense", "DP_GLOBAL_EA", n)
    uniform = pauli_error([(pauli, .25) for pauli in "IXYZ"]).to_instruction()
    rng = np.random.default_rng(seed)
    sim = simulator()
    runs = []
    for secret in secrets:
        kept = int(rng.binomial(shots, p))
        joint = Counter()
        for size, error in ((kept, None), (shots - kept, uniform)):
            if size:
                circuit = C.simulation_circuit(secret, "DP_GLOBAL_EA", error)
                result = sim.run(circuit, shots=size, seed_simulator=int(rng.integers(2**31))).result()
                for w, c in result.get_counts().items():
                    joint[to_key(w, "DP_GLOBAL_EA")] += c
        runs.append({"secret": secret, "shots": shots, "counts": dict(sorted(joint.items()))})
    return runs


def simulate(experiment, configuration, n):
    if experiment == "motivating":
        runs, seeds = run_motivating(configuration)
    else:
        seed = SEED_BASE[experiment] + 10*n + SEED_OFFSET[experiment][configuration]
        seeds = {"seed_simulator": seed}
        runs = run_global(n, seed) if configuration == "DP_GLOBAL_EA" else run_product(experiment, configuration, n, seed)
    x = C.mechanism_parameter(experiment, configuration, n)
    return {"experiment": experiment, "platform": "aer_simulator", "configuration": configuration, "n": n,
            "parameter": {"q" if configuration.startswith("XZ") else "p": x},
            "shots": sum(r["shots"] for r in runs),
            "simulation": {"method": "statevector", "precision": "double", "device_noise": None, **seeds},
            "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="*", metavar="experiment",
                        help=f"any of {', '.join(C.EXPERIMENTS)} (default: all)")
    parser.add_argument("--n", nargs="+", type=int, help="qubit numbers to simulate (default: all, n <= 10)")
    parser.add_argument("--out", default=HERE / "reproduced", type=Path, help="output directory")
    args = parser.parse_args()
    unknown = set(args.experiments) - set(C.EXPERIMENTS)
    if unknown:
        parser.error(f"unknown experiment(s): {', '.join(sorted(unknown))}")
    out = args.out / "simulation"
    out.mkdir(parents=True, exist_ok=True)
    for experiment in args.experiments or C.EXPERIMENTS:
        archived = {(r["configuration"], r["n"]): r for r in
                    json.loads((HERE/"data"/"simulation"/f"{experiment}.json").read_text(encoding="utf-8"))["records"]}
        records = []
        for configuration, n in C.configurations(experiment, "simulation"):
            if args.n and n not in args.n:
                continue
            record = simulate(experiment, configuration, n)
            same = [r["counts"] for r in record["runs"]] == [r["counts"] for r in archived[configuration, n]["runs"]]
            print(f"{experiment:10s} {configuration:13s} n={n:2d}  counts identical to data/simulation: {same}",
                  flush=True)
            records.append(record)
        path = out / f"{experiment}.json"
        path.write_text(json.dumps({"records": records}, indent=1) + "\n", encoding="utf-8")
        print(f"-> {path}")


if __name__ == "__main__":
    main()
