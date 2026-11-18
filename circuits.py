"""Circuits, mechanism parameters and configurations of the experiments in Sec. VI.

Three experiments are covered:

* ``motivating`` (Sec. VI-A, Fig. 2): computation U_D = X^x on one qubit,
  mechanism E_XZ (equal weights on I, X, Z), with the entanglement attack
  (``XZ_EA``: one Bell pair, Bell measurement) or without a reference
  (``XZ_NR``: input |0>, measurement in the computational basis).
* ``attack`` (Sec. VI-B, Fig. 4): phase oracle U_D|a> = (-1)^{a.x}|a> on n
  qubits, mechanisms E_XZ,q^{(x)n} and D_p^{(x)n} calibrated to a total QDP
  (eta = 1) parameter of 3, with (``*_EA``) or without (``*_BV``) a reference.
* ``defense`` (Sec. VI-C, Fig. 5): the same oracle and entanglement attack,
  mechanisms D_p^{(x)n} and D_p^glob calibrated to a total EA-QDP parameter of 3.

Two families of circuits are built:

* hardware circuits (``hardware_circuit``) realize each mechanism with
  environment qubits, exactly as submitted to ``ibm_fez``;
* simulation circuits (``simulation_circuit``) apply the mechanism directly as
  a Pauli error on each system qubit and are sampled by Qiskit Aer
  (see ``do_simulation.py``).

Conventions. Character i of a secret string x is the bit x_i encoded on system
qubit S_i, paired with reference qubit R_i. In the entanglement attack, R_i and
S_i start in |Phi+> and are finally measured in the Bell basis: CNOT(R_i -> S_i)
followed by H(R_i). The R_i outcome is the phase bit and the S_i outcome is the
parity bit. In the BV attack (no reference), S is measured after H^{(x)n}.

The BV layers H^{(x)n} before and after the oracle (Fig. 3) are omitted in the
entanglement-attack circuits: both mechanisms commute with conjugation by H up to
relabeling Pauli branches of equal weight, so the outcome statistics are the same
up to a relabeling of the Bell outcomes.
"""
from math import acos, asin, expm1, log, log1p, sqrt, tanh
import random

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

BACKEND = "ibm_fez"
SHOTS = 2048                 # shots per configuration, split evenly over its secret strings
BUDGET = 3                   # total privacy budget of the attack and defense experiments
SECRET_SEED = 20261112       # seed for sampling secret strings when n >= 4
SECRETS_SAMPLED = 8          # number of secret strings sampled when n >= 4
N_RANGE = range(1, 11)       # n = 1, ..., 10 (Figs. 4 and 5)
GLOBAL_HARDWARE_N = range(1, 7)  # D_p^glob on hardware is shown for n <= 6 (Fig. 5b)

EXPERIMENTS = {
    "motivating": ("XZ_NR", "XZ_EA"),
    "attack": ("XZ_BV", "XZ_EA", "DP_BV", "DP_EA"),
    "defense": ("DP_EA", "DP_GLOBAL_EA"),
}


# --------------------------------------------------------------------------
# Secret strings
# --------------------------------------------------------------------------
def secret_strings(n):
    """All 2^n strings for n <= 3; otherwise 8 strings drawn uniformly without replacement.

    The draws for n = 4, 5, ... come sequentially from one random.Random(SECRET_SEED)
    stream, so the strings for a given n depend only on n. All configurations with the
    same n use the same strings.
    """
    if n <= 3:
        return [format(x, f"0{n}b") for x in range(2**n)]
    rng = random.Random(SECRET_SEED)
    for m in range(4, n + 1):
        drawn = sorted(format(x, f"0{m}b") for x in rng.sample(range(2**m), SECRETS_SAMPLED))
    return drawn


# --------------------------------------------------------------------------
# Mechanism parameters and privacy parameters
# --------------------------------------------------------------------------
def xz_qdp_epsilon(q):
    """QDP (eta = 1) parameter of one copy of E_XZ,q: ln[(1+s)/(1-s)], s = max(q, |2q-1|)."""
    s = max(q, abs(2*q - 1))
    return log((1 + s) / (1 - s))


def depolarizing_qdp_epsilon(p, d=2):
    """QDP (eta = 1) parameter of D_p on dimension d: ln[1 + d p/(1-p)]."""
    return log1p(d * p / (1 - p))


def depolarizing_ea_epsilon(p, d=2):
    """EA-QDP parameter of D_p on dimension d: ln[1 + d^2 p/(1-p)]."""
    return log1p(d * d * p / (1 - p))


def mechanism_parameter(experiment, configuration, n):
    """q of E_XZ,q or keep probability p of D_p, as used in the experiment."""
    if experiment == "motivating":
        return 1/3
    if experiment == "attack":
        if configuration.startswith("XZ"):
            # Larger root of n * eps_QDP(E_XZ,q) = 3, i.e. q = tanh(3/(2n)); infeasible
            # for n >= 5 since eps_QDP(E_XZ,q) >= ln 2, where q = 1/3 is used instead.
            return tanh(BUDGET / (2*n)) if n * log(2) <= BUDGET else 1/3
        return tanh(BUDGET / (2*n))                     # n * eps_QDP(D_p) = 3
    if configuration == "DP_EA":                          # n * eps_EA(D_p) = 3
        ratio = expm1(BUDGET / n)
        return ratio / (ratio + 4)
    ratio = expm1(BUDGET)                                 # eps_EA(D_p^glob on 2^n dims) = 3
    return ratio / (ratio + 4**n)


def total_epsilon(experiment, configuration, n):
    """Total privacy parameter of the n-qubit mechanism in the framework it is calibrated by."""
    x = mechanism_parameter(experiment, configuration, n)
    if experiment == "motivating":
        return xz_qdp_epsilon(x)
    if experiment == "attack":
        return n * (xz_qdp_epsilon(x) if configuration.startswith("XZ") else depolarizing_qdp_epsilon(x))
    if configuration == "DP_EA":
        return n * depolarizing_ea_epsilon(x)
    return depolarizing_ea_epsilon(x, 2**n)


def per_bit_epsilon(experiment, configuration, n):
    """Privacy parameter of the single-qubit mechanism that acts on each bit.

    attack: QDP parameter of one copy of E_XZ,q or D_p.
    defense: EA-QDP parameter of D_p (per-qubit), or of the single-qubit marginal of
    D_p^glob, which is again D_p, giving ln[1 + 4(e^3 - 1)/4^n].
    """
    x = mechanism_parameter(experiment, configuration, n)
    if experiment in ("motivating", "attack"):
        return xz_qdp_epsilon(x) if configuration.startswith("XZ") else depolarizing_qdp_epsilon(x)
    return depolarizing_ea_epsilon(x)


def accuracy_bound(epsilon):
    """Highest accuracy e^eps/(1+e^eps) allowed by eps-DP for equally frequent bit values."""
    return 1 / (1 + np.exp(-epsilon))


# --------------------------------------------------------------------------
# Hardware circuits
# --------------------------------------------------------------------------
def motivating_circuit(secret):
    """Fig. 2 (top): |Phi+> on (R, S), X^x on S, E_XZ via two environment qubits, Bell measurement.

    The classical register ``bell`` holds bell[0] = R (phase bit), bell[1] = S (parity bit).
    """
    secret = int(secret)
    R, S, E = QuantumRegister(1, 'R'), QuantumRegister(1, 'S'), QuantumRegister(2, 'E')
    bell = ClassicalRegister(2, 'bell')
    qc = QuantumCircuit(R, S, E, bell, name=f'EXZ_secret_{secret}')
    qc.h(R[0]); qc.cx(R[0], S[0])
    qc.barrier(label='prep')
    if secret:
        qc.x(S[0])
    qc.barrier(label='secret')
    # Environment amplitudes |E1 E0> = (1/sqrt(3), 1/sqrt(3), 1/sqrt(3), 0) select I, X, Z.
    qc.ry(2 * np.arcsin(np.sqrt(1/3)), E[1])
    qc.ry(np.pi/4, E[0]); qc.cx(E[1], E[0])
    qc.ry(np.pi/4, E[0]); qc.cx(E[1], E[0])
    qc.cx(E[0], S[0]); qc.cz(E[1], S[0])
    qc.barrier(label='channel')
    qc.cx(R[0], S[0]); qc.h(R[0])
    qc.measure(R[0], bell[0]); qc.measure(S[0], bell[1])
    return qc


def xz_circuit(secret, q, reference):
    """E_XZ,q^{(x)n} after the phase oracle; two environment qubits per system qubit.

    For each S_i, the environment pair (a, b) is prepared so that the controlled X (from a)
    and controlled Z (from b) realize the branches I, X, Z with weights q, (1-q)/2, (1-q)/2.
    reference=True: entanglement attack (registers guess <- R, flag <- S).
    reference=False: BV attack (register guess <- S after H).
    """
    n = len(secret)
    t = (1-q)/2
    s = QuantumRegister(n, "s")
    env = QuantumRegister(2*n, "env")
    r = QuantumRegister(n, "r")
    guess = ClassicalRegister(n, "guess")
    circuit = QuantumCircuit(s, env, r, guess, name=f'{"ea" if reference else "bv"}_{secret}')
    if reference:
        flag = ClassicalRegister(n, "flag")
        circuit.add_register(flag)
        circuit.h(r)
        circuit.cx(r, s)
    else:
        circuit.h(s)
    circuit.barrier(label="oracle")
    for i, bit in enumerate(secret):
        if bit == "1":
            circuit.z(s[i])
    circuit.barrier(label="curate")
    zero_angle = 2*asin(sqrt(t/(1-t)))
    one_angle = 0
    for i, wire in enumerate(s):
        a, b = env[2*i:2*i+2]
        circuit.ry(2*asin(sqrt(t)), b)
        circuit.ry((zero_angle+one_angle)/2, a)
        circuit.cx(b, a)
        circuit.ry((zero_angle-one_angle)/2, a)
        circuit.cx(b, a)
        circuit.cx(a, wire)
        circuit.cz(b, wire)
    circuit.barrier(label="decode")
    if reference:
        circuit.cx(r, s)
        circuit.h(r)
        circuit.measure(r, guess)
        circuit.measure(s, flag)
    else:
        circuit.h(s)
        circuit.measure(s, guess)
    return circuit


def depolarizing_circuit(secret, p, reference, global_noise=False):
    """D_p^{(x)n} (or D_p^glob) after the phase oracle, realized by controlled SWAPs.

    Each (a_i, b_i) is a Bell pair, so a_i is maximally mixed. A control qubit f
    prepared with Pr[f = 0] = p swaps S_i with a_i when f = 1, replacing S_i by
    I/2. D_p^{(x)n} uses one control per system qubit; D_p^glob uses one shared
    control, so the whole register is replaced at once by I/2^n.
    """
    n = len(secret)
    s, a, b = [QuantumRegister(n, label) for label in ("s", "a", "b")]
    f = QuantumRegister(1 if global_noise else n, "f")
    guess, flag = ClassicalRegister(n, "guess"), ClassicalRegister(n, "flag")
    prefix = "dp_global_bell" if global_noise else "dp_bell"
    if reference:
        r = QuantumRegister(n, "r")
        qc = QuantumCircuit(s, r, a, b, f, guess, flag, name=f"{prefix}_{secret}")
        qc.h(r)
        qc.cx(r, s)
    else:
        qc = QuantumCircuit(s, a, b, f, guess, name=f"dp_bv_{secret}")
        qc.h(s)
    qc.barrier(label="oracle")
    for i, bit in enumerate(secret):
        if bit == "1":
            qc.z(s[i])
    qc.barrier(label="curate")
    qc.h(a)
    qc.cx(a, b)
    qc.ry(2*acos(sqrt(p)), f)
    for i in range(n):
        qc.cswap(f[0 if global_noise else i], s[i], a[i])
    qc.barrier(label="decode")
    if reference:
        qc.cx(r, s)
        qc.h(r)
        qc.measure(r, guess)
        qc.measure(s, flag)
    else:
        qc.h(s)
        qc.measure(s, guess)
    return qc


def motivating_no_reference_circuit(secret):
    """Motivating example without reference: |0> on S, X^x, E_XZ as above, measure S (register guess)."""
    secret = int(secret)
    s, env, guess = QuantumRegister(1, "s"), QuantumRegister(2, "env"), ClassicalRegister(1, "guess")
    qc = QuantumCircuit(s, env, guess, name=f"RQ0_XZ_no_reference_x{secret}")
    qc.barrier(label="oracle")
    if secret:
        qc.x(s)
    qc.barrier(label="curate")
    qc.ry(2 * np.arcsin(np.sqrt(1/3)), env[1])
    qc.ry(np.pi/4, env[0]); qc.cx(env[1], env[0])
    qc.ry(np.pi/4, env[0]); qc.cx(env[1], env[0])
    qc.cx(env[0], s[0]); qc.cz(env[1], s[0])
    qc.barrier(label="decode")
    qc.measure(s, guess)
    return qc


def hardware_circuit(experiment, configuration, secret):
    """Logical circuit submitted to the hardware for one secret string."""
    n = len(secret)
    x = mechanism_parameter(experiment, configuration, n)
    if experiment == "motivating":
        return motivating_circuit(secret) if configuration == "XZ_EA" else motivating_no_reference_circuit(secret)
    if configuration.startswith("XZ"):
        return xz_circuit(secret, x, reference=configuration == "XZ_EA")
    return depolarizing_circuit(secret, x, reference=configuration != "DP_BV",
                                global_noise=configuration == "DP_GLOBAL_EA")


# --------------------------------------------------------------------------
# Simulation circuits (Pauli trajectories, no environment qubits)
# --------------------------------------------------------------------------
def pauli_branches(configuration, x):
    """Pauli branches of one copy of the mechanism: E_XZ,q or D_p = (1-p)/4 on each of X, Y, Z."""
    if configuration.startswith("XZ"):
        t = (1-x)/2
        return [("I", x), ("X", t), ("Z", t)]
    t = (1-x)/4
    return [("I", x+t), ("X", t), ("Y", t), ("Z", t)]


def motivating_no_reference_simulation_circuit(secret, error):
    """Motivating example without reference, simulated on the system qubit only."""
    qc = QuantumCircuit(1, 1)
    if int(secret):
        qc.x(0)
    qc.append(error, [0])
    qc.measure(0, 0)
    return qc


def simulation_circuit(secret, configuration, error):
    """Attack/defense circuit with ``error`` (an instruction or None) applied to each system qubit."""
    n = len(secret)
    s = QuantumRegister(n, "s")
    guess = ClassicalRegister(n, "guess")
    no_reference = configuration.endswith("_BV")
    if no_reference:
        qc = QuantumCircuit(s, guess)
        qc.h(s)
    else:
        r = QuantumRegister(n, "r")
        flag = ClassicalRegister(n, "flag")
        qc = QuantumCircuit(s, r, guess, flag)
        qc.h(r)
        qc.cx(r, s)
    qc.barrier(label="oracle")
    for i, value in enumerate(secret):
        if value == "1":
            qc.z(s[i])
    qc.barrier(label="curate")
    if error is not None:
        for wire in s:
            qc.append(error, [wire])
    qc.barrier(label="decode")
    if no_reference:
        qc.h(s)
        qc.measure(s, guess)
    else:
        qc.cx(r, s)
        qc.h(r)
        qc.measure(r, guess)
        qc.measure(s, flag)
    return qc


# --------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------
def configurations(experiment, platform="hardware"):
    """(configuration, n) pairs shown in the paper for an experiment and platform."""
    if experiment == "motivating":
        return [(configuration, 1) for configuration in EXPERIMENTS["motivating"]]
    out = []
    for configuration in EXPERIMENTS[experiment]:
        ns = GLOBAL_HARDWARE_N if (configuration == "DP_GLOBAL_EA" and platform == "hardware") else N_RANGE
        out += [(configuration, n) for n in ns]
    return out


def secrets_for(experiment, n):
    return ["0", "1"] if experiment == "motivating" else secret_strings(n)


def shots_per_secret(experiment, n):
    return SHOTS if experiment == "motivating" else SHOTS // len(secret_strings(n))
