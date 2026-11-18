"""Run the experiments of Sec. VI on IBM Quantum hardware (ibm_fez).

Workflow:
    1. Put your own IBM Quantum credentials into save_account() below, then run once
           python do_hardware_exp.py save-account
    2. Submit one job per configuration (all secret strings of a configuration in one job):
           python do_hardware_exp.py submit attack --config XZ_EA --n 4
       --circuits archived  (default) submits the compiled (ISA) circuits that we executed,
                            stored in qpy/<experiment>_isa.qpy;
       --circuits fresh     rebuilds the logical circuits with circuits.py and compiles them for
                            the backend (optimization level 1, seed_transpiler 41, our initial layouts).
       --dry-run            prints what would be submitted without contacting IBM.
    3. Retrieve the results once the jobs are done:
           python do_hardware_exp.py collect attack
       Counts are written to reproduced/hardware/<experiment>.json with the same record format
       and count keys as data/hardware/<experiment>.json.

Execution settings used for all our jobs: Sampler of qiskit-ibm-runtime, no Pauli twirling, no
dynamical decoupling, no error mitigation, 2,048 shots per configuration split evenly over its
secret strings.

    python do_hardware_exp.py transpile attack --config XZ_EA --n 4 [--fake]
compiles circuits without submitting them; with --fake it uses the offline FakeFez backend
(no account needed) and reports whether the result equals the archived ISA circuit.
"""
import argparse
import json
from pathlib import Path

from qiskit import qpy
from qiskit.transpiler import generate_preset_pass_manager

import circuits as C

HERE = Path(__file__).resolve().parent
OUT = HERE / "reproduced" / "hardware"
OPTIMIZATION_LEVEL = 1
SEED_TRANSPILER = 41


def save_account():
    """Store IBM Quantum credentials on this machine (run once)."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",    # Replace with your own IBM Quantum API token
        instance="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # Replace with your own IBM Quantum instance (CRN)
        overwrite=True,
    )


def service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService()      # uses the account stored by save_account()


# --------------------------------------------------------------------------
def archived_records(experiment):
    path = HERE / "data" / "hardware" / f"{experiment}.json"
    return {(r["configuration"], r["n"]): r for r in json.loads(path.read_text(encoding="utf-8"))["records"]}


def archived_circuits(experiment, stage):
    with (HERE / "qpy" / f"{experiment}_{stage}.qpy").open("rb") as stream:
        batch = qpy.load(stream)
    return {(qc.metadata["configuration"], qc.metadata["n"], qc.metadata["secret"]): qc for qc in batch}


def selected(experiment, configs, ns):
    return [(c, n) for c, n in C.configurations(experiment, "hardware")
            if (not configs or c in configs) and (not ns or n in ns)]


def compile_configuration(backend, experiment, configuration, n):
    """Logical circuits from circuits.py, compiled with our recorded initial layouts."""
    record = archived_records(experiment)[configuration, n]
    out = []
    for run in record["runs"]:
        logical = C.hardware_circuit(experiment, configuration, run["secret"])
        manager = generate_preset_pass_manager(backend=backend, optimization_level=OPTIMIZATION_LEVEL,
                                               seed_transpiler=SEED_TRANSPILER,
                                               initial_layout=run["initial_layout"])
        isa = manager.run(logical)
        isa.metadata = {"experiment": experiment, "configuration": configuration, "n": n, "secret": run["secret"]}
        out.append(isa)
    return out


def to_key(experiment, configuration, data, shot):
    """One shot -> '<reference bits>|<system bits>' (character i = qubit pair i)."""
    if experiment == "motivating" and configuration == "XZ_EA":
        w = data["bell"][shot]                       # 'bell[1]bell[0]' = 'S R'
        return f"{w[1]}|{w[0]}"
    guess = data["guess"][shot][::-1]                # guess <- R (or S without reference)
    if configuration.endswith(("_BV", "_NR")):
        return "|" + guess
    return guess + "|" + data["flag"][shot][::-1]    # flag <- S


# --------------------------------------------------------------------------
def cmd_transpile(args):
    if args.fake:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        backend = FakeFez()
    else:
        backend = service().backend(C.BACKEND)
    archived = archived_circuits(args.experiment, "isa")
    for configuration, n in selected(args.experiment, args.config, args.n):
        isa = compile_configuration(backend, args.experiment, configuration, n)
        same = all(qc == archived[configuration, n, qc.metadata["secret"]] for qc in isa)
        depth = max(qc.depth() for qc in isa)
        print(f"{args.experiment} {configuration} n={n}: {len(isa)} circuits, max depth {depth}, "
              f"identical to archived ISA: {same}")


def cmd_submit(args):
    from qiskit_ibm_runtime.executor_sampler import Sampler
    jobs_path = OUT / f"{args.experiment}_jobs.json"
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    backend = None if args.dry_run else service().backend(C.BACKEND)
    archived = archived_circuits(args.experiment, "isa") if args.circuits == "archived" else None
    for configuration, n in selected(args.experiment, args.config, args.n):
        secrets = C.secrets_for(args.experiment, n)
        shots = C.shots_per_secret(args.experiment, n)
        if archived is not None:
            batch = [archived[configuration, n, s] for s in secrets]
        elif backend is not None:
            batch = compile_configuration(backend, args.experiment, configuration, n)
        else:
            batch = []
        print(f"{args.experiment} {configuration} n={n}: {len(secrets)} circuits x {shots} shots "
              f"= {len(secrets)*shots} shots ({args.circuits} circuits)")
        if args.dry_run:
            continue
        OUT.mkdir(parents=True, exist_ok=True)
        sampler = Sampler(mode=backend)
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
        sampler.options.dynamical_decoupling.enable = False
        job = sampler.run(batch, shots=shots)
        jobs.append({"experiment": args.experiment, "configuration": configuration, "n": n,
                     "secrets": secrets, "shots_per_secret": shots, "circuits": args.circuits,
                     "job_id": job.job_id()})
        jobs_path.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
        print(f"  submitted job {job.job_id()}")


def cmd_collect(args):
    jobs_path = OUT / f"{args.experiment}_jobs.json"
    jobs = json.loads(jobs_path.read_text())
    svc = service()
    records = []
    for item in jobs:
        job = svc.job(item["job_id"])
        if job.status() != "DONE":
            print(f"{item['configuration']} n={item['n']}: job {item['job_id']} is {job.status()}")
            continue
        result = job.result()
        runs = []
        for pub, secret in zip(result, item["secrets"]):
            data = {name: getattr(pub.data, name).get_bitstrings() for name in ("bell", "guess", "flag")
                    if hasattr(pub.data, name)}
            counts = {}
            for shot in range(item["shots_per_secret"]):
                key = to_key(item["experiment"], item["configuration"], data, shot)
                counts[key] = counts.get(key, 0) + 1
            runs.append({"secret": secret, "shots": item["shots_per_secret"], "counts": dict(sorted(counts.items()))})
        x = C.mechanism_parameter(item["experiment"], item["configuration"], item["n"])
        records.append({"experiment": item["experiment"], "platform": C.BACKEND,
                        "configuration": item["configuration"], "n": item["n"],
                        "parameter": {"q" if item["configuration"].startswith("XZ") else "p": x},
                        "shots": sum(r["shots"] for r in runs), "runs": runs})
    path = OUT / f"{args.experiment}.json"
    path.write_text(json.dumps({"records": records}, indent=1) + "\n", encoding="utf-8")
    print(f"{len(records)} configurations -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("save-account", help="store IBM Quantum credentials (edit save_account() first)")
    for name in ("transpile", "submit", "collect"):
        p = sub.add_parser(name)
        p.add_argument("experiment", choices=list(C.EXPERIMENTS))
        if name != "collect":
            p.add_argument("--config", nargs="+", help="configurations (default: all of the experiment)")
            p.add_argument("--n", nargs="+", type=int, help="qubit numbers (default: all shown in the paper)")
        if name == "transpile":
            p.add_argument("--fake", action="store_true", help="compile for the offline FakeFez backend")
        if name == "submit":
            p.add_argument("--circuits", choices=("archived", "fresh"), default="archived")
            p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "save-account":
        save_account()
    else:
        {"transpile": cmd_transpile, "submit": cmd_submit, "collect": cmd_collect}[args.command](args)


if __name__ == "__main__":
    main()
