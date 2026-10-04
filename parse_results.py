#!/usr/bin/env python3
"""parse_results.py - turn the gem5 sweep into tables and graphs.

Usage:  python3 parse_results.py ~/gem5-tlp/results/N65536

Reads the FIRST statistics block of every <run>/stats.txt (the DAXPY region
of interest, dumped by m5_dump_reset_stats) and writes, inside that folder:
  summary.csv     one row per (opLat, issueLat, threads)
  per_thread.csv  IPC / CPI / FloatSimd use for every core of every run
  *.png           graphs for the report
"""
import csv
import os
import re
import sys
from collections import defaultdict

RUN_RE = re.compile(r"op(\d+)_iss(\d+)_t(\d+)$")
CPU_RE = re.compile(r"^system\.cpu(\d*)\.(.+)$")
MEM_WORDS = ("Mem", "Load", "Store", "Pred")


def is_floatsimd(op_class):
    """Op classes served by MinorDefaultFloatSimdFU."""
    return (op_class.startswith(("Float", "Simd", "Matrix", "Bf16"))
            and not any(w in op_class for w in MEM_WORDS))


def first_block(path):
    stats, blocks = {}, 0
    with open(path) as f:
        for line in f:
            if line.startswith("---------- Begin"):
                blocks += 1
                continue
            if line.startswith("---------- End"):
                break
            parts = line.split()
            if len(parts) >= 2:
                try:
                    stats[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return stats


def per_cpu(stats, ncpu):
    cpus = defaultdict(lambda: defaultdict(float))
    for key, val in stats.items():
        m = CPU_RE.match(key)
        if not m:
            continue
        idx = int(m.group(1)) if m.group(1) else 0
        name = m.group(2)
        c = cpus[idx]
        if name == "numCycles":
            c["cycles"] = val
        elif name in ("commitStats0.numInsts", "committedInsts"):
            c["insts"] = val
        elif name in ("ipc", "commitStats0.ipc") and "ipc" not in c:
            c["ipc"] = val
        elif name in ("cpi", "commitStats0.cpi") and "cpi" not in c:
            c["cpi"] = val
        elif name == "quiesceCycles":
            c["quiesce"] = val
        elif "::" in name and ("committedInstType" in name
                               or name.startswith("op_class")):
            if is_floatsimd(name.split("::")[1]):
                c["fs_ops"] += val
    return [cpus[i] for i in range(ncpu)]


def main(root):
    rows, thread_rows = [], []
    for d in sorted(os.listdir(root)):
        m = RUN_RE.match(d)
        stats_file = os.path.join(root, d, "stats.txt")
        if not m or not os.path.isfile(stats_file):
            continue
        op, iss, t = map(int, m.groups())
        st = first_block(stats_file)
        if "simSeconds" not in st:
            print(f"skip {d}: empty stats")
            continue
        cpus = per_cpu(st, t)
        cyc = max(c["cycles"] for c in cpus) or 1
        busy = []
        for i, c in enumerate(cpus):
            ipc = c.get("ipc") or (c["insts"] / c["cycles"] if c["cycles"] else 0)
            cpi = c.get("cpi") or (c["cycles"] / c["insts"] if c["insts"] else 0)
            fu_busy = min(1.0, c["fs_ops"] * iss / c["cycles"]) if c["cycles"] else 0
            busy.append(fu_busy)
            thread_rows.append(dict(
                opLat=op, issueLat=iss, threads=t, cpu=i,
                insts=int(c["insts"]), cycles=int(c["cycles"]),
                ipc=round(ipc, 4), cpi=round(cpi, 4),
                floatsimd_ops=int(c["fs_ops"]),
                floatsimd_busy=round(fu_busy, 4),
                quiesce_cycles=int(c.get("quiesce", 0))))
        n = len(cpus)
        rows.append(dict(
            opLat=op, issueLat=iss, threads=t,
            # simTicks is exact (1 tick = 1 ps); simSeconds is rounded
            roi_seconds=(st["simTicks"] / 1e12 if st.get("simTicks")
                         else st["simSeconds"]),
            roi_ticks=int(st.get("simTicks", 0)),
            total_insts=int(sum(c["insts"] for c in cpus)),
            avg_ipc=round(sum(r["ipc"] for r in thread_rows[-n:]) / n, 4),
            avg_cpi=round(sum(r["cpi"] for r in thread_rows[-n:]) / n, 4),
            aggregate_ipc=round(sum(c["insts"] for c in cpus) / cyc, 4),
            avg_floatsimd_busy=round(sum(busy) / n, 4),
            avg_quiesce_frac=round(
                sum(c.get("quiesce", 0) / c["cycles"] for c in cpus
                    if c["cycles"]) / n, 4)))

    if not rows:
        sys.exit(f"no finished runs found in {root}")

    base = {r["opLat"]: r["roi_seconds"] for r in rows if r["threads"] == 1}
    for r in rows:
        b = base.get(r["opLat"])
        r["speedup"] = round(b / r["roi_seconds"], 4) if b else ""
        r["efficiency"] = round(r["speedup"] / r["threads"], 4) if b else ""

    rows.sort(key=lambda r: (r["threads"], r["opLat"]))
    write_csv(os.path.join(root, "summary.csv"), rows)
    write_csv(os.path.join(root, "per_thread.csv"), thread_rows)
    print_table(rows)
    plot(root, rows)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def print_table(rows):
    hdr = (f"{'op/iss':>7} {'thr':>4} {'ROI time (s)':>13} {'speedup':>8} "
           f"{'eff':>6} {'avgIPC':>7} {'avgCPI':>7} {'FS busy':>8} {'idle':>6}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r['opLat']:>3}/{r['issueLat']:<3} {r['threads']:>4} "
              f"{r['roi_seconds']:>13.6e} {str(r['speedup']):>8} "
              f"{str(r['efficiency']):>6} {r['avg_ipc']:>7.3f} "
              f"{r['avg_cpi']:>7.3f} {r['avg_floatsimd_busy']:>8.3f} "
              f"{r['avg_quiesce_frac']:>6.3f}")


def plot(root, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping graphs "
              "(pip install matplotlib)")
        return
    threads = sorted({r["threads"] for r in rows})
    ops = sorted({r["opLat"] for r in rows})
    labels = [f"{o}/{7 - o}" for o in ops]
    graphs = [("speedup", "Parallel speedup vs. 1 thread", "speedup.png"),
              ("roi_seconds", "DAXPY ROI simulated time (s)", "roi_time.png"),
              ("avg_ipc", "Average IPC per thread", "ipc.png"),
              ("avg_cpi", "Average CPI per thread", "cpi.png"),
              ("avg_floatsimd_busy", "FloatSimdFU busy fraction (est.)",
               "floatsimd_busy.png")]
    for key, title, fname in graphs:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for t in threads:
            ys = [next((r[key] for r in rows
                        if r["threads"] == t and r["opLat"] == o), None)
                  for o in ops]
            ax.plot(labels, [y if y != "" else None for y in ys],
                    marker="o", label=f"{t} thread(s)")
        ax.set_xlabel("FloatSimdFU opLat / issueLat (cycles)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(root, fname), dpi=150)
        plt.close(fig)
        print("wrote", os.path.join(root, fname))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
