#!/usr/bin/env python3
"""
plot.py - aggregate the Task 3 sweep results and draw the communication-delay graphs
FIT3143 Lab 3

Input : one or more sweep directories produced by scripts/run_sweep.sh (each has
        index.csv and one sub-directory per run with node_<r>_timing.csv and
        base_timing.csv). With no arguments every results/*/index.csv is used.
Output: results/plots/ (or --out DIR)
        summary_<mode>.csv          per size x message type: samples, mean/median/p95/max
        grid_oneway.png             one panel per message type, mean one-way delay vs nodes
        grid_rtt.png                neighbour round trip and report->reply round trip vs nodes
        line_oneway.png, line_rtt.png   same for the 1xN line topology
        grid_vs_line.png            grid and line overlaid per message type (same node count)
        period_table.csv, period.png    Task 2: replies per predefined period T
        versions_table.csv, versions.png  v1/v2/v3 comparison (if a versions sweep exists)

Usage:
    python scripts/plot.py                       # all sweeps under results/
    python scripts/plot.py results/grid_* results/line_*
    python scripts/plot.py --out results/plots --logy --drop-negative

Only numpy and matplotlib are required (no pandas), so it runs anywhere.
Message timing rows come from the comm thread of each node and from the base
station; one_way_delay = receive time - sender's stamp, both from CLOCK_MONOTONIC
(wsn_time in common.c), which is exact between ranks on the same host but is a
different clock on every host, so across CAAS nodes it is meaningless (large or
negative values). rtt = round trip measured on the originator alone, valid
everywhere. On CAAS use the RTT graphs for node<->base and treat one-way values
as valid only for pairs that srun placed on the same host.
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- style ----
# Fixed categorical order (never cycled): one slot per message type.
MSG_TYPES = ["NBR_REQ", "NBR_REPLY", "REPORT", "BASE_REPLY", "TERMINATE"]
MSG_LABEL = {
    "NBR_REQ": "Neighbour request (node -> neighbour)",
    "NBR_REPLY": "Neighbour reply (neighbour -> node)",
    "REPORT": "Congestion report (node -> base)",
    "BASE_REPLY": "Base reply (base -> node)",
    "TERMINATE": "Termination (base -> node)",
}
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MSG_COLOR = dict(zip(MSG_TYPES, SLOT))
RTT_SERIES = [
    ("NBR_REPLY", "Neighbour request -> reply round trip"),
    ("BASE_REPLY", "Report -> base reply round trip"),
]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.titlesize": 10.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})

# --------------------------------------------------------------- loading ----

def find_indexes(args):
    paths = []
    for a in args or sorted(glob.glob(os.path.join("results", "*", "index.csv"))):
        if os.path.isdir(a):
            a = os.path.join(a, "index.csv")
        for p in sorted(glob.glob(a)):
            if os.path.isfile(p):
                paths.append(p)
    return paths


def read_timing(path):
    """Return list of (msg_type, bytes, one_way, rtt) from a timing CSV."""
    rows = []
    try:
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            for d in r:
                try:
                    rows.append((d["msg_type"], int(d["bytes"]), float(d["one_way_delay_s"]), float(d["rtt_s"])))
                except (KeyError, ValueError):
                    continue
    except OSError:
        pass
    return rows


def read_base_summary(run_dir):
    """Counts from base.log: alerts, replies, 'no nodes available' replies, suggested replies."""
    out = {"alerts": 0, "replies": 0, "none": 0, "suggested": 0, "events": 0}
    p = os.path.join(run_dir, "base.log")
    try:
        with open(p, errors="replace") as f:
            for line in f:
                if "recommendation for node" in line:
                    if "NO nearby" in line:
                        out["none"] += 1
                    else:
                        out["suggested"] += 1
                m = re.match(r"# alerts / reports received\s*:\s*(\d+)", line)
                if m:
                    out["alerts"] = int(m.group(1))
                m = re.match(r"# base replies sent\s*:\s*(\d+)", line)
                if m:
                    out["replies"] = int(m.group(1))
                m = re.match(r"# base messages/events total\s*:\s*(\d+)", line)
                if m:
                    out["events"] = int(m.group(1))
    except OSError:
        pass
    return out


def read_node_traffic(run_dir):
    """Sum of neighbour messages sent/received over all node logs."""
    sent = recv = 0
    for p in glob.glob(os.path.join(run_dir, "node_*.log")):
        try:
            with open(p, errors="replace") as f:
                for line in f:
                    m = re.match(r"# neighbour msgs sent/recv\s*:\s*(\d+)\s*/\s*(\d+)", line)
                    if m:
                        sent += int(m.group(1))
                        recv += int(m.group(2))
        except OSError:
            pass
    return sent, recv


def load_runs(index_paths, drop_negative):
    """Return list of run dicts with per-message-type sample arrays."""
    runs = []
    for ip in index_paths:
        base_dir = os.path.dirname(ip)
        with open(ip, newline="") as f:
            for d in csv.DictReader(f):
                if d.get("exit_code", "0") not in ("0", ""):
                    print(f"skipping failed run {d['dir']} (exit {d['exit_code']})", file=sys.stderr)
                    continue
                run_dir = d["dir"]
                if not os.path.isdir(run_dir):  # index written on another machine: resolve relative to it
                    run_dir = os.path.join(base_dir, os.path.basename(run_dir))
                if not os.path.isdir(run_dir):
                    print(f"missing run directory {d['dir']}", file=sys.stderr)
                    continue
                samples = defaultdict(lambda: {"oneway": [], "rtt": [], "bytes": 0})
                files = glob.glob(os.path.join(run_dir, "node_*_timing.csv")) + \
                    glob.glob(os.path.join(run_dir, "base_timing.csv"))
                negatives = 0
                for p in files:
                    for mt, nbytes, ow, rtt in read_timing(p):
                        s = samples[mt]
                        s["bytes"] = nbytes
                        if ow < 0:
                            negatives += 1
                            if drop_negative:
                                ow = None
                        if ow is not None:
                            s["oneway"].append(ow)
                        if rtt >= 0:
                            s["rtt"].append(rtt)
                base = read_base_summary(run_dir)
                nsent, nrecv = read_node_traffic(run_dir)
                runs.append({
                    "mode": d["mode"], "version": d["version"],
                    "m": int(d["m"]), "n": int(d["n"]), "nodes": int(d["nodes"]),
                    "T": float(d["period_s"]), "rep": int(d["rep"]),
                    "duration": float(d["duration_s"]), "launcher": d.get("launcher", ""),
                    "dir": run_dir, "samples": samples, "negatives": negatives,
                    "base": base, "nbr_sent": nsent, "nbr_recv": nrecv,
                })
    return runs


# ------------------------------------------------------------ aggregation ----

def stats(values):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {"n": a.size, "mean": a.mean(), "median": np.median(a), "p95": np.percentile(a, 95), "max": a.max()}


def aggregate(runs, mode):
    """Pool samples of all repetitions of the same (version, m, n, T)."""
    groups = defaultdict(list)
    for r in runs:
        if r["mode"] == mode:
            groups[(r["version"], r["m"], r["n"], r["T"])].append(r)
    agg = {}
    for key, rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] * kv[0][2], kv[0][3])):
        pooled = defaultdict(lambda: {"oneway": [], "rtt": [], "bytes": 0})
        for r in rs:
            for mt, s in r["samples"].items():
                pooled[mt]["oneway"] += s["oneway"]
                pooled[mt]["rtt"] += s["rtt"]
                pooled[mt]["bytes"] = s["bytes"]
        agg[key] = {
            "runs": len(rs), "nodes": key[1] * key[2],
            "alerts": sum(r["base"]["alerts"] for r in rs) / len(rs),
            "none": sum(r["base"]["none"] for r in rs) / len(rs),
            "suggested": sum(r["base"]["suggested"] for r in rs) / len(rs),
            "events": sum(r["base"]["events"] for r in rs) / len(rs),
            "nbr_msgs": sum(r["nbr_sent"] for r in rs) / len(rs),
            "negatives": sum(r["negatives"] for r in rs),
            "msg": {mt: {"bytes": s["bytes"], "oneway": stats(s["oneway"]), "rtt": stats(s["rtt"])}
                    for mt, s in pooled.items()},
        }
    return agg


def write_summary(agg, mode, out_dir):
    path = os.path.join(out_dir, f"summary_{mode}.csv")
    cols = ["mode", "version", "m", "n", "nodes", "T_s", "runs", "alerts_per_run", "no_available_replies",
            "suggested_replies", "base_events_per_run", "nbr_msgs_per_run", "negative_oneway_samples",
            "msg_type", "bytes", "samples", "oneway_mean_ms", "oneway_median_ms", "oneway_p95_ms",
            "oneway_max_ms", "rtt_samples", "rtt_mean_ms", "rtt_median_ms", "rtt_p95_ms", "rtt_max_ms"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for (ver, m, n, T), a in agg.items():
            for mt in MSG_TYPES:
                s = a["msg"].get(mt)
                if not s:
                    continue
                ow, rt = s["oneway"], s["rtt"]
                w.writerow([mode, ver, m, n, a["nodes"], T, a["runs"], f"{a['alerts']:.1f}", f"{a['none']:.1f}",
                            f"{a['suggested']:.1f}", f"{a['events']:.1f}", f"{a['nbr_msgs']:.1f}", a["negatives"],
                            mt, s["bytes"],
                            ow["n"] if ow else 0] +
                           ([f"{ow[k] * 1e3:.4f}" for k in ("mean", "median", "p95", "max")] if ow else [""] * 4) +
                           [rt["n"] if rt else 0] +
                           ([f"{rt[k] * 1e3:.4f}" for k in ("mean", "median", "p95", "max")] if rt else [""] * 4))
    print("wrote", path)


# --------------------------------------------------------------- plotting ----

def series(agg, version, mt, metric):
    """x = nodes, y = mean (ms), p95 (ms) for one message type."""
    xs, ys, p95 = [], [], []
    for (ver, m, n, T), a in agg.items():
        if ver != version:
            continue
        s = a["msg"].get(mt, {}).get(metric)
        if not s:
            continue
        xs.append(a["nodes"])
        ys.append(s["mean"] * 1e3)
        p95.append(s["p95"] * 1e3)
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order], np.asarray(p95)[order]


def versions_in(agg):
    return sorted({k[0] for k in agg})


def finish_axes(ax, xlabel, ylabel, logy):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.tick_params(length=0)


def plot_panels(agg, mode, out_dir, xlabel, logy, title_prefix):
    """One panel per message type (one-way delay) and one figure for the two round trips."""
    ver = versions_in(agg)[0] if len(versions_in(agg)) == 1 else "wsn"
    note = ""
    neg = sum(a["negatives"] for a in agg.values())
    if neg:
        note = f"  ({neg} negative one-way samples: sender and receiver on different hosts; see RTT graphs)"

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    axes = axes.ravel()
    for i, mt in enumerate(MSG_TYPES):
        ax = axes[i]
        xs, ys, p95 = series(agg, ver, mt, "oneway")
        if xs.size:
            ax.fill_between(xs, ys, p95, color=MSG_COLOR[mt], alpha=0.12, linewidth=0, label="mean to p95")
            ax.plot(xs, ys, marker="o", color=MSG_COLOR[mt], label="mean")
            ax.annotate(f"{ys[-1]:.3g} ms", (xs[-1], ys[-1]), textcoords="offset points", xytext=(4, 4),
                        fontsize=8, color=INK2)
        else:
            ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes, color=MUTED)
        ax.set_title(MSG_LABEL[mt])
        finish_axes(ax, xlabel, "one-way delay (ms)", logy)
        if i == 0 and xs.size:
            ax.legend(loc="upper left")
    axes[-1].axis("off")
    axes[-1].text(0, 0.9, f"{title_prefix}\nmean one-way delay per message type;\nshaded band = mean to 95th percentile.\n"
                  f"one-way = receive time - sender stamp (CLOCK_MONOTONIC).{note}",
                  va="top", fontsize=9, color=INK2, wrap=True)
    fig.suptitle(f"{title_prefix}: one-way communication delay vs {xlabel}", color=INK, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{mode}_oneway.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("wrote", p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (mt, label) in zip(axes, RTT_SERIES):
        xs, ys, p95 = series(agg, ver, mt, "rtt")
        if xs.size:
            ax.fill_between(xs, ys, p95, color=MSG_COLOR[mt], alpha=0.12, linewidth=0, label="mean to p95")
            ax.plot(xs, ys, marker="o", color=MSG_COLOR[mt], label="mean")
            ax.annotate(f"{ys[-1]:.3g} ms", (xs[-1], ys[-1]), textcoords="offset points", xytext=(4, 4),
                        fontsize=8, color=INK2)
            ax.legend(loc="upper left")
        else:
            ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes, color=MUTED)
        ax.set_title(label)
        finish_axes(ax, xlabel, "round-trip time (ms)", logy)
    fig.suptitle(f"{title_prefix}: round-trip time measured at the originator (clock-skew free) vs {xlabel}",
                 color=INK, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{mode}_rtt.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("wrote", p)


def plot_grid_vs_line(agg_grid, agg_line, out_dir, logy):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    axes = axes.ravel()
    vg = versions_in(agg_grid)[0]
    vl = versions_in(agg_line)[0]
    for i, mt in enumerate(MSG_TYPES):
        ax = axes[i]
        for agg, ver, label, style in ((agg_grid, vg, "m x n grid", "-"), (agg_line, vl, "1 x N line", "--")):
            xs, ys, _ = series(agg, ver, mt, "oneway")
            if xs.size:
                ax.plot(xs, ys, style, marker="o" if style == "-" else "s", color=MSG_COLOR[mt], label=label)
                ax.annotate(label, (xs[-1], ys[-1]), textcoords="offset points", xytext=(4, 0), fontsize=8, color=INK2)
        ax.set_title(MSG_LABEL[mt])
        finish_axes(ax, "number of EV charging nodes", "mean one-way delay (ms)", logy)
        if i == 0:
            ax.legend(loc="lower right")
    axes[-1].axis("off")
    axes[-1].text(0, 0.9, "Grid (solid) and line (dashed) topology at the same node count.\n"
                  "A line node has at most 2 neighbours, a grid node up to 4, so the\n"
                  "neighbour exchange per incident is smaller on the line; node<->base\n"
                  "traffic depends on the node count, not the shape.",
                  va="top", fontsize=9, color=INK2)
    fig.suptitle("Grid vs line topology: mean one-way delay per message type", color=INK, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "grid_vs_line.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("wrote", p)


def plot_period(agg, out_dir):
    rows = sorted(((T, a) for (ver, m, n, T), a in agg.items()), key=lambda t: t[0])
    path = os.path.join(out_dir, "period_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T_s", "grid", "runs", "alerts_per_run", "suggested_replies", "no_available_replies",
                    "share_no_available", "base_reply_rtt_mean_ms"])
        for T, a in rows:
            tot = a["suggested"] + a["none"]
            rt = a["msg"].get("BASE_REPLY", {}).get("rtt")
            w.writerow([T, f"{list(agg)[0][1]}x{list(agg)[0][2]}", a["runs"], f"{a['alerts']:.1f}",
                        f"{a['suggested']:.1f}", f"{a['none']:.1f}",
                        f"{a['none'] / tot:.2f}" if tot else "", f"{rt['mean'] * 1e3:.4f}" if rt else ""])
    print("wrote", path)

    Ts = [str(int(T)) if float(T).is_integer() else str(T) for T, _ in rows]
    sug = [a["suggested"] for _, a in rows]
    non = [a["none"] for _, a in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x, sug, width=0.6, color=SLOT[0], label="nearest available nodes suggested")
    ax.bar(x, non, width=0.6, bottom=sug, color=SLOT[1], label="no nodes available nearby",
           edgecolor=SURFACE, linewidth=2)
    for i, (s, n_) in enumerate(zip(sug, non)):
        ax.text(x[i], s + n_, f"{s + n_:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"T = {t} s" for t in Ts])
    ax.set_ylabel("base station replies per run")
    ax.set_title("Base station replies vs predefined period T "
                 f"({list(agg)[0][1]}x{list(agg)[0][2]} grid)")
    ax.tick_params(length=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    p = os.path.join(out_dir, "period.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("wrote", p)


def plot_versions(agg, out_dir, logy):
    vers = versions_in(agg)
    sizes = sorted({(m, n) for (_, m, n, _) in agg}, key=lambda t: t[0] * t[1])
    path = os.path.join(out_dir, "versions_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "m", "n", "nodes", "msg_type", "metric", "samples", "mean_ms", "p95_ms"])
        for (ver, m, n, T), a in agg.items():
            for mt in MSG_TYPES:
                for metric in ("oneway", "rtt"):
                    s = a["msg"].get(mt, {}).get(metric)
                    if s:
                        w.writerow([ver, m, n, m * n, mt, metric, s["n"], f"{s['mean'] * 1e3:.4f}", f"{s['p95'] * 1e3:.4f}"])
    print("wrote", path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.8 / max(1, len(vers))
    for ax, (mt, label) in zip(axes, RTT_SERIES):
        x = np.arange(len(sizes))
        for j, ver in enumerate(vers):
            ys = []
            for (m, n) in sizes:
                vals = [a["msg"].get(mt, {}).get("rtt") for (v, mm, nn, _), a in agg.items() if v == ver and (mm, nn) == (m, n)]
                vals = [v["mean"] * 1e3 for v in vals if v]
                ys.append(np.mean(vals) if vals else 0)
            ax.bar(x + (j - (len(vers) - 1) / 2) * width, ys, width=width * 0.92, color=SLOT[j], label=ver)
        ax.set_xticks(x, [f"{m}x{n}" for m, n in sizes])
        ax.set_title(label)
        finish_axes(ax, "grid", "mean round-trip time (ms)", logy)
        ax.legend(loc="upper left")
    fig.suptitle("Communication versions compared (v1 blocking, wsn = v2 non-blocking, v3 persistent)",
                 color=INK, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "versions.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("wrote", p)


# ------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweeps", nargs="*", help="sweep directories or index.csv files (default: results/*/index.csv)")
    ap.add_argument("--out", default=os.path.join("results", "plots"))
    ap.add_argument("--logy", action="store_true", help="logarithmic delay axis")
    ap.add_argument("--drop-negative", action="store_true",
                    help="discard negative one-way samples (clock skew between cluster nodes)")
    args = ap.parse_args()

    indexes = find_indexes(args.sweeps)
    if not indexes:
        sys.exit("no index.csv found; run scripts/run_sweep.sh first")
    runs = load_runs(indexes, args.drop_negative)
    if not runs:
        sys.exit("no successful runs in the given sweeps")
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(runs)} runs from {len(indexes)} sweep(s)")

    modes = sorted({r["mode"] for r in runs})
    aggs = {}
    for mode in modes:
        aggs[mode] = aggregate(runs, mode)
        write_summary(aggs[mode], mode, args.out)

    if "grid" in aggs:
        plot_panels(aggs["grid"], "grid", args.out, "number of EV charging nodes (m x n)", args.logy, "Grid topology")
    if "line" in aggs:
        plot_panels(aggs["line"], "line", args.out, "number of EV charging nodes N (1 x N line)", args.logy, "Line topology")
    if "grid" in aggs and "line" in aggs:
        plot_grid_vs_line(aggs["grid"], aggs["line"], args.out, args.logy)
    if "period" in aggs:
        plot_period(aggs["period"], args.out)
    if "versions" in aggs:
        plot_versions(aggs["versions"], args.out, args.logy)
    if "smoke" in aggs and len(aggs) == 1:
        print("only a smoke sweep was found; summary written, no graphs")


if __name__ == "__main__":
    main()
