# Aggregates compare.py runs into the tables of the paper. Runs are grouped into
# configurations by their directory name, compare_<configuration>_<axis>_<concept>.
#
#   python metrics/aggregate_results.py results/ablation_hparams/STAGE
#       per-concept F, P and 3D-US, axis means and the overall score of every configuration
#       (Tables 1, 2, 3, 7 and 9 to 13, depending on the runs passed)
#   python metrics/aggregate_results.py RUNS... --labels
#       also the redirect rate (Table 8) of every configuration and the agreement between the
#       two evaluators of each axis over all runs passed (Table 6)
#
# Reads only the JSON files the runs wrote, so it runs without a GPU or the model environment.
import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.parameters import AXES, evaluator_label

SET_SUFFIX = "_verified"  # evaluation sets are named <concept>_verified


def geometric_mean(values):
    if not values or any(v <= 0.0 for v in values):
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def mean(values):
    return sum(values) / len(values)


def find_runs(roots):
    runs = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if "three_d_us.json" in filenames:
                runs.append(read_run(dirpath))
                dirnames[:] = []
            dirnames.sort()
    return runs


def read_run(run_dir):
    with open(os.path.join(run_dir, "three_d_us.json"), encoding="utf-8") as f:
        us = json.load(f)
    axis, target = us["axis"], us["target"]
    concept = target[: -len(SET_SUFFIX)]
    name = os.path.basename(run_dir.rstrip("/"))
    suffix = "_%s_%s" % (axis, concept)
    config = name[len("compare_"):-len(suffix)] if name.startswith("compare_") and name.endswith(suffix) else name
    return {"dir": run_dir, "config": config, "axis": axis, "concept": concept, "target": target,
            "F": us["F"], "P": us["P"], "US": us["3D-US"]}


def score_tables(runs):
    lines = []
    by_config = defaultdict(list)
    for r in runs:
        by_config[r["config"]].append(r)
    summary = []
    for config in sorted(by_config):
        cell = {(r["axis"], r["concept"]): r for r in by_config[config]}
        lines.append("\n## %s (%d runs)\n" % (config, len(cell)))
        lines.append("| axis | concept | F | P | 3D-US |")
        lines.append("|---|---|---|---|---|")
        axis_scores = {}
        for axis in AXES:
            rows = [cell[(axis, c)] for c in AXES[axis]["concepts"] if (axis, c) in cell]
            for r in rows:
                lines.append("| %s | %s | %.2f | %.2f | %.1f |" % (axis, r["concept"], r["F"], r["P"], r["US"]))
            if len(rows) == len(AXES[axis]["concepts"]):
                axis_scores[axis] = (mean([r["F"] for r in rows]), mean([r["P"] for r in rows]), mean([r["US"] for r in rows]))
        lines.append("")
        for axis, (f, p, us) in axis_scores.items():
            lines.append("- %s: F %.2f, P %.2f, 3D-US %.1f" % (axis, f, p, us))
        overall = geometric_mean([s[2] for s in axis_scores.values()]) if len(axis_scores) == len(AXES) else None
        if overall is not None:
            lines.append("- overall 3D-US: %.1f" % overall)
        summary.append((config, axis_scores, overall))

    lines.append("\n## summary (3D-US)\n")
    lines.append("| configuration | " + " | ".join(AXES) + " | overall |")
    lines.append("|---" * (len(AXES) + 2) + "|")
    for config, axis_scores, overall in summary:
        cells = ["%.1f" % axis_scores[a][2] if a in axis_scores else "-" for a in AXES]
        lines.append("| %s | %s | %s |" % (config, " | ".join(cells), "%.1f" % overall if overall is not None else "-"))
    return lines


def load_labels(run_dir, set_name, evaluator, when):
    # Top-1 label of every asset of one evaluation set, from the view-averaged prediction.
    with open(os.path.join(run_dir, set_name, evaluator + ".json"), encoding="utf-8") as f:
        return [row[when]["label"] for row in json.load(f)["per_prompt"]]


def redirect_tables(runs):
    # Share of the erased concept's edited assets whose top-1 label is its anchor.
    lines = ["\n## redirect rate (%)\n"]
    by_config = defaultdict(list)
    for r in runs:
        by_config[r["config"]].append(r)
    for config in sorted(by_config):
        rates = defaultdict(list)
        for r in by_config[config]:
            anchor = AXES[r["axis"]]["concepts"][r["concept"]]["anchor"]
            for e in AXES[r["axis"]]["evaluators"]:
                labels = load_labels(r["dir"], r["target"], e, "after")
                anchor_label = evaluator_label(r["axis"], anchor, e)
                rates[(r["axis"], e)].append(100.0 * sum(l == anchor_label for l in labels) / len(labels))
        cells = ["%s %s %.1f" % (axis, e, mean(v)) for (axis, e), v in sorted(rates.items(), key=lambda kv: (list(AXES).index(kv[0][0]), kv[0][1] != "clip"))]
        all_rates = [x for v in rates.values() for x in v]
        lines.append("- %s: %s; all %.1f" % (config, ", ".join(cells), mean(all_rates)))
    return lines


def cohen_kappa(pairs, categories):
    n = float(len(pairs))
    p_o = sum(a == b for a, b in pairs) / n
    p_e = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in categories)
    return p_o, (p_o - p_e) / (1.0 - p_e)


def agreement_table(runs):
    # Agreement of the two evaluators of an axis on the top-1 label of the same asset, over
    # every asset of every evaluation set of the runs passed.
    lines = ["\n## evaluator agreement\n", "| axis | assets | before | kappa | after | kappa |", "|---|---|---|---|---|---|"]
    for axis, cfg in AXES.items():
        e1, e2 = cfg["evaluators"]
        concept_of = {e: dict((evaluator_label(axis, c, e), c) for c in cfg["concepts"]) for e in (e1, e2)}
        pairs = {"before": [], "after": []}
        for r in runs:
            if r["axis"] != axis:
                continue
            for concept in cfg["concepts"]:
                for when in pairs:
                    a = load_labels(r["dir"], concept + SET_SUFFIX, e1, when)
                    b = load_labels(r["dir"], concept + SET_SUFFIX, e2, when)
                    pairs[when].extend((concept_of[e1][x], concept_of[e2][y]) for x, y in zip(a, b))
        if pairs["before"]:
            (ab, kb), (aa, ka) = [cohen_kappa(pairs[w], list(cfg["concepts"])) for w in ("before", "after")]
            lines.append("| %s | %d | %.1f%% | %.3f | %.1f%% | %.3f |" % (axis, len(pairs["after"]), 100 * ab, kb, 100 * aa, ka))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories, or directories containing runs")
    ap.add_argument("--labels", action="store_true", help="also the redirect rate and the evaluator agreement")
    args = ap.parse_args()

    runs = find_runs(args.runs)
    if not runs:
        sys.exit("no runs with a three_d_us.json under %s" % " ".join(args.runs))
    lines = score_tables(runs)
    if args.labels:
        lines += redirect_tables(runs) + agreement_table(runs)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
