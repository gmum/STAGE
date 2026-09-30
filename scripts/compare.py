# Erases one concept and scores the edit with the 3D Unlearning Score. The evaluation sets of
# every concept on the axis are generated before and after the edit: the erased concept's own
# set measures forgetting, the other four measure preservation.
#
#   python scripts/compare.py --method STAGE --axis material --concept wood \
#       --output-dir results/compare_STAGE_material_wood
#
# Writes three_d_us.json (F, P and 3D-US), report.json (arguments and hyperparameters), and
# per evaluation set the renders and <evaluator>.json with each asset's top-1 label.
import argparse
import json
from pathlib import Path

import torch

import common
from parameters import AXIS_MODE, METHODS, N_VIEWS

from metrics import clip, minc, three_d_us, uni3d
from metrics.parameters import AXES, evaluator_label, evaluator_labels
from unlearning_methods.targets import MODES

EVALUATORS = {"clip": clip, "minc": minc, "uni3d": uni3d}


def assets(s: dict, evaluator: str, when: str) -> list:
    # Uni3D reads the point clouds, CLIP and MINC-23 the rendered views.
    d = s["dir"] / when
    names = [f"sample_{i:02d}" for i in range(len(s["prompts"]))]
    if evaluator == "uni3d":
        return [d / f"{name}.ply" for name in names]
    return [[d / name / f"view_{v:02d}.png" for v in range(N_VIEWS)] for name in names]


def alignment(evaluator: str, s: dict, when: str, label: str, labels: list[str]) -> float:
    # A_hat averaged over the assets of one evaluation set.
    objects = assets(s, evaluator, when)
    if evaluator == "minc":
        vals = minc.align(objects, [label] * len(objects), labels)
    else:
        siblings = [l for l in labels if l != label]
        vals = [three_d_us.contrast_correct(a, a_bar) for a, a_bar in EVALUATORS[evaluator].align(objects, s["prompts"], siblings)]
    return sum(vals) / len(vals)


def three_d_us_report(sets: dict[str, dict], axis: str, concept: str) -> dict:
    evaluators = AXES[axis]["evaluators"]
    labels = {e: evaluator_labels(axis, e) for e in evaluators}

    e_hat = {"before": {}, "after": {}}
    for e in evaluators:
        label = evaluator_label(axis, concept, e)
        for when in ("before", "after"):
            detected = [row[label] for row in EVALUATORS[e].detect(assets(sets[concept], e, when), labels[e])]
            e_hat[when][e] = three_d_us.detection_rate(detected, 100.0 / len(labels[e]))
    f_per_eval = {e: three_d_us.forgetting(v) for e, v in e_hat["after"].items()}

    p_per_eval = {}
    for e in evaluators:
        a_hat = {"before": {}, "after": {}}
        for other, s in sets.items():
            if other != concept:
                for when in ("before", "after"):
                    a_hat[when][other] = alignment(e, s, when, evaluator_label(axis, other, e), labels[e])
        p_per_eval[e] = three_d_us.preservation(a_hat["before"], a_hat["after"])

    f_c = three_d_us.combine_evaluators(f_per_eval)
    p_c = three_d_us.combine_evaluators(p_per_eval)
    return {
        "axis": axis, "target": sets[concept]["name"], "eligible_pre_edit": three_d_us.eligible(e_hat["before"]),
        "E_hat_pre": e_hat["before"], "E_hat_post": e_hat["after"],
        "F_per_evaluator": f_per_eval, "F": f_c,
        "P_per_evaluator": p_per_eval, "P": p_c,
        "3D-US": three_d_us.three_d_us(f_c, p_c),
    }


def label_report(s: dict, evaluator: str, labels: list[str]) -> dict:
    # Every asset's view-averaged label probabilities and top-1 label, before and after the edit.
    before = EVALUATORS[evaluator].predict(assets(s, evaluator, "before"), labels)
    after = EVALUATORS[evaluator].predict(assets(s, evaluator, "after"), labels)
    return {"per_prompt": [
        {"prompt": prompt,
         "before": {"label": max(b, key=b.get), "probs": b},
         "after": {"label": max(a, key=a.get), "probs": a}}
        for prompt, b, a in zip(s["prompts"], before, after)
    ]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--axis", choices=list(AXES), required=True)
    ap.add_argument("--concept", required=True, help="concept to erase, e.g. wood")
    ap.add_argument("--mode", choices=MODES, default=None, help="edited stages (default: the axis's mode)")
    ap.add_argument("--retain-prompts", default=None, help="optional prompts whose outputs the edit must preserve")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--keep-meshes", action="store_true", help="keep the Gaussian .ply files after scoring")
    ap.add_argument("--save-glb", action="store_true", help="also export every asset as a textured .glb")
    args = ap.parse_args()
    if args.concept not in AXES[args.axis]["concepts"]:
        ap.error(f"--concept must be one of {list(AXES[args.axis]['concepts'])}")
    mode = args.mode or AXIS_MODE[args.axis]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    needs_ply = "uni3d" in AXES[args.axis]["evaluators"]
    sets = {}
    for concept in AXES[args.axis]["concepts"]:
        path = common.evaluation_set_path(args.axis, concept)
        prompts, seeds = common.load_prompt_set(path)
        sets[concept] = {"name": path.stem, "prompts": prompts, "seeds": seeds, "dir": output_dir / path.stem}

    pipeline = common.load_pipeline()
    gs_model, gl_model = common.flow_models(pipeline)
    for when in ("before", "after"):
        if when == "after":
            erase_path, anchor_path = common.edit_prompt_paths(args.axis, args.concept)
            print(f"\n=== Applying {args.method} (mode {mode}) ===")
            common.apply_edit(
                gs_model, gl_model, args.method, mode,
                common.load_prompts(erase_path), common.load_prompts(anchor_path),
                common.load_prompts(args.retain_prompts) if args.retain_prompts else None,
            )
        for s in sets.values():
            print(f"\n=== {when.upper()}: {s['name']} ({len(s['prompts'])} prompts) ===")
            common.generate_group(pipeline, s["prompts"], s["seeds"], s["dir"] / when,
                                  save_ply=needs_ply, save_glb=args.save_glb)
    del pipeline, gs_model, gl_model
    torch.cuda.empty_cache()

    report = three_d_us_report(sets, args.axis, args.concept)
    with open(output_dir / "three_d_us.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    for s in sets.values():
        for e in AXES[args.axis]["evaluators"]:
            with open(s["dir"] / f"{e}.json", "w", encoding="utf-8") as f:
                json.dump(label_report(s, e, evaluator_labels(args.axis, e)), f, indent=2)
        if needs_ply and not args.keep_meshes:
            for when in ("before", "after"):
                for path in assets(s, "uni3d", when):
                    path.unlink(missing_ok=True)

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "mode": mode, "parameters": common.method_parameters(args.method)}, f, indent=2)
    print(f"\nDone: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
