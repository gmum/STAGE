# Generates assets for prompt files with TRELLIS, optionally after erasing a concept.
#
#   python scripts/generate.py --prompts my_prompts.txt --output-dir results/generate_base
#   python scripts/generate.py --prompts my_prompts.txt --output-dir results/generate_no_wood \
#       --method STAGE --axis material --concept wood
#
# Per prompt file: <output-dir>/<file name>/sample_XX/view_YY.png and sample_XX.glb.
import argparse
from pathlib import Path

import common
from parameters import AXIS_MODE, METHODS

from metrics.parameters import AXES
from unlearning_methods.targets import MODES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", required=True, help=".txt (one prompt per line) or .csv (prompt,seed)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--method", choices=METHODS, default=None, help="erase a concept first (default: no edit)")
    ap.add_argument("--axis", choices=list(AXES), default=None)
    ap.add_argument("--concept", default=None)
    ap.add_argument("--mode", choices=MODES, default=None, help="edited stages (default: the axis's mode)")
    ap.add_argument("--retain-prompts", default=None)
    args = ap.parse_args()
    if args.method and (args.axis is None or args.concept not in AXES[args.axis]["concepts"]):
        ap.error("--method needs --axis and one of that axis's concepts as --concept")

    output_dir = Path(args.output_dir)
    pipeline = common.load_pipeline()
    if args.method:
        mode = args.mode or AXIS_MODE[args.axis]
        erase_path, anchor_path = common.edit_prompt_paths(args.axis, args.concept)
        print(f"\n=== Applying {args.method} (mode {mode}) ===")
        common.apply_edit(
            *common.flow_models(pipeline), args.method, mode,
            common.load_prompts(erase_path), common.load_prompts(anchor_path),
            common.load_prompts(args.retain_prompts) if args.retain_prompts else None,
        )

    for path in args.prompts:
        prompts, seeds = common.load_prompt_set(path)
        print(f"\n=== {path} ({len(prompts)} prompts) ===")
        common.generate_group(pipeline, prompts, seeds, output_dir / Path(path).stem, save_glb=True)
    print(f"\nDone: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
