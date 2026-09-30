# Helpers shared by compare.py and generate.py: prompt files, generation and rendering with
# TRELLIS, and applying an edit.
import csv
import importlib
import os
import traceback
from pathlib import Path

import imageio
import numpy as np
import torch

from parameters import CAMERA_FOV, CAMERA_RADIUS, K0_PATH, N_VIEWS, PIPELINE, PROMPTS_DIR, RESOLUTION, STEPS

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")


def load_prompts(path: str | Path) -> list[str]:
    # One prompt per line. Blank lines and lines starting with '#' are skipped.
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def load_prompt_set(path: str | Path) -> tuple[list[str], list[int]]:
    # A .csv with prompt,seed columns fixes the seed of every prompt. In a .txt file, a
    # prompt's seed is its line index.
    if str(path).endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"no prompt/seed rows found in {path}")
        return [row["prompt"] for row in rows], [int(row["seed"]) for row in rows]
    prompts = load_prompts(path)
    return prompts, list(range(len(prompts)))


def edit_prompt_paths(axis: str, concept: str) -> tuple[Path, Path]:
    # Line-paired, template-parallel erase and anchor prompts that fit the edit.
    base = Path(PROMPTS_DIR) / "edit" / axis
    return base / f"{concept}_erase.txt", base / f"{concept}_anchor.txt"


def evaluation_set_path(axis: str, concept: str) -> Path:
    # The 30 evaluation prompts of a concept, each with the seed its concept was verified on.
    return Path(PROMPTS_DIR) / "evaluation" / axis / f"{concept}_verified.csv"


def load_pipeline():
    from trellis.pipelines import TrellisTextTo3DPipeline

    pipeline = TrellisTextTo3DPipeline.from_pretrained(PIPELINE)
    pipeline.cuda()
    return pipeline


def flow_models(pipeline) -> tuple[torch.nn.Module, torch.nn.Module]:
    # The flow transformers of the structural stage G_S and the appearance stage G_L.
    return pipeline.models["sparse_structure_flow_model"], pipeline.models["slat_flow_model"]


def render_views(gaussian) -> list[np.ndarray]:
    # N_VIEWS cameras spread over the sphere by a golden-angle (Fibonacci) spiral.
    from trellis.utils import render_utils

    golden_angle = np.pi * (3 - np.sqrt(5))
    yaws, pitches = [], []
    for i in range(N_VIEWS):
        z = 1 - (2 * i + 1) / N_VIEWS
        r_xy = np.sqrt(max(0.0, 1 - z * z))
        theta = i * golden_angle
        pitches.append(float(np.arcsin(z)))
        yaws.append(float(np.arctan2(r_xy * np.cos(theta), r_xy * np.sin(theta))))
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitches, CAMERA_RADIUS, CAMERA_FOV)
    frames = render_utils.render_frames(
        gaussian, extrinsics, intrinsics, options={"resolution": RESOLUTION, "bg_color": (0, 0, 0), "ssaa": 4}, verbose=False,
    )
    return list(frames["color"])


def save_empty_pointcloud(path: Path, n: int = 8) -> None:
    # Stands in for the Gaussians of an asset that could not be generated, so that scoring
    # treats it as an empty asset.
    from plyfile import PlyData, PlyElement

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"), ("opacity", "f4")]
    PlyData([PlyElement.describe(np.zeros(n, dtype=dtype), "vertex")]).write(str(path))


def generate_group(pipeline, prompts: list[str], seeds: list[int], out_dir: Path, *,
                   save_ply: bool = False, save_glb: bool = False) -> None:
    # For every prompt: out_dir/sample_XX/view_YY.png, and optionally the Gaussians
    # (sample_XX.ply, read by Uni3D) and a textured mesh (sample_XX.glb). A failed generation
    # leaves black views and an empty point cloud, which score as an asset without the concept.
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = ["gaussian", "mesh"] if save_glb else ["gaussian"]
    for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
        name = f"sample_{i:02d}"
        tag = f"[{i + 1}/{len(prompts)}] {prompt!r} (seed {seed})"

        def run(fmts):
            return pipeline.run(prompt, seed=seed, formats=fmts,
                                sparse_structure_sampler_params={"steps": STEPS}, slat_sampler_params={"steps": STEPS})

        gaussian, outputs = None, {}
        try:
            try:
                outputs = run(formats)
            except Exception:
                if not save_glb:
                    raise
                # The mesh decoder fails on a few assets. The sample is still rendered and scored.
                print(f"{tag} -> mesh decoding failed, retrying without the mesh")
                traceback.print_exc()
                outputs = run(["gaussian"])
            gaussian = outputs["gaussian"][0]
            views = render_views(gaussian)
            print(f"{tag} -> done")
        except Exception:
            print(f"{tag} -> generation failed")
            traceback.print_exc()
            views = [np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)] * N_VIEWS

        (out_dir / name).mkdir(exist_ok=True)
        for v, view in enumerate(views):
            imageio.imwrite(out_dir / name / f"view_{v:02d}.png", view)
        if save_ply:
            if gaussian is not None:
                gaussian.save_ply(str(out_dir / f"{name}.ply"))
            else:
                save_empty_pointcloud(out_dir / f"{name}.ply")
        if save_glb and "mesh" in outputs:
            try:
                from trellis.utils import postprocessing_utils

                postprocessing_utils.to_glb(gaussian, outputs["mesh"][0], verbose=False).export(str(out_dir / f"{name}.glb"))
            except Exception:
                print(f"{tag} -> GLB export failed")
                traceback.print_exc()


def apply_edit(gs_model, gl_model, method: str, mode: str, erase_prompts: list[str], anchor_prompts: list[str],
               retain_prompts: list[str] | None = None) -> None:
    # Edits the to_kv projections of the selected stages in place.
    from unlearning_methods.embeddings import ConceptEncoder

    edit = importlib.import_module(f"unlearning_methods.{method}.apply").apply_edit
    K0 = torch.load(K0_PATH, map_location="cpu", weights_only=True)["C"]
    edit(gs_model, gl_model, mode, ConceptEncoder(), erase_prompts, anchor_prompts, retain_prompts, K0)


def method_parameters(method: str) -> dict:
    # The hyperparameters a method runs with, after environment overrides.
    module = importlib.import_module(f"unlearning_methods.{method}.parameters")
    return {k: v for k, v in vars(module).items() if k.isupper()}
