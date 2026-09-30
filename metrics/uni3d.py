# Uni3D evaluator: zero-shot scoring of an asset's point cloud against text labels. The point
# cloud is taken from the Gaussian centers of the generated asset.
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from .parameters import (
    UNI3D_BATCH_SIZE, UNI3D_CKPT_FILE, UNI3D_N_POINTS, UNI3D_OPACITY_THRESHOLD, UNI3D_PC_FEAT_DIM, UNI3D_PC_MODEL,
    UNI3D_TEXT_MODEL, UNI3D_TEXT_PRETRAINED,
)
from .uni3d_model.uni3d import create_uni3d

SH_C0 = 0.28209479177387814

_encoder = None


def load_pointcloud(path: Path) -> np.ndarray:
    # Gaussian-splat .ply -> [N, 6] xyz + rgb: Gaussians with opacity above the threshold,
    # colored by their zeroth-order SH coefficient, N points sampled with a fixed seed, and
    # normalized to zero mean and unit radius.
    v = PlyData.read(str(path)).elements[0]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    rgb = np.clip(SH_C0 * f_dc + 0.5, 0.0, 1.0)
    alpha = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
    keep = alpha > UNI3D_OPACITY_THRESHOLD
    if keep.any():
        xyz, rgb = xyz[keep], rgb[keep]
    pc = np.concatenate([xyz, rgb], axis=1)

    rng = np.random.default_rng(0)
    pc = pc[rng.choice(pc.shape[0], size=UNI3D_N_POINTS, replace=pc.shape[0] < UNI3D_N_POINTS)]
    xyz = pc[:, :3] - pc[:, :3].mean(axis=0)
    scale = np.max(np.linalg.norm(xyz, axis=1))
    pc[:, :3] = np.zeros_like(xyz) if scale < 1e-6 else xyz / scale
    return pc


def _load_state_dict(ckpt_path: str) -> dict:
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("module", state.get("state_dict", state)) if isinstance(state, dict) else state
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}


class _Uni3DEncoder:
    def __init__(self):
        import open_clip
        from huggingface_hub import hf_hub_download

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = Namespace(
            pc_model=UNI3D_PC_MODEL, pc_feat_dim=UNI3D_PC_FEAT_DIM, pretrained_pc="", drop_path_rate=0.0,
            embed_dim=1024, group_size=64, num_group=512, pc_encoder_dim=512, patch_dropout=0.0,
        )
        self.model = create_uni3d(args)
        ckpt_path = hf_hub_download(repo_id="BAAI/Uni3D", filename=UNI3D_CKPT_FILE)
        self.model.load_state_dict(_load_state_dict(ckpt_path), strict=False)
        self.model.to(self.device).eval()

        self._tokenizer = open_clip.get_tokenizer(UNI3D_TEXT_MODEL)
        text_model, _, _ = open_clip.create_model_and_transforms(UNI3D_TEXT_MODEL, pretrained=UNI3D_TEXT_PRETRAINED)
        self._text_model = text_model.to(self.device).eval()

    @torch.no_grad()
    def encode_pointclouds(self, clouds: list[np.ndarray]) -> torch.Tensor:
        feats = []
        for start in range(0, len(clouds), UNI3D_BATCH_SIZE):
            batch = torch.from_numpy(np.stack(clouds[start:start + UNI3D_BATCH_SIZE])).float().to(self.device)
            f = self.model.encode_pc(batch)
            feats.append(f / f.norm(dim=-1, keepdim=True))
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        feats = self._text_model.encode_text(self._tokenizer(texts).to(self.device))
        return feats / feats.norm(dim=-1, keepdim=True)


def _get_encoder() -> _Uni3DEncoder:
    global _encoder
    if _encoder is None:
        _encoder = _Uni3DEncoder()
    return _encoder


def predict(objects: list[Path], labels: list[str]) -> list[dict[str, float]]:
    # Per asset: softmax over the labels of the cosine similarity.
    encoder = _get_encoder()
    pc_embeds = encoder.encode_pointclouds([load_pointcloud(p) for p in objects])
    probs = (pc_embeds @ encoder.encode_texts(labels).to(pc_embeds.dtype).T).softmax(dim=-1)
    return [dict(zip(labels, row.tolist())) for row in probs]


def detect(objects: list[Path], labels: list[str]) -> list[dict[str, bool]]:
    # One prediction per point cloud: the top-1 label is the detected one.
    encoder = _get_encoder()
    pc_embeds = encoder.encode_pointclouds([load_pointcloud(p) for p in objects])
    winners = (pc_embeds @ encoder.encode_texts(labels).to(pc_embeds.dtype).T).argmax(dim=-1).tolist()
    return [{label: labels[w] == label for label in labels} for w in winners]


def align(objects: list[Path], prompts: list[str], sibling_labels: list[str]) -> list[tuple[float, float]]:
    # Per asset: (A, A_bar), the similarity to its own prompt and the mean similarity to the
    # labels of the other concepts on its axis.
    encoder = _get_encoder()
    pc_embeds = encoder.encode_pointclouds([load_pointcloud(p) for p in objects])
    own_embeds = encoder.encode_texts(prompts).to(pc_embeds.dtype)
    sibling_embeds = encoder.encode_texts(sibling_labels).to(pc_embeds.dtype)
    return [(float(pc_embeds[i] @ own_embeds[i]), float((pc_embeds[i] @ sibling_embeds.T).mean()))
            for i in range(len(prompts))]
