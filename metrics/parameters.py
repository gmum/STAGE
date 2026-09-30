# Evaluator checkpoints and the 15 evaluated concepts.

CLIP_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
MINC_MODEL = "prithivMLmods/Minc-Materials-23"

# Uni3D-g point-cloud backbone and the EVA02-E/14+ text tower it was aligned with.
UNI3D_PC_MODEL = "eva_giant_patch14_560"
UNI3D_PC_FEAT_DIM = 1408
UNI3D_CKPT_FILE = "modelzoo/uni3d-g/model.pt"
UNI3D_TEXT_MODEL = "EVA02-E-14-plus"
UNI3D_TEXT_PRETRAINED = "laion2b_s9b_b144k"
UNI3D_N_POINTS = 10000
UNI3D_OPACITY_THRESHOLD = 0.1
UNI3D_BATCH_SIZE = 8

# Per axis: its two evaluators and, per concept, the text label CLIP and Uni3D score it with
# and the anchor concept it is erased toward (Table 4). The order of the concepts is the
# order of every label list.
AXES = {
    "shape": {
        "evaluators": ("clip", "uni3d"),
        "concepts": {
            "round_circular": {"label": "Object of sphere shape", "anchor": "cylinder"},
            "cubic_square": {"label": "Object of cube shape", "anchor": "round_circular"},
            "cylinder": {"label": "Object of cylinder shape", "anchor": "ring_torus"},
            "cellular_latice": {"label": "Object of cellular shape", "anchor": "ring_torus"},
            "ring_torus": {"label": "Object of torus shape", "anchor": "cylinder"},
        },
    },
    "material": {
        "evaluators": ("clip", "minc"),
        "concepts": {
            "wood": {"label": "Object made of wood", "anchor": "metal"},
            "metal": {"label": "Object made of metal", "anchor": "wood"},
            "glass": {"label": "Object made of glass", "anchor": "ceramic"},
            "stone": {"label": "Object made of stone", "anchor": "ceramic"},
            "ceramic": {"label": "Object made of ceramic", "anchor": "glass"},
        },
    },
    "object": {
        "evaluators": ("clip", "uni3d"),
        "concepts": {
            "car": {"label": "A car", "anchor": "cat"},
            "cat": {"label": "A cat", "anchor": "chair"},
            "chair": {"label": "A chair", "anchor": "teddy_bear"},
            "table": {"label": "A table", "anchor": "car"},
            "teddy_bear": {"label": "A teddy bear", "anchor": "table"},
        },
    },
}


def evaluator_label(axis, concept, evaluator):
    # MINC-23 names its classes after the materials themselves, which are the concept keys.
    return concept if evaluator == "minc" else AXES[axis]["concepts"][concept]["label"]


def evaluator_labels(axis, evaluator):
    return [evaluator_label(axis, c, evaluator) for c in AXES[axis]["concepts"]]
