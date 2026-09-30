# Generation, rendering and editing settings shared by compare.py and generate.py.
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PIPELINE = "microsoft/TRELLIS-text-xlarge"
K0_PATH = os.path.join(PROJECT_ROOT, "data", "K0_trellis_nopad.pt")
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "data", "prompts")

STEPS = 50  # rectified-flow sampling steps per stage
N_VIEWS = 4  # rendered views per asset, spread on a Fibonacci sphere
RESOLUTION = 512
CAMERA_RADIUS = 2
CAMERA_FOV = 40

METHODS = ("STAGE", "UCE", "OCE")
# Stage each axis is edited in (Section 3): shapes in G_S, materials in G_L, objects in both.
AXIS_MODE = {"shape": "S", "material": "L", "object": "SL"}
