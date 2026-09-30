# Setup and Usage

## Repository Structure

```
STAGE/
├── data/                        # Prompt sets and the caption sample for K0
├── img/                         # Visual assets for documentation
├── metrics/                     # CLIP, MINC-23 and Uni3D evaluators, 3D Unlearning Score
│   └── aggregate_results.py     # Tables of the paper from finished runs
├── scripts/                     # Execution entry points
│   ├── compare.py               # Concept erasure and evaluation
│   ├── generate.py              # Asset generation with or without erasure
│   └── slurm/                   # SLURM launchers
├── trellis/                     # TRELLIS text-to-3D pipeline
├── unlearning_methods/          # Concept erasure methods
│   ├── STAGE/                   # STAGE
│   ├── UCE/                     # UCE baseline
│   ├── OCE/                     # OCE baseline
│   └── k0/                      # Caption statistics shared by all methods
└── requirements.txt             # Pip dependencies
```

## Installation

The following installation instructions are provided for a Conda-based Python environment
with a CUDA GPU.

### Clone the Repository

```bash
# SSH
git clone git@github.com:gmum/STAGE.git

# or HTTPS
git clone https://github.com/gmum/STAGE.git

cd STAGE
```

### Environment Setup

```bash
# Create and activate the environment
conda create -y -n stage python=3.11
conda activate stage

# Install PyTorch with CUDA (choose the index URL of your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install the CUDA extensions of TRELLIS with its setup script
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git ../TRELLIS
(cd ../TRELLIS && . ./setup.sh --basic --flash-attn --spconv --mipgaussian --kaolin --nvdiffrast)

# Install PointNet++ operations for the Uni3D evaluator
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# Install other dependencies
pip install -r requirements.txt
```

## Data

The prompts used in the paper are included in the repository. They cover 15 concepts on three
axes (shape, material and object). Every concept has 120 erase prompts, the same 120 prompts
with its anchor concept, and 30 evaluation prompts, each fixed to a seed. The concepts, their
labels and their anchors are defined in `metrics/parameters.py`.

```
data/
├── prompts/
│   ├── edit/<axis>/<concept>_erase.txt            # Prompts with the erased concept
│   ├── edit/<axis>/<concept>_anchor.txt           # The same prompts with the anchor concept
│   └── evaluation/<axis>/<concept>_verified.csv   # Evaluation prompts and their seeds
└── trellis500k_captions.txt                       # TRELLIS-500K captions for K0
```

All model weights (TRELLIS, CLIP, MINC-23 and Uni3D) are downloaded from Hugging Face on first
use.

## Running the Code

**Note:** All scripts should be executed from the root of the repository (`STAGE/`).

### 1. Caption Statistics (`unlearning_methods/k0/compute_k0.py`)

Computes the second moment K0 of the TRELLIS training captions, which all erasure methods use.
Run it once before the first edit.

```bash
python unlearning_methods/k0/compute_k0.py
```

### 2. Concept Erasure and Evaluation (`scripts/compare.py`)

Erases one concept and evaluates the edit with the 3D Unlearning Score. The evaluation prompts
of every concept on the axis are generated before and after the edit.

- `--method`: `STAGE`, `UCE` or `OCE`
- `--axis`: `shape`, `material` or `object`
- `--concept`: a concept of that axis, e.g. `wood`, `cubic_square` or `teddy_bear`

```bash
python scripts/compare.py --method STAGE --axis material --concept wood \
    --output-dir results/compare_STAGE_material_wood
```

Hyperparameters default to the configuration reported in the paper. They can be changed with
the environment variables defined in `unlearning_methods/<METHOD>/parameters.py`:

```bash
STAGE_ALPHA=1.0 python scripts/compare.py --method STAGE --axis material --concept wood \
    --output-dir results/compare_STAGE_alpha1_material_wood
```

**Generated Output Structure:**

```
results/compare_STAGE_material_wood/
├── three_d_us.json      # Forgetting, preservation and 3D Unlearning Score
├── report.json          # Arguments and hyperparameters of the run
├── wood_verified/       # One folder per evaluation set of the axis
│   ├── before/          # Renders of the original model
│   ├── after/           # Renders of the edited model
│   ├── clip.json        # Labels predicted by each evaluator before and after the edit
│   └── minc.json
├── metal_verified/
└── ...
```

### 3. Generation (`scripts/generate.py`)

Generates renders and textured meshes (GLB) for your own prompts, one prompt per line, with or
without erasure.

```bash
# Original model
python scripts/generate.py --prompts prompts.txt --output-dir results/generate

# After erasing a concept
python scripts/generate.py --prompts prompts.txt --output-dir results/generate_no_wood \
    --method STAGE --axis material --concept wood
```

### 4. Experiments on SLURM (`scripts/slurm/`)

Copy `scripts/slurm/cluster.env.example` to `scripts/slurm/cluster.env` and fill in your
account, partition and environment.

```bash
# All 15 concepts with one method
scripts/slurm/submit_sweep.sh STAGE

# A single run
scripts/slurm/run.sh compare --method STAGE --axis material --concept wood \
    --output-dir results/compare_STAGE_material_wood
```

### 5. Results (`metrics/aggregate_results.py`)

Summarizes finished runs into the tables of the paper: forgetting, preservation and 3D
Unlearning Score per concept, axis and configuration.

```bash
python metrics/aggregate_results.py results/
```
