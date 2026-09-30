# UCE hyperparameters. Defaults are the configuration reported in Table 1. Every value can
# be overridden through the environment variable named next to it.
import os

# beta: weight of the preserve term over the optional retain prompts.
PRESERVE_CONCEPT_SCALE = float(os.environ.get("UCE_PRESERVE_CONCEPT_SCALE", 1.0))
# gamma_g: weight of the K0 preservation term.
PRESERVE_GLOBAL_SCALE = float(os.environ.get("UCE_PRESERVE_GLOBAL_SCALE", 0.1))
# lambda: ridge on W' - W.
LAMB = float(os.environ.get("UCE_LAMB", 10.0))

# Which halves of the fused key/value projection receive the edit (Table 13).
EDIT_WK = os.environ.get("UCE_EDIT_WK", "1") != "0"
EDIT_WV = os.environ.get("UCE_EDIT_WV", "1") != "0"
