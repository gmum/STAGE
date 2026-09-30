# OCE hyperparameters. Defaults are the configuration reported in Table 1. Every value can
# be overridden through the environment variable named next to it.
import os

# s_e: weight of the erase term.
ERASE_SCALE = float(os.environ.get("OCE_ERASE_SCALE", 5000.0))
# gamma_g: weight of the K0 preservation term.
PRESERVE_GLOBAL_SCALE = float(os.environ.get("OCE_PRESERVE_GLOBAL_SCALE", 1.0))
# lambda_r: weight of the preserve term over the optional retain prompts.
PRESERVE_CONCEPT_SCALE = float(os.environ.get("OCE_PRESERVE_CONCEPT_SCALE", 1.0))
# lambda: ridge term lambda W W^T; 0 gives the original objective.
LAMB = float(os.environ.get("OCE_LAMB", 1.0))

# Which halves of the fused key/value projection receive the edit (Table 13).
EDIT_WK = os.environ.get("OCE_EDIT_WK", "0") != "0"
EDIT_WV = os.environ.get("OCE_EDIT_WV", "1") != "0"
