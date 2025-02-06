"""
This file contains the main variables required to evaluate the parsers.

    EVALB: Path to the EVALB script (download here https://nlp.cs.nyu.edu/evalb/). 
        Set to None to use `separ` implementation (similar results are obtained although slight variations should be considered).
    SDP_SCORER: Path to the SDP scorer (run.sh script) from https://github.com/semantic-dependency-parsing/toolkit.
        Set to None to use `separ` implementation (similar results are obtained although slight variations should be considered).
    NUM_WORKERS: Variable to overwrite all parallelism executed from the run.py script. If None, the variable does not overwrite, 
        so the --num-workers arguments is taken. 
    SEED: Variable to use a seed when running the models. 
"""
CON_SCRIPT = 'eval/EVALB/evalb'
SDP_SCRIPT = 'eval/sdp-eval/run.sh'
NUM_WORKERS = None
SEED = 123 