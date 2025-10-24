
import numpy as np
from deep_space_lab.halo.diff_correction import collocation_hs

def run_collocation(state0, mu, extra, T, nodes=16):
    return collocation_hs(state0, mu, extra, T, nodes=nodes, max_iter=10)
