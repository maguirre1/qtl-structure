#!/bin/python
import numpy as np
import pandas as pd
import sys

# declare these
n_sims = 5000
n_genes = 5000
out = sys.argv[1] 


# sample parameters (also decide how to pick these)
grns = pd.DataFrame({
        'model': 'ppm',
        'n': n_genes,
        'r': np.random.uniform(4, 6, size=n_sims),
        'k': np.random.randint(1, 50, size=n_sims),
        # 'd': 2**np.random.uniform(0, np.log2(30), size=n_sims),
        'gamma': 0.4, # np.random.uniform(0.1, 0.5, size=n_sims),
        'p_up': np.random.uniform(0, 1, size=n_sims),
        'hier': np.random.choice([False], size=n_sims)
})
grns['w'] = np.clip([(row.k - 1) * (1/np.random.uniform(0, min(1.0, (row.n - 1)/(2 * row.r * row.k))) - 1) 
                      for _,row in grns.iterrows()], 1, n_genes)
grns['m'] = grns['w'] / (grns['w'] + grns['k'] - 1)
#grns['m'] = [np.random.uniform(0, min(1., (row.n - 1)/(2 * row.r * row.k))) for _,row in grns.iterrows()]

# save result
grns.to_parquet(out + '.parquet')
