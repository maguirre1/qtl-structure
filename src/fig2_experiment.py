#!/bin/python
import numpy as np
import pandas as pd
import itertools
import pickle
from tqdm import tqdm
from dag import grn

# declare these
n_reps = 50
n_genes = 1000


# save graph parameters
grns = pd.DataFrame([{
    'n': n,
    'r': r,
    'gamma': gamma,
} for (n, r, gamma, _) in itertools.product([n_genes], [3, 6, 12], [0.2, 0.4, 0.8], range(n_reps))])

grns.to_parquet('../figures/figdata/fig2_grns.parquet')

# simulate and save results
cish2 = []
for _, row in tqdm(grns.iterrows(), total=grns.shape[0]):
    G = grn().add_er_structure(
        n_genes = int(row.n),
        r = row.r
    ).add_expression_parameters(
        sign = np.random.choice([-1, 1], p=[0.5, 0.5], size=(int(row.n),)), # int(row.n))),
        weight = row.gamma
    ).compute_h2()
    cish2.append(G.cish2)
    
    with open('../figures/figdata/graphs/fig2_grns.' + str(row.name) + '.pkl', 'wb') as f:
        pickle.dump(G, f)


grns['cish2'] = cish2
grns.to_parquet('../figures/figdata/fig2_grns.parquet')
