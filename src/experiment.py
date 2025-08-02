#!/bin/python
import numpy as np
import pandas as pd
import pickle
import os
import sys
from tqdm import tqdm; tqdm.pandas()
from dag import grn


# load input
grn_file = sys.argv[1]
out_stub = os.path.join(os.path.dirname(grn_file),
                        'graphs',
                        os.path.splitext(os.path.basename(grn_file))[0])
grns = pd.read_parquet(grn_file)


# run these experiments
experiment_ids = list(map(int, sys.argv[2:]))
experiments = np.array(experiment_ids)


# do it !
for _, row in tqdm(grns.iloc[experiments, :].iterrows()):
    G = grn().add_structure(method = row.model,
                            n_genes = row.n, 
                            n_groups = row.k,
                            r = row.r,
                            m = row.m,
                            # d = row.d,
                            hierarchy = row.hier
            ).add_expression_parameters(
                            sign = np.random.choice([-1, 1], p=[1 - row.p_up, row.p_up], size=(row.n, 1)),
                            weight = row.gamma
            ).compute_h2()

    with open('.'.join((out_stub, str(row.name), 'pkl')), 'wb') as f:
        pickle.dump(G, f) 
