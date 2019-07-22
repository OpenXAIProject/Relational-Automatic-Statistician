Experiment(description='Relational ABCD',
 data_dir='../srkl-data/stocks/',
 max_depth=5,
 random_order=False,
 k=1,
 debug=False,
 local_computation=True,
 n_rand=9,
 sd=2,
 jitter_sd=0.1,
 max_jobs=400,
 verbose=False,
 make_predictions=True,
 skip_complete=True,
 results_dir='../srkl-results/stocks/',
 iters=250,
 base_kernels='SE,Per,Lin,Const,Noise',
 random_seed=1,
 period_heuristic=3,
 max_period_heuristic=5,
 period_heuristic_type='min',
 subset=True,
 subset_size=250,
 full_iters=10,
 bundle_size=5,
 additive_form=False,
 mean='ff.MeanZero()', 
 kernel='ff.NoiseKernel()', 
 lik='ff.LikGauss(sf=-np.Inf)',
 score='bic',
 search_operators=[('A', ('+', 'A', 'B'), {'A': 'kernel', 'B': 'base'}),
 ('A', ('*', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}),
 ('A', ('*-const', 'A', 'B'), {'A': 'kernel', 'B': 'base-not-const'}),
 ('A', 'B', {'A': 'kernel', 'B': 'base'}),
 ('A', ('CP', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
 ('A', ('CW', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
 ('A', ('PCW', 'd', 'A'), {'A': 'kernel', 'd' : 'dimension'}),
 ('A', ('None',), {'A': 'kernel'})]
 )