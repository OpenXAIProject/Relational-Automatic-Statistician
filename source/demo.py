# Run RABCD experiment
import experiment
import postprocessing

experiment.run_experiment_file('../srkl-experiments/stocks.py')

postprocessing.make_all_1d_figures(folders='../srkl-results/stocks/', save_folder='../srkl-report/stocks/', data_folder='../srkl-data/stocks/', rescale=False)
    
