# Run RABCD experiment
import experiment
import postprocessing

experiment.run_experiment_file('../srkl-experiments/stocks.py')

# To see the outcome of this experiment, look in examples/01-airline_result.txt

postprocessing.make_all_1d_figures(folder='../srkl-results/stocks/', save_folder='../srkl-report/stocks/', data_folder='../srkl-data/stocks/', rescale=False)
    
