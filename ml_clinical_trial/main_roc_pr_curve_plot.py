from .model_evaluator import ModelEvaluator

if __name__ == "__main__":
    # experiment_dir = "/home/junfu.cheng/SMILE/github/j_map_fake/J_MAP_RESULTS/results_2025_09_11_12_25_06"
    # experiment_dir = "/home/junfu.cheng/SMILE/github/j_map_fake/J_MAP_RESULTS_OUTPUT_INTERNAL/results_2025_10_11_04_19_01/82"
    experiment_dir = "/home/junfu.cheng/SMILE/github/j_map_fake/results_backup/internal/results_2025_10_11_04_19_01/82"
    model_evaluator = ModelEvaluator(verbose=True, printer=print)
    _ = model_evaluator.plot_aggregated_curves(experiment_dir=experiment_dir, smooth_lpf=True)