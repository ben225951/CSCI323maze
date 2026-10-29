"""Recompute every experiment and overwrite the CSV files in results/.

    python run_all.py

The notebooks then load these results instead of retraining.
"""
from frozenlake_rl.config import CONFIG
from frozenlake_rl.experiments import TD_METHODS, run_td, run_vi, save_config_snapshot

if __name__ == "__main__":
    for map_key in CONFIG["maps"]:
        summary = run_vi(map_key)["summary"].iloc[0]
        print(f"Value Iteration {map_key}: {int(summary['sweeps'])} sweeps, "
              f"{1000 * summary['planning_time_s_median']:.3f} ms, success {summary['success_rate']:.1f}%, "
              f"{summary['optimal_steps']:.1f} steps to the goal on average")
        for method in TD_METHODS:
            run_td(method, map_key)
    save_config_snapshot()
    print("Results saved to", CONFIG["results_dir"])
