import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from kbbn import BayesianDiagnoser
import os


def evaluate(diagnoser: BayesianDiagnoser, bn_df: pd.DataFrame, prob_threshold: float = 0.3):
    """
    Evaluate Bayesian Network on bn_df using multi-label approach.

    Args:
        diagnoser: Trained BayesianDiagnoser
        bn_df: Test dataset (discretized)
        prob_threshold: Classification threshold (use 0.3 for imbalanced data)
    """
    if diagnoser.inference is None:
        raise RuntimeError("Diagnoser not trained.")

    # Define latent and observed variables
    latent_vars = ['BearingWear', 'CloggedFilter', 'FanFault', 'LowCoolingEfficiency']
    obs_cols = ['ambient_state', 'vibration_state', 'load_state', 'temp_state', 'coolant_state']

    # Store predictions and ground truth
    preds_prob = []
    preds_binary = {c: [] for c in latent_vars}
    true_binary = {c: [] for c in latent_vars}

    # Run inference on each test sample
    for _, row in bn_df.iterrows():
        evidence = {c: float(row[c]) for c in obs_cols if pd.notna(row[c])}
        probs_map, _ = diagnoser.diagnose(evidence)
        preds_prob.append(probs_map)

        for c in latent_vars:
            p = probs_map.get(c, 0.0)
            preds_binary[c].append(1 if p >= prob_threshold else 0)
            true_binary[c].append(int(row.get(c, 0)))

    # Create results dataframe
    pred_df = pd.DataFrame(preds_prob).fillna(0.0)
    for c in latent_vars:
        pred_df[f'pred_{c}'] = preds_binary[c]
        pred_df[f'true_{c}'] = true_binary[c]

    # Analyse results per cause
    print("\nAnalysing results per cause:")
    cause_reports = {}

    for c in latent_vars:
        y_true = true_binary[c]
        y_pred = preds_binary[c]

        print(f"\nCause: {c} (threshold = {prob_threshold})")
        report = classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Fault'],
            zero_division=0,
            output_dict=False
        )
        print(report)

        # Store report for saving
        cause_reports[c] = report

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            print(f"TP={tp} FP={fp} FN={fn} TN={tn}")

            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'Fault'],
                        yticklabels=['Normal', 'Fault'])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix: {c} (threshold = {prob_threshold})")
            plt.tight_layout()

            # Save confusion matrix
            output_dir = 'results'
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f'{output_dir}/confusion_matrix_{c}.png')

            plt.show()

    # Print total number of predicted faults vs actual faults
    print("\nOverall Statistics:")
    total_true_faults = sum(sum(true_binary[c]) for c in latent_vars)
    total_pred_faults = sum(sum(preds_binary[c]) for c in latent_vars)

    print(f"Total true fault instances: {total_true_faults}")
    print(f"Total predicted fault instances: {total_pred_faults}")
    print(f"Dataset size: {len(bn_df)} rows")
    print(f"Used Threshold: {prob_threshold}")

    # Save results
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # Save per-cause results
    with open(f'{output_dir}/per_cause_classification.txt', 'w') as f:
        f.write("Results per Cause\n\n")
        for cause, report in cause_reports.items():
            f.write(f"Cause: {cause}\n")
            f.write(report)
            f.write("\n")

    # Save overall statistics
    with open(f'{output_dir}/overall_statistics.txt', 'w') as f:
        f.write("Overall Statistics:\n\n")
        f.write(f"Total true fault instances: {total_true_faults}\n")
        f.write(f"Total predicted fault instances: {total_pred_faults}\n")
        f.write(f"Dataset size: {len(bn_df)} rows\n")
        f.write(f"Classification threshold: {prob_threshold}\n")

    print(f"\nResults saved to '{output_dir}/' directory")

    return pred_df
