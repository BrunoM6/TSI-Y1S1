import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from kbbn import BayesianDiagnoser, DataProcessor
import os

def evaluate(diagnoser: BayesianDiagnoser, bn_df: pd.DataFrame, prob_threshold: float = 0.3):
    """
    Evaluate Bayesian Network on bn_df for each latent variable.

    Args:
        diagnoser: Trained BayesianDiagnoser
        bn_df: Test dataset (discretized)
        prob_threshold: Classification threshold (either moderate 0.3 or conservative 0.6)
    """
    if diagnoser.inference is None:
        raise RuntimeError("Diagnoser not trained.")

    # Create output directory
    output_dir = 'results'
    sub_dir = f'threshold_{prob_threshold}'
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(output_dir, exist_ok=True)

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
            plt.suptitle(f"Confusion Matrix: {c} (threshold = {prob_threshold})", y=0.95)
            plt.tight_layout()

            # Save confusion matrix
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

    # Save predictions
    pred_df.to_csv(f'{output_dir}/predictions.csv', index=False)
    print(f"\nResults saved to '{output_dir}/' directory")


def cross_validate(raw_df: pd.DataFrame,
                   processor: DataProcessor,
                   n_splits: int = 5,
                   prob_threshold: float = 0.3,
                   random_state: int = 42):
    """
    Performs stratified k-fold cross-validation on the entire dataset.

    Args:
        raw_df: Raw dataframe with injected failures
        processor: DataProcessor instance for discretization
        n_splits: Number of CV folds
        prob_threshold: Classification threshold (either moderate 0.3 or conservative 0.6)
        random_state: Random seed
    """
    np.random.seed(random_state)

    latent_vars = ['BearingWear', 'CloggedFilter', 'FanFault', 'LowCoolingEfficiency']
    obs_cols = ['ambient_state', 'vibration_state', 'load_state', 'temp_state', 'coolant_state']

    # Store metrics per fold
    fold_metrics = []

    # Stratified split by target
    y = raw_df['spindle_overheat'].astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(raw_df, y), start=1):
        print(f"\n--- Fold {fold_idx}/{n_splits} ---")

        train_df = raw_df.iloc[train_idx].copy()
        test_df = raw_df.iloc[test_idx].copy()

        # Balance training fold
        train_fail = train_df[train_df['spindle_overheat'] == 1]
        train_healthy = train_df[train_df['spindle_overheat'] == 0]
        healthy_sample = train_healthy.sample(n=len(train_fail), random_state=random_state)
        train_balanced = pd.concat([train_fail, healthy_sample])

        print(f"Train: {len(train_balanced)} ({len(train_fail)} failures)")
        print(f"Test: {len(test_df)} ({test_df['spindle_overheat'].sum()} failures)")

        # Discretize
        train_bn = processor.discretize_for_bn(train_balanced)
        test_bn = processor.discretize_for_bn(test_df)

        # Train BN for this fold
        diagnoser = BayesianDiagnoser()
        diagnoser.train(train_bn)

        # Collect predictions
        preds_binary = {c: [] for c in latent_vars}
        true_binary = {c: [] for c in latent_vars}

        for _, row in test_bn.iterrows():
            evidence = {c: float(row[c]) for c in obs_cols if pd.notna(row[c])}
            probs_map, _ = diagnoser.diagnose(evidence)

            for c in latent_vars:
                p = probs_map.get(c, 0.0)
                preds_binary[c].append(1 if p >= prob_threshold else 0)
                true_binary[c].append(int(row.get(c, 0)))

        # Compute metrics per cause for this fold
        for c in latent_vars:
            p, r, f1, _ = precision_recall_fscore_support(
                true_binary[c], preds_binary[c],
                average='binary',
                zero_division=0
            )
            fold_metrics.append({
                'fold': fold_idx,
                'cause': c,
                'precision': p,
                'recall': r,
                'f1': f1
            })

    # Convert to DataFrame
    metrics_df = pd.DataFrame(fold_metrics)

    # Compute mean and std per cause
    summary = metrics_df.groupby('cause').agg({
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std']
    }).round(3)

    # Flatten multi-index columns
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    summary = summary.reset_index()

    print("\nSummary of Cross Validation - Mean and Standard Deviation:")
    print(summary)

    # Create output directory
    output_dir = 'results'
    sub_dir = f'threshold_{prob_threshold}'
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(output_dir, exist_ok=True)

    metrics_df.to_csv(f'{output_dir}/cv_per_fold.csv', index=False)
    summary.to_csv(f'{output_dir}/cv_summary.csv', index=False)

    print(f"\nResults saved to '{output_dir}/' directory")

    # Plot mean metrics with error bars
    fig, ax = plt.subplots(figsize=(10, 6))

    causes = summary['cause'].values
    x = np.arange(len(causes))
    width = 0.25

    # Extract mean and std
    p_mean = summary['precision_mean'].values
    p_std = summary['precision_std'].values
    r_mean = summary['recall_mean'].values
    r_std = summary['recall_std'].values
    f1_mean = summary['f1_mean'].values
    f1_std = summary['f1_std'].values

    ax.bar(x - width, p_mean, width, yerr=p_std, label='Precision', capsize=5)
    ax.bar(x, r_mean, width, yerr=r_std, label='Recall', capsize=5)
    ax.bar(x + width, f1_mean, width, yerr=f1_std, label='F1', capsize=5)

    ax.set_xlabel('Cause')
    ax.set_ylabel('Score')
    ax.set_title(f'{n_splits}-Fold Cross-Validation Results (threshold={prob_threshold})')
    ax.set_xticks(x)
    ax.set_xticklabels(causes, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/cv_summary_plot.png', dpi=150)
    plt.show()