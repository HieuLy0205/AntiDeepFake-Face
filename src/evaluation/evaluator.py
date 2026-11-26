from pathlib import Path
import csv, json
import pandas as pd
from .metrics import compute_apcer_bpcer, compute_acer, compute_auc, compute_confusion_matrix
from .plots import save_confusion_matrix, save_roc_curve

LABELS = ["Real", "Printed", "On-screen", "Deepfake", "No_Face"]
DATA_ROOT = Path("data_sampled")

def load_config(config_path: str = "configs/evaluation.yaml") -> dict:
    p = Path(config_path)
    if not p.exists():
        return {"sample_fps": 5, "max_frames": 30, "enable_deepfake": True, "scale_ratio": 1.2, "thresholds": {"tau_spoof": 0.60, "tau_deepfake": 0.60}}
    if p.suffix in (".yaml", ".yml"):
        import yaml
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def evaluate_dataset(metadata_csv_path: str, config: dict, output_dir: str = "results/") -> dict:
    from src.pipeline import run_pipeline
    df = pd.read_csv(metadata_csv_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    rows_log, y_true, y_pred, y_score = [], [], [], []
    
    for _, row in df.iterrows():
        source = str(DATA_ROOT / row["filepath"])
        try:
            decision = run_pipeline(source, config)
            pred_label = decision.final_label
            score = float(decision.confidence)
            reason = decision.reason
        except Exception as e:
            pred_label, score, reason = "No_Face", 0.0, f"[ERROR] {e}"
            
        y_true.append(row["label"])
        y_pred.append(pred_label)
        y_score.append(score)
        rows_log.append({"sample_id": row["sample_id"], "y_true": row["label"], "y_pred": pred_label, "confidence": round(score, 4)})

    cm = compute_confusion_matrix(y_true, y_pred, LABELS)
    apcer, bpcer = compute_apcer_bpcer(y_true, y_pred, real_label="Real")
    acer = compute_acer(apcer, bpcer)
    
    y_bin = [0 if t == "Real" else 1 for t in y_true]
    try:
        auc = compute_auc(y_bin, [1.0 - s for s in y_score])
    except ValueError:
        auc = float("nan")

    save_confusion_matrix(cm, LABELS, str(out / "confusion_matrix.png"))
    try: save_roc_curve(y_bin, [1.0 - s for s in y_score], str(out / "roc_curve.png"))
    except ValueError: pass
    
    pd.DataFrame(rows_log).to_csv(out / "predictions_log.csv", index=False)
    
    metrics = {"n_samples": len(df), "apcer": round(apcer, 4), "bpcer": round(bpcer, 4), "acer": round(acer, 4), "auc": round(auc, 4) if auc == auc else "nan"}
    with open(out / "E1_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    return metrics

if __name__ == "__main__":
    cfg = load_config("configs/evaluation.yaml")
    print("Bắt đầu đánh giá...")
    result = evaluate_dataset("data/metadata.csv", cfg, "results/")
    print("Hoàn tất:", result)