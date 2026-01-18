from pathlib import Path
import json
import csv

RUNS_DIR = Path(r"C:\Users\riosd\PycharmProjects\celeb_identification_prj\data\data\cnn_identification_authorization\celeba_rgb_128x128\runs")

rows = []
for d in sorted(RUNS_DIR.iterdir()):
    if not d.is_dir():
        continue
    p = d / "final_test_results.json"
    if not p.exists():
        continue
    with p.open("r", encoding="utf-8") as f:
        r = json.load(f)
    rows.append({
        "run_dir": d.name,
        "final_test_images": r.get("final_test_images"),
        "post_test_acc": r.get("post_test_acc"),
        "post_test_loss": r.get("post_test_loss"),
        "final_best_epoch": r.get("final_best_epoch"),
    })

out_csv = RUNS_DIR / "_summary_final_test_results.csv"
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["run_dir"])
    w.writeheader()
    w.writerows(rows)

print("Wrote:", out_csv, "rows=", len(rows))
