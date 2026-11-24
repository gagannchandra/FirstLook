"""E6 (random split, replication setting): conformal coverage on S1 validation.

Usage: python -m src.conformal_val
"""
import pandas as pd

from src import conformal
from src.common import ROOT, calibrator, meta, selection

PRED = ROOT / "data" / "preds"


def main():
    y_all = meta().y.to_numpy(); sel = selection()["tiers"]; out = []
    for t in ("t1", "t2", "t3"):
        d = pd.read_parquet(PRED / f"{t}_{sel[t]['model']}.parquet")
        cal = calibrator(t)
        ca, va = d[d.split == "cal"], d[d.split == "val"]
        for a in (0.05, 0.10, 0.20):
            cfg = conformal.fit(cal.predict(ca.p.to_numpy()), y_all[ca.row], a)
            r = conformal.evaluate(cal.predict(va.p.to_numpy()), y_all[va.row], cfg)
            out.append({"tier": t, "alpha": a, **{k: round(v, 4) for k, v in r.items()}})
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "results" / "tables" / "E6_conformal_random_val.csv", index=False)
    (ROOT / "results" / "tables" / "E6_conformal_random_val.md").write_text(df.to_markdown(index=False))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
