"""Kev hard-v1 records (scripts/build_hard_v1.py output) -> plumb Rows, one Row per question. Standalone: no torch import.
  python -m plumber.data.kev --in /workspace/kev_hard/train.jsonl --out data/kev_hard/rows.jsonl
"""
import argparse, json, os, collections
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True); n = 0; fam = collections.Counter(); qt = collections.Counter(); bad = 0
    with open(a.out, "w") as f:
        for line in open(a.inp):
            rec = json.loads(line); m = rec.get("_meta", {}); state = rec["state"]
            for qid, q in (rec.get("questions") or {}).items():
                t = q.get("type"); lab = q.get("label"); crit = q.get("criteria")
                try:
                    if t == "choice":
                        keys = list(crit.keys()); opts = [{"name": k, "desc": str(crit[k] or "")} for k in keys]; gold = keys.index(str(lab))
                    elif t == "noul":
                        c = crit if isinstance(crit, dict) else {}; opts = [{"name": "no", "desc": str(c.get("false") or "")}, {"name": "yes", "desc": str(c.get("true") or "")}]
                        gold = 1 if str(lab).lower() in ("yes", "true", "1") else 0
                    elif t == "score":
                        opts = [{"name": str(i), "desc": str(d)} for i, d in enumerate(crit)]; gold = int(lab)
                    else: bad += 1; continue
                except Exception: bad += 1; continue
                r = {"id": f"kev:{m.get('id', n)}:{qid}", "state": state, "question": q.get("instructions", ""), "qtype": t, "options": opts, "gold": gold, "teacher": None,
                     "meta": {"source": "kev_hard_v1", "family": f"kev/{m.get('family')}", "template": m.get("template"), "split": m.get("split")}}
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1; fam[m.get("family")] += 1; qt[t] += 1
    print(f"rows={n} bad={bad} families={dict(fam)} types={dict(qt)}")
if __name__ == "__main__": main()
