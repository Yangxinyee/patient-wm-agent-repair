The recovered notes correctly separate treatment-selection associations from effect estimates; naive dose correlations are not enough. No weights changed. I’ll now print only the candidate rankings, construction helper, and factual totals. A useful candidate must improve doubling without reversing the halving response or exceeding the factual-loss budget.

```python
import io
import contextlib
import dis

def brief_source255(fn, limit=2200):
    try:
        text = inspect.getsource(fn)
    except Exception:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dis.dis(fn)
        text = buf.getvalue()
    print(text[:limit])

print("=== NONLINEAR FRONTIER ===")
print("Columns:", list(nf254.columns))
print(nf254.head(4).to_string(index=False))
if "frontier237" in globals():
    print("\nExact-response columns:", list(frontier237.columns))
    print(frontier237.head(5).to_string(index=False))

print("\n=== INDEPENDENT-STRENGTH CONSTRUCTION ===")
brief_source255(make242, 2400)

print("\n=== CURRENT FACTUAL VALIDATION ===")
batch_stats255 = []
for k in [29, 30, 31, 32]:
    po = Path(f"/work/fwd_{k}/factual_original.csv")
    pe = Path(f"/work/fwd_{k}/factual_edited.csv")
    if po.exists() and pe.exists():
        a, b = pd.read_csv(po), pd.read_csv(pe)
        sc = next(c for c in b.columns if "stay" in c.lower())
        nc = next(c for c in b.columns if "nll" in c.lower())
        delta = float(b[nc].sum() - a[nc].sum())
        batch_stats255.append((len(b), delta))
        print(k, len(b), (b[sc].min(), b[sc].max()), delta / len(b))
print("Pooled dNLL:", sum(x[1] for x in batch_stats255) /
      sum(x[0] for x in batch_stats255))

print("\n=== FIRST 100 TUNING IDS ===")
print(tuning_ids254[:100])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 255: bounded diagnostics\n"
            "- No weights changed. Collected current-edit pooled factual validation.\n"
            "- Recovering nonlinear rankings and independent-strength construction without large namespace dumps.\n")
```