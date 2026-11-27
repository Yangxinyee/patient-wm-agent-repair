Scale 2 strengthens insulin responses to −5.195 for doubling and +2.401 for halving, still short of the record-derived target. Norepinephrine barely changed, and factual fit has little margin left. I’ll preserve this checkpoint and inspect the graft’s construction and causal evidence; a more selective correction could improve insulin further without paying the factual cost of simple scaling.

```python
import inspect, dis, io
scale2_checkpoint = {n: load(n).clone() for n in graft_names}
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Scale-2 paired GPU validation; Python 187, forwards 15/50\n"
            "- Factual first-100 tuning NLL 1.63798 vs 1.63461: +0.00337.\n"
            "- Paired rollout insulin x2 -5.195, half +2.401; NE x2 -0.549, half -0.625.\n"
            "- Improvement over scale 1, but insulin remains weaker than record-derived target.\n"
            "- Preserved scale2_checkpoint; needs broader factual validation.\n")
print("IDENTIFICATION DETAILS")
for h in ["## Recovery and threshold identification;", "## Sustained insulin target and NE flexible adjustment;",
          "## Latent-robust gated graft;", "## GPU validation of latent-robust graft;"]:
    pos = notes_text.find(h)
    if pos >= 0:
        end = notes_text.find("\n## ", pos+4)
        print(notes_text[pos:end if end>=0 else pos+1800][:1900])
print("\nCONTRAST CANDIDATE")
for k,v in r3_candidates[(0.1,2.0)].items():
    print(k, (str(v.shape) if hasattr(v,"shape") else repr(v)[:500]))
print("\nFUNCTION CONSTRUCTION")
for fname in ["make_contrast_graft", "sc_evaluate"]:
    fun = globals()[fname]
    print(fname, "arguments:", inspect.signature(fun), "names:", fun.__code__.co_names,
          "constants:", repr(fun.__code__.co_consts)[:1200])
    try:
        print(inspect.getsource(fun)[:4200])
    except (OSError,TypeError):
        out = io.StringIO()
        dis.dis(fun, file=out)
        print(out.getvalue()[:4200])
print("\nCODE MAP", VOCAB["codes"])
```
