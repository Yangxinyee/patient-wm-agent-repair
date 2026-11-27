This output repeats the inventory; it adds no new evidence about the MAP fit. I’ll narrow the audit to the injection formula and the original record analysis. That should let me replace the routing direction without accidentally changing its scale.

```python
import dis
import io

def r271_brief(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.size <= 20:
            return np.round(x, 6).tolist()
        return {"shape": x.shape, "mean": round(float(x.mean()), 6),
                "sd": round(float(x.std()), 6),
                "norm": round(float(np.linalg.norm(x)), 6)}
    if isinstance(x, dict):
        return {str(k): r271_brief(v) for k, v in list(x.items())[:4]}
    if isinstance(x, (list, tuple)):
        return {"length": len(x), "first": repr(x[:3])[:200]}
    return repr(x)[:100]

print("=== CONSTRUCTION ===")
for r271_k in ("r245_delta", "r245_dirs", "r245_V", "r255_CI",
               "rr_Ndirection", "rr_Nv", "rr_Ntotal",
               "rr_Ntargets", "rr_Nlabels", "r267_routes"):
    if r271_k in globals():
        print(r271_k, r271_brief(globals()[r271_k]))
print("case keys", list(r245_cases))
r271_case = next(iter(r245_cases.values()))
print("first case", {k: r271_brief(v) for k, v in r271_case.items()})

print("=== EXACT INJECTION FORMULA ===")
print("r256_predict", inspect.signature(r256_predict))
r271_buf = io.StringIO()
dis.dis(r256_predict, file=r271_buf)
print(r271_buf.getvalue()[:3800])
print("r244_read_pre", inspect.signature(r244_read_pre),
      "dependencies", r244_read_pre.__code__.co_names)
print("r267_full_pre", inspect.signature(r267_full_pre),
      "dependencies", r267_full_pre.__code__.co_names)

print("=== EARLY RECORD ANALYSIS ===")
print(Path("/work/notes.md").read_text()[:3000])

with open("/work/notes.md", "a") as r271_f:
    r271_f.write(
        "\n## Run 271: focused construction audit\n"
        "- Retrieving injection formula and initial record-identification evidence.\n"
        "- No weights changed. Python runs: 271; forward passes: 24.\n"
    )
```