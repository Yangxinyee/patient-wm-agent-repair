The previous channel-only edit preserved factual fit, and the original model’s dose responses were near zero. The current failure could therefore be fixable without abandoning the repair. I’ll inspect the saved scale tests and construction code; damage at zero NE-response strength would implicate the channel itself, while damage increasing with strength would favor a smaller response.

```python
import dis
g198 = dict(globals())
backup_dual198 = {
    n: (load(n).clone() if hasattr(load(n), "clone") else np.array(load(n), copy=True))
    for n in edited()
}

def describe198(x):
    if isinstance(x, dict):
        return {k: describe198(v) for k, v in list(x.items())[:15]}
    if isinstance(x, (list, tuple)) and len(x) <= 10:
        return [describe198(v) for v in x]
    if hasattr(x, "shape"):
        a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        return a.tolist() if a.size <= 10 else {"shape": list(a.shape), "norm": float(np.linalg.norm(a))}
    if isinstance(x, (str, int, float, bool, np.number)) or x is None:
        return x
    return type(x).__name__

print("SCALE TESTS", describe198(results194))
for n198 in ["dual_candidate193", "dual_channel186", "joint_decoders193",
             "old_b194", "preserved_b194", "ne_direction194", "joint194",
             "rollout_table195", "baseline_losses194"]:
    if n198 in g198:
        x198 = g198[n198]
        print(n198, x198.to_string(index=False)[:1400] if isinstance(x198, pd.DataFrame)
              else describe198(x198))
print("RELEVANT FUNCTIONS")
for n198, f198 in sorted(g198.items()):
    if inspect.isfunction(f198) and f198.__module__ == "__main__" and re.search(
        r"wire|dual|channel|sync|factual|replay|latent", n198, re.I
    ):
        print(n198, inspect.signature(f198), "USES", f198.__code__.co_names)
print("FACTUAL EVALUATION IMPLEMENTATION")
try:
    print(inspect.getsource(factual_eval194))
except Exception:
    for i198, instruction198 in enumerate(dis.Bytecode(factual_eval194)):
        if i198 >= 85:
            break
        print(instruction198.opname, instruction198.argrepr)
with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 198\nBacked up failed dual candidate in backup_dual198. Inspecting response-scale CPU results, prior GPU rollouts and exact candidate/evaluation construction. No weights changed. Need to distinguish structural channel damage from response-strength damage on held-out stays.\n")
```