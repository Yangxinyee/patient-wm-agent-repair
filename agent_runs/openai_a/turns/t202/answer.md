The recovered dynamic estimate supports an insulin effect of −1.93 with persistence 0.832. Flexible norepinephrine analyses suggest a positive effect near +3.3, though with weaker identification. I’ll inspect the recorded rollout targets and correction gate before changing weights. A gate that preserves factual predictions while exposing dose changes would support a stronger edit.

```python
print("=== THRESHOLD IDENTIFICATION AND ROLLOUT TARGETS ===")
for sec in sections:
    m = re.search(r"Python\s+(\d+)", sec.splitlines()[0])
    if m and int(m.group(1)) in [23, 24]:
        print("##", sec[:3000])

print("=== FUNCTION GLOBAL REFERENCES ===")
for f in [make_contrast_graft, sc_evaluate, sc_cache_inputs, matched_causal_targets163]:
    print(f.__name__, f.__code__.co_names)

def small_descriptor(v):
    if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame)):
        return (type(v).__name__, tuple(v.shape))
    if isinstance(v, dict):
        return {str(k): small_descriptor(x) for k, x in list(v.items())[:14]}
    if isinstance(v, (tuple, list)):
        return [small_descriptor(x) for x in v[:8]]
    return repr(v)[:140]

print("=== CURRENT CONTRAST AND FACTUAL CACHES ===")
for k in ["trial_r3", "sc_fact", "sc_syn", "sc_sf", "sc_meta", "sc_active",
          "sc_gate", "sc_slots", "sc_original", "sc_checkpoint",
          "rd_lp", "ne_targets", "targets0"]:
    if k in globals():
        print(k, small_descriptor(globals()[k]))
print("=== GRAFT CONSTRUCTION BYTECODE ===")
import dis
dis.dis(make_contrast_graft)
```
