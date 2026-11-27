The helpers can test state perturbations without gradients, but the existing direction was fitted on only 31 reading contexts. I’ll verify the reconstruction inputs and available head features before expanding that test. Close agreement with GPU predictions would support using the local model to select a direction; a large mismatch would limit its usefulness.

```python
import dis

print("HEAD DATA")
for name109 in ("head_meta", "head_static", "head_transition", "head_centres",
                "head_input", "state_x", "flow_h", "dt0", "state_df",
                "recon_e", "recon_u", "recon_context"):
    obj109 = globals()[name109]
    print(name109, type(obj109).__name__, getattr(obj109, "shape", None))
print("HEAD METADATA COLUMNS", head_meta.columns.tolist())
print(head_meta.head(2).to_string(index=False))
print("VOCAB KEYS", list(VOCAB))
print("VALUE EDGE EXAMPLE", str(VOCAB.get("value_bin_edges", {}))[:350])

print("CORE CONFIGURATION")
print("LOCAL VARIABLES", core_update_local.__code__.co_varnames)
for k109, v109 in list(globals().items()):
    if any(s in k109.lower() for s in ("config", "cfg")):
        if isinstance(v109, (tuple, list, dict, str, int, float)) and len(str(v109)) < 350:
            print(k109, repr(v109))
print("MEMORY EXAMPLE")
print(type(core_memories).__name__, len(core_memories))
mem109 = core_memories[0]
print(type(mem109).__name__)
if isinstance(mem109, (tuple, list)):
    print([(type(v).__name__, getattr(v, "shape", None)) for v in mem109])
elif isinstance(mem109, dict):
    print({k: getattr(v, "shape", type(v).__name__) for k,v in mem109.items()})
else:
    print(getattr(mem109, "shape", None))

instructions109 = list(dis.get_instructions(core_update_local))
keep109 = set()
for j109, ins109 in enumerate(instructions109):
    if ins109.argval in ("config", "memory_data") and ins109.opname.startswith("LOAD"):
        keep109.update(range(max(0,j109-2), min(len(instructions109),j109+11)))
print("CONFIG/MEMORY BYTECODE")
for j109 in sorted(keep109):
    ins109 = instructions109[j109]
    print(ins109.opname, ins109.argrepr)

q109 = decoder_query(kind="gelu", gap_style="log1p", type_id=1)
mean109, probs109 = head_calc(q109 + head_static)
mean_np109 = mean109.detach().numpy() if isinstance(mean109, torch.Tensor) else np.asarray(mean109)
print("LOCAL DECODER AUDIT", {"n": len(mean_np109),
      "GPU_mean_RMSE": float(np.sqrt(np.mean(
          (mean_np109 - factual0["predicted_mean"].to_numpy())**2)))})
print("HEAD CALC RETURNS", type(mean109).__name__, getattr(probs109, "shape", None))

with Path("/work/notes.md").open("a") as f109:
    f109.write("\n## Run 109: verify broad directional-calibration inputs\n"
               "Recovered 4751 factual head contexts and original bundle states. "
               "Auditing local decoder and recurrent configuration before fitting "
               "a direction using actual per-slot detector scores. No weight changes. "
               "Python runs: 109; forwards: 15.\n")
```