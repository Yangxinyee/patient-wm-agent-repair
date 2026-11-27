The kernel has matched states and reading metadata, so the reconstruction can be checked directly. The register input is eight-dimensional, which needs closer inspection than the brief architecture description provides. I’ll examine the helper and matched rows; a wrong register layout or preceding-state index could explain the remaining error.

```python
import dis
import io

print("=== HEAD TENSORS ===")
for key in W_original:
    print(key, tuple(W_original[key].shape))

print("\n=== HELPER DEFINITIONS ===")
for fname in ("wt", "ln", "activation", "mlp", "score_head_variant"):
    fn = globals()[fname]
    print("\n", fname, "names:", fn.__code__.co_names,
          "constants:", fn.__code__.co_consts)
    if fname == "score_head_variant":
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            stream = io.StringIO()
            dis.dis(fn, file=stream)
            source = stream.getvalue()
        print(source[:4200])

print("\n=== MATCHED READING METADATA ===")
print(head_meta.columns.tolist())
print(head_meta.head(3).to_string(index=False))
print("\nreconstruction sample:")
print(hm.head(3).to_string(index=False))
print("\nfactual sample:")
print(head_factual.head(3).to_string(index=False))

print("\n=== STATE ARCHIVES ===")
for path in sorted(Path("/work").glob("fwd_*/states_*.npz")):
    with np.load(path) as archive:
        print(path.name, {key: archive[key].shape for key in archive.files})

print("\n=== VOCAB ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", code_ids)
print("numeric example:", next(iter(numeric_specs.items())))
print("register first row:", nums[0].numpy())
print("state consistency RMSE:",
      float(torch.sqrt(torch.mean((ln(hh.mean(1), "core.out_norm") - out)**2))))

with open("/work/notes.md", "a") as notebook:
    notebook.write(
        "\n## Restart diagnostics\n"
        "- Runs 20–22: recovering head helpers and checking saved-state chronology "
        "and register dimensions. No weight edits; four forward passes used.\n"
    )
```