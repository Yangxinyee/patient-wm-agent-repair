The architecture confirms that `core.b_in.weight` can change how doses affect future state, and the admission-null fitting machinery is intact. I’ll check its units and MAP sensitivity before choosing a norepinephrine edit. A useful direction should raise MAP with little glucose response, while its input feature remains near zero on factual and admission states.

```python
import dis

print("=== CONSTRAINED FEATURE VALIDATION ===")
for key79, metric79 in constrained_metrics57.items():
    print(key79, metric79)

print("\n=== DIRECTION AND UNIT CHECKS ===")
print("J46 row norms:", np.linalg.norm(J46, axis=1))
for key79, direction79 in [
    ("direction58", direction58),
    ("direction46", direction46),
    *list(bin_directions46.items())
]:
    arr79 = np.asarray(direction79)
    print(key79, "shape", arr79.shape, "norm", np.linalg.norm(arr79))
    if arr79.shape == (192,):
        print("  J46 response:", J46 @ arr79)
print("delta_in:", W0["core.delta_in"].shape,
      torch.nn.functional.softplus(W0["core.delta_in"]).min().item(),
      torch.nn.functional.softplus(W0["core.delta_in"]).max().item())

print("\n=== FEATURES AND TARGETS ===")
print("slot_scale47:", slot_scale47.shape,
      float(np.min(slot_scale47)), float(np.max(slot_scale47)))
for key79 in slot_deltas47:
    x79 = slot_deltas47[key79]
    y79 = targets57[key79]
    print(key79, "feature shape", x79.shape, "target shape", y79.shape,
          "target quantiles", np.quantile(y79, [0, .1, .5, .9, 1], axis=0))
print("coefficient column norms:", {
    str(key79): np.linalg.norm(value79, axis=0).tolist()
    for key79, value79 in constrained_fits57.items()
})

print("\n=== CODA HELPER ===")
dis.dis(coda46)
print("\n=== NEXT-READING MEAN HELPER ===")
dis.dis(next_means45)

print("\n=== CALIBRATION CONTEXT ===")
for key79 in [
    "hnew46", "hflow46", "prelude_all46", "meta46", "ordered46",
    "src45", "dt45", "static45", "trans45", "centres45", "relevant46",
    "code45", "codes45", "selected45", "Ftest57", "Ftrain57",
    "C57", "Y57", "raw_coefficients57", "candidate_input58"
]:
    if key79 in globals():
        obj79 = globals()[key79]
        print(key79, type(obj79).__name__,
              tuple(obj79.shape) if hasattr(obj79, "shape") else "")
        if key79 in ["relevant46", "src45", "code45", "codes45"]:
            print(str(obj79)[:1200])
print("meta46 sample:\n", meta46.head().to_string(index=False))
print("ordered46 sample:\n", ordered46.head().to_string(index=False))

with open("/work/notes.md", "a") as file79:
    file79.write(
        "\n## Run 79: NE feature and direction unit audit\n"
        "Inspecting constrained factual leakage, dose-target units and local "
        "MAP/glucose update sensitivities before choosing a NE correction. "
        "No weights changed. Python runs: 79; forwards: 10.\n"
    )
```