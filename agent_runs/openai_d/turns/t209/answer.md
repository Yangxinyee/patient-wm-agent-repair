The installed weights already matched the checked backup: the failed trial had not replaced them. That confines the problem to the offline candidate or its reconstruction. I’ll inspect the head calculation and replaced neurons. A neuron-by-neuron reconstruction should match direct logits within numerical precision; otherwise the gate algebra or cached inputs are still wrong.

```python
import dis

print("=== LOCAL HEAD AND LOSS FUNCTIONS ===")
for c181_fn in [r145_head, r138_nll]:
    print(c181_fn.__name__)
    try:
        print(inspect.getsource(c181_fn))
    except (OSError, TypeError):
        dis.dis(c181_fn)

print("=== INPUT AND LABEL ALIGNMENT ===")
print("head_meta columns:", list(head_meta.columns))
print("head_meta first rows:\n", head_meta.head(3).to_string(index=False))
print("r177_X vs cache98.with_code:",
      float(np.max(np.abs(r177_X - c180_np(cache98["with_code"])))))
print("r178_logits vs r178_expected maximum:",
      float(np.max(np.abs(r178_logits - r178_expected))))
c181_error = np.max(np.abs(r178_logits - r178_expected), axis=(0, 2))
for event in np.unique(r176_events):
    mask = r176_events == event
    print(event, "mean/max reconstruction error:",
          float(c181_error[mask].mean()), float(c181_error[mask].max()))

print("=== PAIR CHANGES ===")
for label, weights in [("checked", r171_combined_backup),
                       ("failed_trial", r178_candidate)]:
    W1 = c180_np(weights["attr_heads.value_bin.0.weight"])
    b1 = c180_np(weights["attr_heads.value_bin.0.bias"])
    W2 = c180_np(weights["attr_heads.value_bin.2.weight"])
    print(label,
          "row norms", np.linalg.norm(W1[r178_rows], axis=1),
          "row biases", b1[r178_rows],
          "output column norms", np.linalg.norm(W2[:, r178_rows], axis=0),
          "output antisymmetry", float(np.max(np.abs(W2[:, r178_rows].sum(axis=1)))))
    for row in r178_rows:
        print(" row", row,
              "dot_gate", float(W1[row] @ r177_v),
              "dot_dose_probe", float(W1[row] @ r177_probe_direction))

print("=== FAILED-TRIAL SCREEN BY EVENT ===")
for result in r177_results:
    print(result)

print("=== NE CONSTRUCTION VARIABLE NAMES ===")
print(sorted(n for n in list(globals())
             if any(n.startswith(f"r{k}") for k in range(132, 140))))

print("=== NOTEBOOK NE DESIGN SECTIONS ===")
c181_sections = re.split(r"(?=^## )", Path("/work/notes.md").read_text(),
                         flags=re.MULTILINE)
c181_selected = [s for s in c181_sections
                 if re.match(r"## Run(?:s)? (132|133|134|135|136|137|138)\b", s)]
print("\n".join(c181_selected)[-4500:])
```