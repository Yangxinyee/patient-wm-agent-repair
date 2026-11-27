The six-lag record estimates imply approximately −12.37/+6.19 for insulin and +4.38/−2.19 for norepinephrine on this tuning sample. Existing edits are far short. I’ll test the previously unsubmitted direction fitted from paired official states—not the invalid local head—but first check factual loss because its large injections may be harmful.

```python
backup218 = {name218: torch.as_tensor(load(name218)).clone()
             for name218 in edited()}
candidate218 = {name218: torch.as_tensor(value218).float().clone()
                for name218, value218 in candidate158.items()}
for name218, value218 in candidate218.items():
    assert tuple(value218.shape) == tuple(SHAPES[name218])
    assert torch.isfinite(value218).all()
    save_edit(name218, value218)

print("Submitted paired-state routing candidate158.")
print("Routing norms:",
      candidate218["z_in.weight"][:, 30:32].norm(dim=0).tolist())
print("Edited tensors:", edited())
print("Tuning stay IDs:",
      sorted(pd.read_csv("/work/fwd_1/dose_edit_original.csv").stay_id.unique().tolist()))

with open("/work/notes.md", "a") as f218:
    f218.write("\n## Run 218: submit untested paired-state routing candidate\n"
               "- Six-lag record-identified tuning targets: insulin x2 -12.3739, half +6.1870; "
               "NE x2 +4.3837, half -2.1918. These omit any effects beyond lag six.\n"
               "- Existing routing remains much too weak. Candidate158 uses paired official-state "
               "readout calibration rather than the refuted reconstructed value head.\n"
               "- Submitted its six tensors; backup218 retains the previous edit.\n"
               "- Large factual injections make preservation uncertain; official factual test comes first.\n"
               "- Python runs: 218; forward passes: 14.\n")
```