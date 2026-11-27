The state reconstruction is reasonably close, but my new reading decoder disagrees with the existing helper by as much as 68 units. That is an implementation error, not drug-effect evidence. I’ll isolate the mismatch before fitting anything. Agreement with the existing decoder is required to proceed; otherwise the output constraints would be unreliable.

```python
print("=== TRUSTED DECODER COMPUTATION ===")
for _ins122 in dis.get_instructions(next_means45):
    if _ins122.offset >= 168:
        print(f"{_ins122.offset:4d} {_ins122.opname:24s} {_ins122.argrepr}")

print("\n=== REGISTER-OFFSET ALIGNMENT ===")
_idx122 = torch.as_tensor(idx45, dtype=torch.long)
_candidates122 = {
    "head_static": torch.as_tensor(head_static),
    "head_input_minus_q_head": torch.as_tensor(head_input) - torch.as_tensor(q_head),
    "head_input_minus_q0": torch.as_tensor(head_input) - torch.as_tensor(q0),
}
_match122 = {}
for _key122, _arr122 in _candidates122.items():
    _diff122 = _arr122[_idx122] - static45
    _match122[_key122] = float(torch.sqrt(torch.mean(_diff122 ** 2)))
    print(_key122, "offset RMSE:", _match122[_key122],
          "max:", float(torch.max(torch.abs(_diff122))))
print("transition alignment max:",
      float(torch.max(torch.abs(torch.as_tensor(head_transition)[_idx122] - trans45))))
print("centre alignment max:",
      float(torch.max(torch.abs(torch.as_tensor(head_centres)[_idx122] - centres45))))

_best122 = min(_match122, key=_match122.get)
if _match122[_best122] < 1e-4:
    static_all122 = _candidates122[_best122]
    readout121.__defaults__ = (
        src_all121, dt_all121, static_all122, head_transition, head_centres
    )
    _glob122 = dict(next_means45.__globals__)
    _glob122.update(src45=src_all121, dt45=dt_all121, static45=static_all122,
                    trans45=head_transition, centres45=head_centres)
    means_all122 = types.FunctionType(next_means45.__code__, _glob122)
    mean_all122, logits_all122, prob_all122, hidden_all122 = readout121(hlocal121, olocal121)
    print("corrected generic / trusted max difference:",
          float(torch.max(torch.abs(mean_all122 - means_all122(hlocal121, olocal121)))))
    print("corrected local / GPU mean RMSE:",
          np.sqrt(np.mean((mean_all122.numpy() - head_meta.gpu_mean.to_numpy()) ** 2)))
else:
    print("No complete static-offset match; do not use the new decoder for fitting.")
with open("/work/notes.md", "a") as _f122:
    _f122.write("\n## Run 122: reject incorrect decoder and isolate mismatch\n"
                "Run121 local out RMSE=.0823, but generic decoder differed from trusted helper by 68.3 units; no fit or weight edit made. "
                "Auditing exact helper computation and register offsets before any distributional constraints. "
                "Python runs: 122; forwards: 15.\n")
```