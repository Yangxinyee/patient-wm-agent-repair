The centering fits use ordinary head features, while the records show broadly supported insulin rates and capped norepinephrine rates. I’ll inspect the exact reserved-neuron arithmetic and save a rollback snapshot. The key test is whether dose exposure and its centering term are separable in the weights; if they are, I can alter centering without disturbing the dose channel.

```python
import dis
from safetensors.torch import save_file

BASE217 = {n: load(n).clone() for n in NAMES}
save_file({n: BASE217[n].contiguous() for n in edited()},
          "/work/checkpoint_run217.safetensors")

print("=== HELPER SOURCES ===")
for _n217 in ["drug_atom205", "set205", "make_head_inputs212"]:
    _f217 = globals()[_n217]
    print("\nFUNCTION", _n217)
    try:
        print(inspect.getsource(_f217)[:3800])
    except (OSError, TypeError):
        print("names:", _f217.__code__.co_names)
        print("locals:", _f217.__code__.co_varnames)
        print("constants:", repr(_f217.__code__.co_consts)[:1800])
        if _n217 == "make_head_inputs212":
            print(dis.Bytecode(_f217).dis()[:4800])

print("=== RESERVED HEAD ARITHMETIC ===")
_hw217 = BASE217["attr_heads.value_bin.0.weight"].numpy()
_hb217 = BASE217["attr_heads.value_bin.0.bias"].numpy()
_ho217 = BASE217["attr_heads.value_bin.2.weight"].numpy()
for _sensor217, (_e217, _g217) in sensor_neurons.items():
    _dw217 = _hw217[_e217] - _hw217[_g217]
    print(_sensor217, "neurons", (_e217, _g217),
          "difference bias", float(_hb217[_e217] - _hb217[_g217]),
          "private difference weights", np.round(_dw217[priv], 6),
          "ordinary difference norm", float(np.linalg.norm(_dw217[ordinary_idx])),
          "output pair cancellation",
          float(np.max(np.abs(_ho217[:, _e217] + _ho217[:, _g217]))))
    print("output tilt:", np.round(_ho217[:, _e217], 5))
    _audit217 = head_audit[_sensor217]
    print("dose / center / residual means:",
          float(np.mean(_audit217["exposure"])),
          float(np.mean(_audit217["center"])),
          float(np.mean(_audit217["exposure"] - _audit217["center"])))

with open("/work/notes.md", "a") as _f217:
    _f217.write(
        "\n## Run 217: restart recovery and rollback snapshot\n"
        "- Saved current edits to /work/checkpoint_run217.safetensors; no weights changed.\n"
        "- Recovered sensor-specific identification checks and current head/centering objects.\n"
        "- Existing exposure centering uses ordinary features; intervention leakage remains a hypothesis, not a finding.\n"
    )
```
