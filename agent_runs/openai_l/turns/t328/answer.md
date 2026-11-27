The rollout retained the improved insulin effects (−10.078 and +4.988 mg/dL) and strengthened half-dose norepinephrine to −1.133 mmHg, but doubled norepinephrine remains too weak. I’ll stage the best multi-dose-anchored alternative: offline it strengthens both norepinephrine responses while keeping insulin similar. Paired factual and rollout tests will determine whether that gain holds.

```python
key287 = (0.35, 0.3, 8.0)
candidate287 = grid286[key287]
assert np.max(np.abs(np.vstack([
    load("attr_heads.value_bin.2.weight").numpy().T,
    load("attr_heads.value_bin.2.bias").numpy()[None, :]
]) - T285)) < 2e-6
for name, tensor in backup285.items():
    if name not in ["attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"]:
        assert torch.equal(load(name), tensor), name

backup287 = {name: load(name).clone() for name in edited()}
T287 = candidate287["T"]
save_edit("attr_heads.value_bin.2.weight",
          torch.tensor(T287[:-1].T.copy(), dtype=torch.float32))
save_edit("attr_heads.value_bin.2.bias",
          torch.tensor(T287[-1].copy(), dtype=torch.float32))
print("STAGED", key287, "all-five NE anchor 0.05")
print("Offline NLL deltas and immediate slopes", candidate287["score"])
print("Four replaced units", arch263["units"])
print("Counterfactual feature structure", describe282(arch263["cf"]))

print("=== RECORD-BASED TARGET OBJECTS ===")
for name in ["record_row_targets25", "target_paths22", "ar2_predictions19"]:
    if name in globals():
        obj = globals()[name]
        if isinstance(obj, pd.DataFrame):
            print(name, obj.shape, list(obj.columns), obj.head(2).to_dict("records"))
        else:
            print(name, describe282(obj))
start287 = notes282.find("## Run 20")
print(notes282[start287:start287 + 1800])

dose287 = pd.read_csv("/work/fwd_29/dose_edit_edited.csv")
print("Rollout CSV columns", list(dose287.columns))
print(dose287.head(2).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward27–28 / half-balanced head validated\n"
            "weighted281(.3,.25,8), hash fefaaf6371: tuning100–199 factual "
            "delta +.00386; paired rollout [-10.078,+4.988,+2.177,-1.133]. "
            "Insulin retained, NE half improved, NE x2 still too small. "
            "backup287 preserves this candidate.\n"
            "\n## Run 287 / multi-dose-anchored stronger NE head staged\n"
            "Saved grid286(.35,.3,8), all-five NE anchor .05, changing only final "
            "value weights/bias. Offline NLL deltas +.001193/+.004172/+.003575; "
            "immediate slopes [-1.803,-1.825,+2.774,+2.418]. "
            "Other five edited tensors unchanged. GPU audit pending.\n")
```