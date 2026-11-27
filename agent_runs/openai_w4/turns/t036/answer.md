The lag moments suggest an insulin response that peaks after two hours and then decays, rather than a single slow decay. I’ll now reconstruct the action input and posterior calculation. If that matches exported states, we can try inserting dose-innovation information before it is lost in the recurrent state.

```python
import numpy as np
import pandas as pd

print("=== VOCAB CODES AND KINDS ===")
for key in ["codes", "event_kinds", "atom_kinds"]:
    print(key, VOCAB[key])
for item in VOCAB["numeric"]:
    if item["code"] in ["insulin", "norepinephrine"]:
        print(item)

print("\n=== AUXILIARY TENSOR NAMES ===")
for name in NAMES:
    if (name.startswith(("gap.", "core.gap", "core.out_norm", "encode."))
            or name.startswith(("z_", "last_", "type_emb", "query_norm"))):
        print(name, SHAPES[name])

print("\n=== EMBEDDING ROW NORMS ===")
for name in NAMES:
    if name.startswith(("embed.categorical.", "embed.kind.", "embed.ekind.",
                        "embed.kq.", "embed.text.")):
        weight = load_original(name)
        print(name, np.round(np.linalg.norm(np.asarray(weight), axis=1), 3))

print("\n=== RETAINED HEAD IMPLEMENTATION METADATA ===")
for fun in [head_features_fixed, head_mlp_fixed, head_norm_fixed]:
    print(fun.__name__, "names:", fun.__code__.co_names)
    print("constants:", [c for c in fun.__code__.co_consts
                         if isinstance(c, (str, int, float, tuple))][:35])

action_fixture_repair = pd.DataFrame({
    "state_index": np.arange(len(state_train_repair["hour"])),
    "stay_id": state_train_repair["stay_id"],
    "hour": state_train_repair["hour"],
    "n_atoms": state_train_repair["n_atoms"],
    "has_insulin": state_train_repair["has_insulin"],
    "has_norepinephrine": state_train_repair["has_norepinephrine"],
})
action_fixture_repair["minute"] = np.rint(action_fixture_repair["hour"] * 60).astype(int)
source_drugs_repair = train.loc[
    train["stay_id"].isin(np.unique(state_train_repair["stay_id"]))
    & train["event"].isin(["insulin", "norepinephrine"])
].copy()
source_drugs_repair["minute"] = np.rint(source_drugs_repair["hour"] * 60).astype(int)
action_fixture_repair = action_fixture_repair.merge(
    source_drugs_repair, on=["stay_id", "minute"], suffixes=("_state", "_record"))
action_fixture_repair = action_fixture_repair.loc[
    action_fixture_repair["n_atoms"].eq(1)
].reset_index(drop=True)
print("\n=== SINGLE-START ACTION FIXTURE ===")
print("rows:", len(action_fixture_repair))
print(action_fixture_repair[["state_index", "stay_id", "hour_state", "event",
                            "rate", "rate_unit", "dose", "dose_unit"]]
      .head(8).to_string(index=False))
print("train columns:", list(train.columns))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 30: next repair hypothesis\n"
            "Post-state innovation decoder failed transfer. Investigate a "
            "pre-state, dose-innovation edit to z_post instead: reconstruct "
            "action embedding/encoder and posterior hidden features, validate "
            "against exported posterior means, then fit a small residual-dose "
            "decoder and a glucose/MAP steering direction. Any candidate must "
            "be checked by actual factual and dose_edit forwards. "
            "No weights edited. action_fixture_repair holds isolated drug "
            "starts from the 100 original training-state stays.\n")
```