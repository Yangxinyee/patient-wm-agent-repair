The generic atom helper did not reproduce the cached action vectors, so the calibration stopped before any edit. I’ll compare it with the previously validated drug helper and proceed only with a matching construction. Near-zero disagreement would identify the correct convention.

```python
import numpy as np
import torch

print("=== SELECT EXACT ACTION CONSTRUCTION ===")
action_builders124 = {
    "validated_drug": lambda rows: latent_action_encode(
        [latent_start_atom(r) for r in rows]) if rows else torch.zeros(192),
    "validated_atoms_pool": lambda rows: pool109(
        [latent_start_atom(r) for r in rows], "encode_action", "zero"),
    "generic_atoms_pool": lambda rows: pool109(
        [atom84(r) for r in rows], "encode_action", "zero"),
}
action_errors124 = {}
for action_name124, action_builder124 in action_builders124.items():
    action_probe124 = torch.stack([
        action_builder124([r for r in c[1]
                           if r["event"] in ("insulin", "norepinephrine")])
        for c in drug_cases110[:24]]).detach().cpu().numpy()
    action_errors124[action_name124] = float(np.sqrt(
        np.mean((action_probe124-action_U122[:24])**2)))
print(action_errors124)
action_choice124 = min(action_errors124, key=action_errors124.get)
assert action_errors124[action_choice124] < 1e-5
action_builder124 = action_builders124[action_choice124]

action_edit_U123, action_edit_Y123 = {}, {}
for action_drug124 in ("insulin", "norepinephrine"):
    action_col124 = ("insulin", "norepinephrine").index(action_drug124)
    for action_factor124 in (0.0, 0.5, 1.5, 2.0, 3.0):
        action_vectors124 = []
        for action_case124 in drug_cases110:
            action_rows124 = []
            for action_row124 in action_case124[1]:
                if action_row124["event"] not in ("insulin", "norepinephrine"):
                    continue
                action_r124 = dict(action_row124)
                if action_r124["event"] == action_drug124:
                    action_r124["rate"] = float(action_r124["rate"])*action_factor124
                    action_r124["dose"] = float(action_r124["dose"])*action_factor124
                action_rows124.append(action_r124)
            action_vectors124.append(action_builder124(action_rows124))
        action_key124 = (action_drug124, action_factor124)
        action_edit_U123[action_key124] = torch.stack(action_vectors124).detach().cpu().numpy()
        action_edit_Y123[action_key124] = action_Y122.copy()
        action_edit_Y123[action_key124][:,action_col124] *= action_factor124

action_augU123 = np.concatenate([action_U122]+list(action_edit_U123.values()))
action_augY123 = np.concatenate([action_Y122]+list(action_edit_Y123.values()))
action_augtr123 = np.tile(action_tr122, 1+len(action_edit_U123))
action_augX123 = np.column_stack(((action_augU123-action_mu122)/action_sd122,
                                 np.ones(len(action_augU123))))
action_augdec123 = {}
print("\n=== HELD-STAY RATE AND CROSS-DRUG SHIFT CALIBRATION ===")
for action_lam124 in (0.001, 0.01, 0.1, 1.0):
    action_pen124 = np.eye(action_augX123.shape[1])*action_lam124
    action_pen124[-1,-1] = 1e-8
    action_B124 = np.linalg.solve(
        action_augX123[action_augtr123].T@action_augX123[action_augtr123]+action_pen124,
        action_augX123[action_augtr123].T@action_augY123[action_augtr123])
    action_augdec123[action_lam124] = action_B124
    action_basepred124 = action_X122@action_B124
    action_metrics124 = []
    for action_drug124 in ("insulin","norepinephrine"):
        action_col124 = ("insulin","norepinephrine").index(action_drug124)
        action_ds124, action_ts124, action_cross124 = [], [], []
        for action_key124, action_U124 in action_edit_U123.items():
            if action_key124[0] != action_drug124:
                continue
            action_X124 = np.column_stack(((action_U124-action_mu122)/action_sd122,
                                           np.ones(len(action_U124))))
            action_dp124 = action_X124@action_B124-action_basepred124
            action_dt124 = action_edit_Y123[action_key124]-action_Y122
            action_mask124 = action_te122 & (action_Y122[:,action_col124]>0)
            action_ds124.extend(action_dp124[action_mask124,action_col124])
            action_ts124.extend(action_dt124[action_mask124,action_col124])
            action_cross124.extend(action_dp124[action_mask124,1-action_col124])
        action_ds124, action_ts124 = np.asarray(action_ds124), np.asarray(action_ts124)
        action_metrics124.extend([
            action_ds124@action_ts124/(action_ts124@action_ts124),
            np.sqrt(np.mean((action_ds124-action_ts124)**2)),
            np.sqrt(np.mean(np.asarray(action_cross124)**2))])
    print(action_lam124, np.round(action_metrics124,4))

print("\n=== FULL-BUNDLE INPUT AND LATENT-DIRECTION INVENTORY ===")
for action_name124, action_obj124 in list(globals().items()):
    if action_name124.startswith(("audit_", "action_")):
        continue
    if hasattr(action_obj124,"shape"):
        action_shape124 = tuple(action_obj124.shape)
        if action_shape124 in ((4035,192),(4035,32),(32,),(6,),(32,6),(6,32)):
            print(action_name124, action_shape124)
    elif action_name124.endswith("104") and isinstance(action_obj124,(int,float,str)):
        print(action_name124,repr(action_obj124))

with open("/work/notes.md","a") as f:
    f.write("\n## Python 123–124 — resolve action helper convention\n"
            "- Generic atom helper mismatched cached action vectors; stopped before calibration or edits.\n"
            "- Selected an exact action construction and tested augmented dose decoders on held training stays.\n"
            "- No weights changed.\n")
```