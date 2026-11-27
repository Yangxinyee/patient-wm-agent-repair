```python
import json, hashlib, re
from pathlib import Path
import numpy as np
import pandas as pd

def final_audit_300():
    work = Path("/work")
    notes_path = work / "notes.md"
    notes = notes_path.read_text()

    print("ARCHITECTURE")
    print(Path("/model/ARCHITECTURE.md").read_text())
    print("\nEARLY RECORD-IDENTIFICATION NOTES")
    print(notes[:11000])
    evidence_lines = [
        line for line in notes.splitlines()
        if any(term in line.lower() for term in (
            "instrument", "identifi", "randomiz", "all-output",
            "joint loss", "joint nll", "regulariz", "six-lag"
        ))
    ]
    print("\nSELECTED EVIDENCE / LOSS NOTES")
    print("\n".join(evidence_lines)[:11000])

    def as_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    tensor_name = "core.b_in.weight"
    current = as_numpy(load(tensor_name)).astype(np.float32, copy=False)
    original = as_numpy(load_original(tensor_name)).astype(np.float32, copy=False)
    current_hash = hashlib.sha256(np.ascontiguousarray(current).tobytes()).hexdigest()
    print("\nWEIGHT AUDIT")
    print("Edited tensors:", edited())
    print("Current tensor shape:", current.shape)
    print("SHA256:", current_hash)
    print("Edit Frobenius norm:", float(np.linalg.norm(current - original)))
    print("Changed entries:", int(np.count_nonzero(current != original)))

    archive_matches = {}
    for filename in (
        "validated_current400.npz", "margin_validated400.npz",
        "final_switch_audit.npz"
    ):
        path = work / filename
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as archive:
            matches = []
            for key in archive.files:
                arr = archive[key]
                if arr.shape == current.shape and np.issubdtype(arr.dtype, np.number):
                    matches.append({
                        "key": key,
                        "exact_match": bool(np.array_equal(arr, current)),
                        "max_abs_difference": float(np.max(np.abs(arr - current)))
                    })
            archive_matches[filename] = matches
            print(filename, json.dumps(matches))

    def column(frame, candidates, contains=None):
        for name in candidates:
            if name in frame.columns:
                return name
        if contains:
            for name in frame.columns:
                if contains in name.lower():
                    return name
        return None

    factual_pairs = []
    factual_batches = []
    for folder in ("fwd_27", "fwd_29", "fwd_30", "fwd_31"):
        original_df = pd.read_csv(work / folder / "factual_original.csv")
        edited_df = pd.read_csv(work / folder / "factual_edited.csv")
        sc = column(original_df, ["stay_id", "stay", "id"], "stay")
        nc = column(original_df, ["fine_nll", "nll", "value_nll", "nll_fine"], "nll")
        print("\nFactual schema", folder, original_df.columns.tolist())
        if sc is None or nc is None:
            raise RuntimeError("Cannot identify factual stay/NLL columns.")
        keys = [
            c for c in ("stay_id", "stay", "hour", "time", "event", "event_code", "code")
            if c in original_df.columns and c in edited_df.columns
        ]
        if keys and not original_df[keys].equals(edited_df[keys]):
            original_df = original_df.sort_values(keys, kind="stable").reset_index(drop=True)
            edited_df = edited_df.sort_values(keys, kind="stable").reset_index(drop=True)
        if len(original_df) != len(edited_df) or (
            keys and not original_df[keys].equals(edited_df[keys])
        ):
            raise RuntimeError("Factual rows do not pair exactly.")
        delta = edited_df[nc].to_numpy() - original_df[nc].to_numpy()
        factual_pairs.append(pd.DataFrame({
            "stay": original_df[sc].to_numpy(), "delta": delta
        }))
        factual_batches.append(original_df[sc].drop_duplicates().tolist())
        print(folder, "readings", len(delta), "paired delta", float(delta.mean()))

    factual = pd.concat(factual_pairs, ignore_index=True)
    by_stay = factual.groupby("stay")["delta"].agg(["sum", "count"])
    rng = np.random.default_rng(300)
    indices = rng.integers(0, len(by_stay), size=(6000, len(by_stay)))
    bootstrap = (
        by_stay["sum"].to_numpy()[indices].sum(axis=1)
        / by_stay["count"].to_numpy()[indices].sum(axis=1)
    )
    factual_delta = float(factual["delta"].mean())
    factual_ci = np.quantile(bootstrap, [0.025, 0.5, 0.95, 0.975]).tolist()
    print("\nFINAL CANDIDATE FACTUAL VALIDATION")
    print(json.dumps({
        "readings": len(factual), "stays": len(by_stay),
        "value_nll_delta": factual_delta,
        "stay_bootstrap_quantiles_025_50_95_975": factual_ci
    }, indent=2))

    def read_dose(folder, model):
        frame = pd.read_csv(work / folder / f"dose_edit_{model}.csv")
        ec = column(frame, ["edit", "dose_edit", "intervention", "scenario"])
        vc = column(frame, ["shift", "mean_shift", "predicted_shift", "delta"], "shift")
        sc = column(frame, ["stay_id", "stay", "id"], "stay")
        if ec is None or vc is None or sc is None:
            print(folder, model, frame.columns.tolist(), frame.head(3).to_dict("records"))
            raise RuntimeError("Cannot identify dose-edit columns.")
        if "reached" in frame.columns:
            reached = frame["reached"].astype(str).str.lower().isin(["true", "1", "1.0"])
            frame = frame.loc[reached].copy()
        result = frame[[ec, sc, vc]].rename(columns={ec: "edit", sc: "stay", vc: "shift"})
        result["folder"] = folder
        return result

    summaries = []
    current_reached_sets = []
    for label, folders in (
        ("stronger_insulin_first300", ("fwd_46", "fwd_47", "fwd_48")),
        ("margin_all400", ("fwd_42", "fwd_43", "fwd_44", "fwd_45"))
    ):
        for model in ("original", "edited"):
            frames = [read_dose(folder, model) for folder in folders]
            pooled = pd.concat(frames, ignore_index=True)
            summary = pooled.groupby("edit").agg(
                mean_shift=("shift", "mean"),
                shift_sum=("shift", "sum"),
                n_stay_hours=("shift", "size"),
                n_stays=("stay", "nunique")
            ).reset_index()
            summary.insert(0, "model", model)
            summary.insert(0, "cohort", label)
            summaries.append(summary)
            print("\n", label, model)
            print(summary.to_string(index=False, float_format=lambda v: f"{v:.10f}"))
            if label == "stronger_insulin_first300" and model == "original":
                current_reached_sets = [set(frame["stay"]) for frame in frames]

    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_summary.to_csv(work / "final_comparison_partial300.csv", index=False)

    batch_sets = [set(batch) for batch in factual_batches]
    used_batches = []
    for folder, reached in zip(("fwd_46", "fwd_47", "fwd_48"), current_reached_sets):
        matches = [i for i, batch in enumerate(batch_sets) if reached <= batch]
        print("Cohort match:", folder, matches, "reached stays:", len(reached))
        if len(matches) != 1:
            raise RuntimeError("Cannot uniquely recover the completed 100-stay cohorts.")
        used_batches.append(matches[0])
    missing = sorted(set(range(len(factual_batches))) - set(used_batches))
    if len(missing) != 1 or len(factual_batches[missing[0]]) != 100:
        raise RuntimeError("Expected exactly one untested 100-stay cohort.")
    final_stays = factual_batches[missing[0]]
    final_request = {
        "type": "dose_edit", "model": "both", "split": "tuning", "stays": final_stays
    }

    comparison_path = work / "final_rollout400_record_comparison.csv"
    if comparison_path.exists():
        print("\nRECORD-DERIVED TARGET COMPARISON (ESTIMATES, NOT INTERVENTION TRUTH)")
        print(pd.read_csv(comparison_path).to_string(index=False))

    print("\nNEXT FORWARD REQUEST")
    print(json.dumps(final_request))
    print("\nWeights were not changed. Python budget is now exhausted.")

    entry = (
        "\n\n## Run 300: final unchanged-candidate audit and partial rollout pooling\n"
        f"- Edited tensors: {edited()}; core.b_in.weight SHA256: {current_hash}.\n"
        f"- Independent 400-stay factual validation: paired value-NLL delta "
        f"{factual_delta:.12f} over {len(factual)} readings; bootstrap quantiles "
        f"[2.5%,50%,95%,97.5%]={factual_ci}.\n"
        "- Pooled the stronger-insulin candidate's first 300 tuning stays and the "
        "margin candidate's complete 400-stay rollouts; saved "
        "final_comparison_partial300.csv.\n"
        + combined_summary.to_string(index=False)
        + "\n- Recovered the remaining matched 100-stay cohort for one final dose-edit "
          "validation. No further weight changes are planned.\n"
        "- Norepinephrine remains under-responsive; record-convolution targets are "
        "identification-dependent estimates, not observed counterfactual truth. "
        "Additional intervention factors, start times, cross-sensor readouts and "
        "all-output loss are not independently certified by this audit.\n"
        "- No weights changed, gradients or weight optimizer used. "
        "Python runs: 300/300; forward passes: 47/50.\n"
        "- Pending final request: " + json.dumps(final_request) + "\n"
    )
    notes_path.write_text(notes + entry)
    return {
        "request": final_request,
        "factual_delta": factual_delta,
        "factual_bootstrap": factual_ci,
        "dose_summary": combined_summary,
        "archive_matches": archive_matches,
        "weight_hash": current_hash
    }

FINAL_AUDIT_300 = final_audit_300()
```