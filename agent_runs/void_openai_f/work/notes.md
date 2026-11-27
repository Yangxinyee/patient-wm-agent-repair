# Model repair notebook
## Initial plan
Infer drug effects only from simulated records. Inspect architecture first. Check treatment selection, pre-dose histories, dose variation, and overlap. Use analytic weight edits without gradients, and validate factual NLL and the exact paired dose-edit rollout.
Python runs: 1. Forward passes: 0.

## Inspection
Architecture inspected. Actions use a separate encoder, but also enter the latent posterior and recurrent state. Value predictions combine a state head with same-code last-value registers and transition logits. Dose-edit rollouts feed sampled readings back. No drug-effect conclusion yet.
Python runs: 2. Forward passes: 0.

## Forward service issue
First original dose_edit/tuning request failed because backend cuts_tuning.parquet was missing. It was not counted. No model rollout evidence yet. Built hourly panels: readings at integer hours, doses at hour + 0.0833; absent infusion means rate zero.
Python runs: 3. Forward passes: 0.

## Record alignment and confounding
Same-hour finger/lab difference is approximately zero (SD 9.93); arterial/cuff difference is -3.01 (SD 6.72). Raw dose–next-reading associations strongly reflect selection. Original factual/tuning also failed on missing cuts_tuning.parquet, not counted. Computed pooled and within-stay distributed-lag regressions, including history and individual-trend sensitivity checks; results retained in estimates.
Python runs: 4. Forward passes: 0.

## Restart recovery
No edits submitted. Recovering architecture, retained estimates, and forward-error artifacts before further inference. Python runs: 6; successful forward passes: 0.

## Regression sensitivity
Insulin and norepinephrine lag estimates change strongly with stay-trend controls; these regressions alone do not identify causal effects. Run 7 inspection stopped on index-name typo (h, not hour); run 8 completes inspection. No edits. Python runs: 8; successful forward passes: 0.

## Cross-fitted causal analysis
Built features using only measurements and drug histories available before each action, including multi-sensor levels, ages, and trailing trends. Two folds split by stay; nuisance models fitted separately from world-model weights. Results retained in crossfit. Forward backend missing cuts files for both splits; no factual/rollout validation available yet. Python runs: 9; successful forward passes: 0.

## Horizon and nuisance checks
State forward also failed on missing cuts_tuning.parquet. One-hour cross-fit estimates: insulin 0.00139 (SE 0.03899), norepinephrine 0.19499 (SE 0.08746) per rate unit. Run 10 tests horizons 1-8, combined sensors, and richer nuisance fits; full results retained in effect_table. No model edits. Python runs: 10; successful forward passes: 0.
          drug         analysis  horizon hours     coef  cluster_se      N
       insulin          primary        1   all  0.00958     0.03883 354871
       insulin          primary        1  h12+  0.02784     0.04841 218088
       insulin          primary        2   all -0.01656     0.04321 338109
       insulin          primary        2  h12+ -0.02259     0.05425 200258
       insulin          primary        3   all  0.08335     0.04771 321329
       insulin          primary        3  h12+  0.14627     0.06008 182342
       insulin          primary        4   all  0.06044     0.05227 304391
       insulin          primary        4  h12+  0.12973     0.06831 164445
       insulin          primary        6   all -0.03613     0.06313 269934
       insulin          primary        6  h12+  0.10509     0.08676 128365
       insulin          primary        8   all -0.03965     0.07496 235091
       insulin          primary        8  h12+  0.16411     0.11188  91981
       insulin combined sensors        1   all  0.00994     0.03704 403386
       insulin richer nuisances        1   all  0.00524     0.03878 354871
       insulin richer nuisances        1  h12+  0.01759     0.04827 218088
norepinephrine          primary        1   all  0.17925     0.08704 277758
norepinephrine          primary        1  h12+  0.19658     0.10810 166758
norepinephrine          primary        2   all  0.00448     0.10003 263868
norepinephrine          primary        2  h12+  0.03747     0.12504 152856
norepinephrine          primary        3   all -0.11908     0.10938 249987
norepinephrine          primary        3  h12+ -0.05307     0.13764 138985
norepinephrine          primary        4   all -0.00912     0.11942 236147
norepinephrine          primary        4  h12+ -0.00922     0.15398 125056
norepinephrine          primary        6   all -0.11252     0.14421 208379
norepinephrine          primary        6  h12+ -0.15520     0.19448  97305
norepinephrine          primary        8   all -0.11896     0.17155 180628
norepinephrine          primary        8  h12+ -0.03122     0.25008  69500
norepinephrine combined sensors        1   all  0.23381     0.09313 559057
norepinephrine richer nuisances        1   all  0.16736     0.08632 277758
norepinephrine richer nuisances        1  h12+  0.18155     0.10721 166758
