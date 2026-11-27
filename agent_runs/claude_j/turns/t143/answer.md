The random part of the dose is only partly present right after an infusion bundle (CV R² 0.38 in the slot-mean of `h`, 0.07 in `out`). By the next reading it has mostly faded (0.15 in q). A readout fitted on factual states alone can't separate dose from glucose.

Next plan: generate counterfactual states (doses scaled, recorded readings kept) with the CPU replica. Then fit the readout on factual and counterfactual states together, so it learns how the state actually moves when doses change. That needs a full-stay replica loop. `run_variant3` has one, so I'll get a compact trace of the globals it calls per source line. I'll also save today's findings to the notebook first.

```python
import warnings, dis; warnings.filterwarnings('ignore')
with open('/work/notes.md','a') as fh: fh.write("""
## Session 5 (runs 112-126)
- No source history; replica fns are black boxes. BUN keys int; maps from hourly_maps(tus) are nested dict[stay][hour].
- fwd factual csv row order != reading_table order: MERGE on (stay_id,hour,event).
- Drug bundles are separate (h+0.083), readings at h. Heads average probs over 4 prior-z draws.
- Linear readout from head query q (100 tuning stays): CV R2 RI .89, exogenous part TI .15, RN .62.
- Right after drug-A bundle, dose-deviation CV R2: out .07, hmean .38, z .18 -> deviation weakly encoded.
- PLAN: replica counterfactual states (doses scaled, readings recorded) -> joint ridge readout (factual+cf) -> head unit edit.
""")
lines={}
for ins_ in dis.get_instructions(run_variant3):
    if ins_.starts_line: cur=ins_.starts_line if isinstance(ins_.starts_line,int) else ins_.positions.lineno
    if ins_.opname in ('LOAD_GLOBAL','LOAD_ATTR','LOAD_METHOD','STORE_FAST','LOAD_CONST') and ins_.argval is not None:
        lines.setdefault(ins_.positions.lineno,[]).append(str(ins_.argval)[:12])
for ln in sorted(lines)[:45]: print(ln, ' '.join(lines[ln])[:110])
```