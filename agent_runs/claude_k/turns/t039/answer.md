The states came back: 5,244 bundles from 99 stays. Each row has out (0–191), the 4 h slots (192–959) and the z posterior mean (960–991), plus stay_id, hour, has_insulin, has_norepinephrine and n_atoms.

Next I'll record progress in the notebook and inspect the bundle structure: bundles per stay, hour patterns, atoms per action bundle (END atoms?), and whether action bundles are separate. Then I'll test linear decodability. I'll regress the most recent insulin rate (and the rate two hours back) on `out` and on the full state at the bundle just before each glucose reading, with cross-validation, and check calibration at high doses. An R² above about 0.95 with no saturation at high rates would support reading dose history from the state. Low R² or flattening at high doses would point to another mechanism.

```python
open('/work/notes.md','a').write("""
## Model vs truth (fwd 1, original, 100 tuning stays: 87 both drugs + 13 norepi/art)
- readout rows = stay-hours 13-24 with the readout reading recorded, 'reached' = hour > first dose hour (>=12).
- model: ins x2 -2.15, ins half +1.23, ne x2 +0.34, ne half -0.06
- my truth (impulse h applied to recorded doses): ins x2 -13.8, half +6.9, ne x2 +4.6, half -2.3
- model direct (lag-1) sensitivity: insulin -0.35/unit (x2), -0.62 (half); norepi +0.11/unit. Truth: -2.0 and +3.35.
- data conditional kernel (DML, holding 4h readings): insulin c_l = -2.01,-1.45,-0.47,+0.22,+0.73,+1.09 ; norepi 3.30,0.99,-0.25,-0.99,-1.09,-1.45
  (positive late terms = baseline-correction / partly truncation confounding) -> calibrate empirically instead.
- fwd 2: states (original) on 99 train stays (sel_states.json).
""")
S2 = np.load('/work/fwd_2/states_original.npz')
print({k: S2[k].shape for k in S2.files})
st = pd.DataFrame({'stay':S2['stay_id'],'hour':S2['hour'],'hi':S2['has_insulin'],'hn':S2['has_norepinephrine'],'na':S2['n_atoms']})
print(st.head(40).to_string())
print(st.groupby('stay').size().describe())
```