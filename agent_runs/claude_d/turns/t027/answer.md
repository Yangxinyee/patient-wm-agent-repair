MAP measurement is a stay-level feature. 52% of stays never have an arterial line. The rest have arterial MAP in about 95% of hours whatever the last value, plus cuff readings about 30% of the time. The norepinephrine readouts are on map_arterial, so they come only from arterial-line stays. The off-threshold starts at MAP 66–75 follow a low previous-hour MAP (mean 66 vs 74). This looks like restarts after a stop at a tiny rate, so I will build the norepinephrine instrument from hours where an infusion was already running. There the multiplicative titration noise is the experiment.

Next I will:
- confirm arterial measurement within arterial-line stays is unrelated to norepinephrine;
- fit E[nor_s | recorded history] flexibly with gradient boosting on hours where norepinephrine ran at s−1, including stops and the cap;
- run the same sequential g-estimation for the norepinephrine → map_arterial impulse response, lags 1–12.

A response that builds up and persists would make the x2 readout large; a quickly fading one keeps it small.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
MA = arr('ma'); MC = arr('mc')
art_stay = (~np.isnan(MA)).mean(1) > 0.5
print('arterial stays', art_stay.sum())
# measurement within arterial stays vs NE at t-1
NP = np.zeros_like(NOR); NP[:,1:] = NOR[:,:-1]
mm = ~np.isnan(MA[art_stay][:,1:]); npv = NP[art_stay][:,1:]>0
print('P(ma meas) art stays: NE prev on', mm[npv].mean(), 'off', mm[~npv].mean())
# Build NE policy dataset: s with NOR[s-1]>0
rows = []
NP2 = np.zeros_like(NOR); NP2[:,2:] = NOR[:,:-2]
MAp = np.full_like(MA, np.nan); MAp[:,1:] = MA[:,:-1]; MCp = np.full_like(MC, np.nan); MCp[:,1:] = MC[:,:-1]
Mdec = np.where(~np.isnan(MA), MA, MC)
hh = np.tile(np.arange(-1,25), (len(stays),1))
cont = (NP>0)
Xp = np.stack([NP, np.nan_to_num(Mdec,nan=-1), np.isnan(Mdec).astype(float), np.nan_to_num(MA,nan=-1), np.nan_to_num(MC,nan=-1), np.nan_to_num(MAp,nan=-1), np.nan_to_num(MCp,nan=-1), NP2, hh], -1)
Xc_ = Xp[cont]; yc_ = NOR[cont]
gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=100)
from sklearn.model_selection import cross_val_predict
pred = cross_val_predict(gb, Xc_, yc_, cv=3)
EPSN = np.zeros_like(NOR); EPSN[cont] = yc_ - pred
print('NE cont n', cont.sum(), 'R2', 1-np.var(yc_-pred)/np.var(yc_), 'eps sd', np.std(yc_-pred))
```