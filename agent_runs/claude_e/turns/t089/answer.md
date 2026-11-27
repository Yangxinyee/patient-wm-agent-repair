The head's view of the lag-1 dose is limited. A gradient-boosted model reaches R² 0.19 from q and 0.35 from `out` plus slot-mean h. The z posterior, which the head never sees, reaches 0.70.

A value-head edit can therefore carry only a partial lag-1 signal. Its usefulness depends on the trade-off between rollout response and factual NLL cost, and I will measure both.

To place a code-gated readout inside the value MLP, I need two things. First, hidden units that can be repurposed cheaply, which I find by computing the factual NLL increase from removing each unit, offline. Second, a gate direction in the head input that cleanly separates glucose codes from MAP codes. Next I check both.

```python
def head_parts(Xs, R_, nz=8, seed=0):
    Q = q_from_rows(Xs, torch.as_tensor(R_.prev.values), torch.as_tensor(R_.dt.values, dtype=torch.float64), nz=nz, seed=seed)
    f = reg_feats(torch.as_tensor(R_.has.values), torch.as_tensor(R_.age.values), torch.as_tensor(R_.zl.values))
    lg, wc, hid = value_logits(Q, torch.as_tensor(R_.code.values), torch.as_tensor(R_.lvb.values), f)
    return lg, wc, hid
RR = pd.concat([RA.assign(src='A'), R1.assign(src='1')], ignore_index=True)
lgA, wcA, hidA = head_parts(XA, RA); lg1, wc1, hid1 = head_parts(X1, R1)
LG = torch.cat([lgA, lg1], 1); WC = torch.cat([wcA, wc1], 1); HID = torch.cat([hidA, hid1], 1)
ob = torch.as_tensor(RR.obin.values)
def nll_of(lg): 
    p = torch.softmax(lg, -1).mean(0); return -torch.log(p[torch.arange(p.shape[0]), ob])
base_nll = nll_of(LG); print('base NLL', base_nll.mean().item())
imp = []
for j in range(192):
    lgj = LG - HID[..., j:j+1] * W2[:, j][None, None, :]
    imp.append((nll_of(lgj).mean() - base_nll.mean()).item())
imp = np.array(imp); order = np.argsort(imp)
print('least important units:', order[:12].tolist(), np.round(imp[order[:12]], 5).tolist())
print('activation frac>0.1 of those:', [(HID[..., j].abs()>0.1).float().mean().item().__round__(3) for j in order[:12]])
# gate direction feasibility
E = T('embed.code.weight'); cg = (E[6]+E[10])/2 - (E[7]+E[5])/2
codes = RR.code.values
proj = (WC.mean(0) @ cg).numpy()
for c in [6,10,7,5]: print('code', c, 'v.wc range', np.percentile(proj[codes==c], [0, 1, 50, 99, 100]).round(2))
```