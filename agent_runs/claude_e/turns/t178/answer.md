The leak is real and does no good. 536 glucose_finger rows had no insulin in either of the previous two hours, yet the correction there averages |0.53| mg/dL and costs 0.0043 per row, about 2.3 of the 16 nats the whole edit costs on these 100 stays. The few rows with insulin two hours earlier show no lag-2 signal in the leaked readout (corr -0.10).

So the insulin gate (insulin in previous bundle) is not sharp enough on unseen stays. Tightening it would free budget for a larger insulin coefficient without reducing the response on rows that matter.

I'll check the gate on the held-out sel1 rows (tuning stays, head inputs WC with 8 draws). First I need each row's previous-bundle insulin flag, so I'll look at the columns of RR and the keys of the fwd_2 states file.

```python
print(RR.columns.tolist())
z2 = np.load('/work/fwd_2/states_original.npz', allow_pickle=True)
print({k: z2[k].shape for k in z2.files})
```