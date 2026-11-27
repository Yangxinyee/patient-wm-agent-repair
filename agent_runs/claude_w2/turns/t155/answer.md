The head is now exact up to the tool's 4-draw z noise. Tool vs expectation and my 4-draw vs expectation agree (predicted-mean RMSE 1.53 vs 1.57, NLL RMSE 0.162 vs 0.161), and my mean NLL matches the tool's (1.426 vs 1.4235). So I have a complete local reproduction: encoder, core and value head.

Next I'll build a local version of the dose_edit rollout, to evaluate candidate edits without spending forward passes. To validate it I need the tool's original-model dose_edit results on sel1 and their exact format: which rows count, and how they are averaged over hours, paths and stays. So I'll inspect the fwd_1/fwd_2 outputs.

```python
for d_ in ['fwd_1','fwd_2','fwd_3','fwd_4']:
    print(d_, os.listdir('/work/'+d_))
de1=pd.read_csv('/work/fwd_1/'+[f for f in os.listdir('/work/fwd_1') if f.endswith('.csv')][0])
print(de1.shape, de1.columns.tolist()); print(de1.head(8))
print(de1.groupby('edit').size() if 'edit' in de1.columns else '')
for f in os.listdir('/work/fwd_1'):
    if f.endswith('.json') or f.endswith('.txt'): print(f, open('/work/fwd_1/'+f).read()[:1500])
```