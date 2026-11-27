The two files line up row for row (4,751 readings each). Next I will compute the head-0 ×4 NLL change per event. I will also split glucose readings by whether insulin ran in the preceding hour, using the tuning records, after checking their columns.

```python
m=fo.merge(fe,on=['stay_id','hour','event','observed'],suffixes=('_o','_e'))
m['d']=m.nll_e-m.nll_o; m['dp']=m.predicted_mean_e-m.predicted_mean_o
print(m.groupby('event').agg(n=('d','size'),dnll=('d','mean'),dp=('dp','mean'),adp=('dp',lambda x: x.abs().mean())).round(4))
tu=pd.read_csv('/data/tuning.csv'); print(list(tu.columns)[:12], tu.shape)
```