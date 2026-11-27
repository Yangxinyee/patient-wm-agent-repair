Passes 2 and 10 hold original factual files, pass 22 is head-0 ×4 and pass 15 is u=0. Next I will list the columns of these files and check whether pass 2 and pass 22 can be merged reading by reading, so I can split the NLL change by code.

```python
fo=pd.read_csv('/work/fwd_2/factual_original.csv'); fe=pd.read_csv('/work/fwd_22/factual_edited.csv')
print(list(fo.columns)); print(fo.shape, fe.shape)
print(fo.head(2).to_string()[:400])
```