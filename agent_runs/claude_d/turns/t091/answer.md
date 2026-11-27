The fwd_4 file is a factual pass of the original model on 100 training stays (5,138 readings). Next I'll print its NLL summary and the residual (observed minus predicted mean) spread by event. That gives the noise scale against which any bias from an imperfectly centred edit is paid in NLL.

```python
print(s4['results'])
f4['res']=f4.observed-f4.predicted_mean
print(f4.groupby('event').agg(n=('res','size'),sd=('res','std'),mean=('res','mean'),nll=('nll','mean')).round(3))
print('stays same as selS:', set(f4.stay_id)==set(json.load(open('/work/selS.json'))))
```