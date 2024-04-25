# attention

Implementation of causal self-attention that closely follows [3Blue1Brown's video](https://youtu.be/eMlx5fFNoYc?si=sMv_A4gNVFOWIkso).

## Files

`attention.ipynb` contains the code sections that closely corresponds to what 3B1B is talking about in the video in chronological order.

`attention.py` has the code bundled into a nice class ready for use.

## Usage

### Single head

```python
import attention

d_in, d_out_kq, d_out_v = 120, 12, 120
s = attention.SelfAttention(d_in, d_out_kq, d_out_v)
```

Output shape: `number of tokens, token embedding dimension`

### Multi-head

```python
import attention

d_in, d_out_kq, d_out_v, num_heads = 120, 12, 120, 4
m = attention.MultiHeadAttention(d_in, d_out_kq, d_out_v, num_heads)
```

Output shape: `num_heads, number of tokens, token embedding dimension`
