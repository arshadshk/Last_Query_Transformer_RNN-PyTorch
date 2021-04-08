# Last-Query-Transformer-RNN  
  
Implementation of the paper [Last Query Transformer RNN for knowledge tracing](https://arxiv.org/abs/2102.05038).  The novel point of the model is that it only uses the last input as query in transformer encoder, instead of all sequence, which makes QK matrix multiplication in transformer Encoder to have O(L) time complexity, instead of O(L^2). It allows the model to input longer sequence.    

## Model architecture  
<img src="https://github.com/arshadshk/Last_Query_Transformer_RNN-PyTorch/blob/main/lqtrnn.JPG">

## Usage 
```python
from last_query_model import *

seq_len = 100
total_ex = 1200
total_cat = 234
total_in = 2


in_ex, in_cat, in_in = random_data(64, seq_len , total_ex, total_cat, total_in)

model = last_query_model(dim_model=128,
            heads_en=1,
            total_ex=total_ex,
            total_cat=total_cat,
            seq_len=seq_len,
            total_in=2
            )

outs,attn_w = model(in_ex, in_cat,in_in)

print('Output lstm shape- ',outs.shape)

```  


## Parameters
- `seq_len` : int.  
Sequence length of inputs.  
- `dim_model`: int.  
Dimension of model ( embeddings, attention, linear layers).  
- `heads_en`: int.  
Number of heads in multi-head attention block in each layer of encoder.
- `total_ex`: int.  
Total number of unique excercise.
- `total_cat`: int.  
Total number of unique concept categories.
- `total_in`: int.  
Total number of unique interactions.  
- `use_lstm`: bool.  
Use LSTM layer after multi-head attention. (default : True)  




This model is 1st place solution in kaggle competetion- [Riiid! Answer Correctness Prediction](https://www.kaggle.com/c/riiid-test-answer-prediction)    

## Note 
I have just implemented this model. The Credits for model architecture and solution to goes to [Keetar](https://www.kaggle.com/keetar). Refer this [link](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/218318) for more information.

## Citations  

```bibtex
@article{jeon2021last,
  title={Last Query Transformer RNN for knowledge tracing},
  author={Jeon, SeungKee},
  journal={arXiv preprint arXiv:2102.05038},
  year={2021}
}
```

```bibtex
@misc{vaswani2017attention,
    title   = {Attention Is All You Need},
    author  = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year    = {2017},
    eprint  = {1706.03762},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```
