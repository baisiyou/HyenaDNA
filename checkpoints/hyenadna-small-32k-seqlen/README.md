---
license: bsd-3-clause
---

# HyenaDNA

Welcome! HyenaDNA is a long-range genomic foundation model pretrained on context lengths of up to **1 million tokens** at **single nucleotide resolution**. 

See below for an [overview](#model) of the model and training. Better yet, check out these resources.

**Resources:**  

- [arxiv](https://arxiv.org/abs/2306.15794)  
- [blog](https://hazyresearch.stanford.edu/blog/2023-06-29-hyena-dna)  
- [colab](https://colab.research.google.com/drive/1wyVEQd4R3HYLTUOXEEQmp_I8aNC_aLhL?usp=sharing)
- [github](https://github.com/HazyResearch/hyena-dna)


**Links to all HuggingFace models:**  

- [tiny-1k](https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen/tree/main)
- [tiny-1k-d256](https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-d256/tree/main)
- [small-32k](https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen/tree/main)
- [medium-160k](https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen/tree/main)
- [medium-450k](https://huggingface.co/LongSafari/hyenadna-medium-450k-seqlen/tree/main)
- [large-1m](https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen/tree/main)

See [GPU requirements](#hardware) for each model.

### Sample snippet


This code example lets you select which pretrained model to load from HuggingFace, perform inference and get embeddings.

See the [colab](https://colab.research.google.com/drive/1wyVEQd4R3HYLTUOXEEQmp_I8aNC_aLhL?usp=sharing) for these classes, or the ['huggingface.py'](https://github.com/HazyResearch/hyena-dna/blob/main/huggingface.py) script in the main [github](https://github.com/HazyResearch/hyena-dna).


```python

# instantiate pretrained model
pretrained_model_name = 'hyenadna-medium-450k-seqlen'
max_length = 450_000

model = HyenaDNAPreTrainedModel.from_pretrained(
    './checkpoints',
    pretrained_model_name,
)

# create tokenizer, no training involved :)
tokenizer = CharacterTokenizer(
    characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters
    model_max_length=max_length,
)

# create a sample
sequence = 'ACTG' * int(max_length/4)
tok_seq = tokenizer(sequence)["input_ids"]

# place on device, convert to tensor
tok_seq = torch.LongTensor(tok_seq).unsqueeze(0).to(device)  # unsqueeze for batch dim

# prep model and forward
model.to(device)
model.eval()  # deterministic

with torch.inference_mode():
    embeddings = model(tok_seq)

print(embeddings.shape)  # embeddings here!


```

### How to use pretrained weights

- [colab](https://colab.research.google.com/drive/1wyVEQd4R3HYLTUOXEEQmp_I8aNC_aLhL?usp=sharing)  

The colab is the easiest entry point, you can finetune a small model, and do inference on DNA sequences up to 450k on the free tier (T4 GPU), and up to 1 million on the paid tier (A100). It handles all the HuggingFace integration for you, so it's helpful to see this example first.

- [github](https://github.com/HazyResearch/hyena-dna)

Otherwise, checkout of the main HyenaDNA repo for how to load weights into Pytorch Lightning. We use Pytorch Lightning for pretraining and fine-tuning all of our models. If you want to use our actual pretraining code, you can clone this HuggingFace repo to download the actual weights.ckpt, and then pass it to Pytorch Lightning via command line or config. See the [github](https://github.com/HazyResearch/hyena-dna) README for how to do all that.


If you want a standalone version that's easy to port into your own code (and not tied to our repo or Pytorch Lightning), we have that and a HuggingFace example in ['huggingface.py'](https://github.com/HazyResearch/hyena-dna/blob/main/huggingface.py) too.


### GPU requirements (suggested)
<a name="hardware"></a>

Here are suggestions on the hardware (preferred minimum) we think you can use for each model.

GPU during: Pretrain, fine-tune, inference

- [tiny-1k](https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen/tree/main): (T4, T4, T4)
- [small-32k](https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen/tree/main): (A100-40, T4, T4)
- [medium-160k](https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen/tree/main): (A100-40, A100-40, T4)
- [medium-450k](https://huggingface.co/LongSafari/hyenadna-medium-450k-seqlen/tree/main): (A100-40, A100-40, T4)
- [large-1m](https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen/tree/main): (A100-80, A100-80, A100-40)


T4: 16GB  
A100-40: 40GB  
A100-80: 80GB  


## Model & Training Overview
<a name="model"></a>

HyenaDNA uses a simple stack of [Hyena](https://arxiv.org/abs/2302.10866) operators, which are a subquadratic drop-in replacement for attention in Transformers. The Hyena operator is able to match quality in language modeling by using modified input projections, implicit convolutions and gating, all subquadratic operations.

This enables HyenaDNA to reach context lengths of up to 500x longer than previous genomic Transformer models using dense attention, and train 160x faster at sequence length 1M (compared to Flash Attention).

We use a single character tokenizer with a primary vocab of 4 nucleotides (plus special tokens), enabling the single nucleotide resolution, a first in genomic foundation models. In addition, the implicit long convolution enables a **global receptive field** at each layer.

We pretrain using next token (nucleotide) prediction on the human reference genome (HG38).

HyenaDNA sets new SotA on 23 downstream tasks including predicting regulatory elements, chromatin profiles, and species classification. We also explore what new capabilities open up with long context in genomics, including the first use of in-context learning with soft prompt tuneable tokens and instruction fine-tuning.


Check out our [blog](https://hazyresearch.stanford.edu/blog/2023-06-29-hyena-dna) for more details on HyenaDNA!

### Authors

Eric Nguyen*, Michael Poli*, Marjan Faizi*, Armin Thomas, Callum Birch-Sykes, Michael Wornow, Aman Patel, Clayton Rabideau, Stefano Massaroli, Yoshua Bengio, Stefano Ermon, Stephen Baccus, Chris Re.

**Contact**

Eric Nguyen, etnguyen@stanford.edu  
Michael Poli, poli@stanford.edu  
Marjan Faizi, Marjan_Faizi@hms.harvard.edu  


## Citation


Feel free to cite us :)

```
@article{nguyen2023hyenadna,
      title={HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution}, 
      author={Eric Nguyen and Michael Poli and Marjan Faizi and Armin Thomas and Callum Birch-Sykes and Michael Wornow and Aman Patel and Clayton Rabideau and Stefano Massaroli and Yoshua Bengio and Stefano Ermon and Stephen A. Baccus and Chris Ré},
      year={2023},
      eprint={2306.15794},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

```















