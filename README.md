# HyenaDNA: Training and Inference Example

Welcome! Thank you for checking out **HyenaDNA**. This tutorial walks you through setting up a **HyenaDNA** model for:

- Training on the **GenomicBenchmarks** dataset from scratch or fine-tuning (for sequence-level classification tasks).
- Loading pretrained models from **HuggingFace**.
- Running inference on sequences up to **1 million tokens**.

After completing this tutorial, you can explore more advanced training configurations via the official **HyenaDNA** repository, which integrates **PyTorch Lightning** and **Hydra** for experiment management.

## 🔗 Resources

- **Paper:** [HyenaDNA on arXiv](https://arxiv.org/abs/2306.15794)  
- **Blog:** [HyenaDNA Blog Post](https://hazyresearch.stanford.edu/blog/2023-06-29-hyena-dna)  
- **GitHub:** [HazyResearch/HyenaDNA](https://github.com/HazyResearch/hyena-dna)  
- **Hugging Face:** [Pretrained Models](https://huggingface.co/LongSafari)  

---

## 📌 Installation

To get started, install the required dependencies:

```bash
pip install einops torchvision transformers==4.26.1 genomic-benchmarks OmegaConf
```

---

## 🚀 Quick Start

### 1️⃣ Import Required Libraries

```python
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from typing import Optional
```

### 2️⃣ Load and Preprocess the Data

HyenaDNA is designed to work with the **GenomicBenchmarks** dataset. To load and preprocess the dataset:

```python
from genomic_benchmarks.data_loader import load_dataset

dataset = load_dataset("human_enhancers_cohn", split="train")
```

---

## 🏗️ Model Architecture

HyenaDNA is built on an advanced sequence processing architecture, leveraging **Hyena Operators** for long-range sequence modeling. It efficiently processes up to **1M tokens** using Fourier-based convolution methods.

Key Components:
- **HyenaFilter:** Implements implicit long filters with modulation.
- **HyenaOperator:** Processes input sequences through the Hyena recurrence mechanism.

Example implementation of **HyenaFilter**:

```python
class HyenaFilter(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model))

    def forward(self, x):
        return x * self.pos_embedding
```

---

## 🎯 Training

To train a HyenaDNA model from scratch:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

---

## 🧪 Inference

Once trained, you can run inference on genomic sequences:

```python
def predict_sequence(sequence, model):
    inputs = tokenizer(sequence, return_tensors="pt")
    outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=-1)

sample_sequence = "AGCTGATCGTACGTAGCTAG"
prediction = predict_sequence(sample_sequence, model)
print("Predicted class:", prediction.item())
```

---

## 🛠 Advanced Usage

For advanced training configurations, check out the official **HyenaDNA** repo, which integrates:
- **PyTorch Lightning** for training pipelines.
- **Hydra** for managing experimental configurations.
- **Pretrained models** from Hugging Face.

---

## 📜 Acknowledgments

Much of this work is based on:
- [HazyResearch's HyenaDNA](https://github.com/HazyResearch/hyena-dna)
- [State Spaces (S4)](https://github.com/HazyResearch/state-spaces)
- [Safari Model](https://github.com/HazyResearch/safari)

---

## 📩 Contact

If you have any questions, feel free to open an issue in the [GitHub repository](https://github.com/HazyResearch/hyena-dna).

Happy coding! 🚀

