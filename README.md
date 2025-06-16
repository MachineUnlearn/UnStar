
# UN-STAR: UNlearning with Self-Taught Anti-sample Reasoning

The key components of machine learning are data samples for training, a model for learning patterns, and a loss function for optimizing accuracy. Analogously, unlearning can potentially be achieved through **anti-data-samples** (or **anti-samples**), an **unlearning method**, and a **reversed loss function**. While prior research has explored unlearning methods and reversed loss functions, the potential of anti-samples remains largely untapped. Although token-based anti-samples have been previously introduced ([Eldan et al., 2023](https://arxiv.org/abs/2310.02238)), the use of **reasoning-driven anti-samples**—constructed with **falsified answers** and **misleading rationales**—remains unexplored.

In this paper, we introduce **UN-STAR**: _**Un**learning with **S**elf-**T**aught **A**nti-Sample **R**easoning_ for large language models (LLMs). Our contributions are threefold:

1. We propose a novel concept of reasoning-based anti-sample-induced unlearning;
2. We generate anti-samples by leveraging **misleading rationales**, which help reverse learned associations and accelerate the unlearning process;
3. We enable **fine-grained targeted unlearning**, allowing for the selective removal of specific associations without impacting related knowledge—something not achievable by previous works.

**Results demonstrate** that anti-samples offer an efficient, targeted unlearning strategy for LLMs, opening new avenues for **privacy-preserving machine learning** and **model modification**.


> 💻 Runs locally on Apple Silicon (M3/M2/M1) using [mlx](https://github.com/ml-explore/mlx), no PyTorch or CUDA needed.

---

## 📦 Setup Instructions

### 1. Download the Mistral model

Download the Mistral model weights compatible with `mlx` and place them in the project directory as follows:

```
unstar/
├── mistral-q/
│   ├── ... (other model files)
├── unlearnHarry.py
```

> Ensure the folder is named exactly **`mistral-q`**.

You can find the quantized Mistral model from:  
👉 https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.2-4bit  
(or use your preferred Mistral variant for `mlx`)

### 2. Install mlx

```bash
pip install mlx
```

If you're on an M-series Mac and haven't installed `mlx` yet, refer to the official setup:  
👉 https://github.com/ml-explore/mlx

### 3. Run Unlearning Script

```bash
python unlearnHarry.py
```

This script triggers the unlearning process targeting the fact **"Harry Potter studied at Hogwarts"** using anti-sample reasoning.

---

## 🧠 What Does UN-STAR Do?

- Generates **anti-samples** via flipped logic and misleading rationales.
- Updates the model to "forget" specific facts while retaining surrounding knowledge.
- Applies a lightweight policy gradient-inspired optimization — all in Apple-friendly `mlx`.

---

## 📝 Citation

Will be updated shortly.

---
