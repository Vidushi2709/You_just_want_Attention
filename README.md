# 🎯 You Just Want Attention

**Implementing Attention Is All You Need — From Scratch**
🚀 German → English Neural Machine Translation with PyTorch

---

## 📜 Overview

Welcome to **You Just Want Attention** — a minimalist yet powerful deep dive into implementing the core of modern NLP: **attention**.

In this project, we build a **Transformer** from scratch (yes, no cheating with Hugging Face's pretrained magic) to translate **German to English**, using the **WMT14 dataset**.

> “Attention is all you need.” — Vaswani et al., 2017
> “Also, please tokenize your inputs.” — Your GPU

---

## 🧠 What You'll Learn

* How **scaled dot-product attention** works
* Why **multi-head attention** matters
* What are **positional encodings**, and why we need them
* How to build a Transformer encoder-decoder from scratch in PyTorch
* End-to-end training for German → English translation
* How to evaluate your model with BLEU scores

---

## 📦 Dataset

We use the **WMT14 English-German** translation dataset:

```python
from datasets import load_dataset

dataset = load_dataset("wmt14", "de-en", split="train")
```

Each example has:

```python
{
  "translation": {
    "de": "Das ist ein Beispiel.",
    "en": "This is an example."
  }
}
```

---

## 🧱 Architecture

Our implementation closely follows the original Transformer paper:

* `Input Embeddings` + `Positional Encoding`
* `Multi-Head Self-Attention`
* `Encoder-Decoder Attention`
* `Feed Forward Networks`
* `Layer Normalization + Residual Connections`

Everything's written from scratch using **PyTorch** — no hidden layers.

---

## 🚀 Training

```bash
python train.py
```

Training is done with:

* Adam Optimizer with Warmup LR Scheduler
* Label smoothing loss
* BLEU score monitoring

---

## 🛠 Requirements

* Python 3.8+
* PyTorch
* `datasets` from Hugging Face
* `tqdm`, `numpy`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📈 Results

Once trained, the model achieves decent translation quality, and BLEU improves over time as attention learns what to focus on — just like in life.

---

## 🧪 Example Translation

```text
German:    Ich liebe maschinelles Lernen.
Predicted: I love machine learning.
```

---

## 🤖 Inspiration

This project was inspired by:

* [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
* Curiosity to see *how attention really works* under the hood

---

## 💬 Fun Fact

> “You just want attention” — Charlie Puth
> He was clearly talking to his encoder.


---

## 📬 Contact

Feel free to open issues or reach out if you want to collaborate!

---

## 🌟 Star This Repo

If you're learning NLP from scratch, this one's for you. Give it a ⭐ if it helped you understand attention better!
