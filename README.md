# 🤖 Custom GPT Language Model & Chatbot
_A Transformer-based LLM built from scratch that powers a command-line chatbot_

## 📌 Project Overview

In this project, I built my own **GPT-style Large Language Model (LLM)** from scratch by implementing the **transformer architecture** introduced in the research paper *"Attention is All You Need."*  
The model was trained using **OpenWebText (60 GB)**, the open-source version of the WebText dataset used in GPT-2.  
After training, I used the model to build a working **chatbot** that operates via the **command prompt**.
The model is trained using:
- 🔧 PyTorch
- 📘 Custom implementation of multi-head attention, feed-forward layers, and transformer blocks
- 💻 Nvidia GPU on a virtual environment called `cuda`
- 🧠 Tokenization and text normalization
- 📊 Training for 200 to 3000 epochs (proof of concept)

### ✅ Key Features
- Custom-built transformer-based GPT model
- Command-line chatbot that generates responses based on user input
- Modular design of transformer components: multi-head attention, feed-forward networks, and embedding layers
- Used **AdamW optimizer** for efficient learning
- Dataset: OpenWebText (GPT-2 training dataset)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/TARUNELANGO/Gpt-from-scratch.git
cd Gpt-from-scratch
```

### 2. Setup Virtual Environment
```bash
python -m venv cuda
cuda\Scripts\activate
```

### 3. Install Required Packages
```bash
pip3 install matplotlib numpy ipykernel jupyter torch --index-url https://download.pytorch.org/whl/cu118
```

## Google Colab for those who don't have a GPU:
https://colab.research.google.com/drive/1_7TNpEEl8xjHlr9JzKbK5AuDKXwAkHqj?usp=sharing

If you don't have an NVIDIA GPU, then the `device` parameter will default to `'cpu'` since `device = 'cuda' if torch.cuda.is_available() else 'cpu'`. If device is defaulting to `'cpu'` that is fine, you will just experience slower runtimes.

---

## OpenWebText Download
- https://skylion007.github.io/OpenWebTextCorpus/
- if this doesn't work, default to the wizard of oz mini dataset for training / validation

---

## 🚀 How to Use
1. Run the training script to begin training the GPT model (you may limit the dataset or the no.of epochs):
```bash
python training.py -batch_size 32
```

2. Once training is complete, run the chatbot in command line:
```bash
python chatbot.py -batch_size 32
```

3. The model will generate responses to your inputs using a custom generate() function built on top of the trained transformer.
However, unlike openAI's GPT-2, the results are not much relevant unless trained over more than a billion parameters for several thousand iterations.

---

## 🧠 Technical Highlights
- Implemented key components of transformer from scratch:
   - MultiHeadAttention
   - FeedForwardNetwork
   - TransformerBlock
   - GPTLanguageModel
- Used PyTorch's AdamW optimizer for better convergence
- Tokenization and position encoding embedded into training pipeline
- Performed testing using a subset of the dataset with 300 epochs
- Design is scalable for larger runs with sufficient compute resources

## Research Papers
Attention is All You Need - https://arxiv.org/pdf/1706.03762.pdf

A Survey of LLMs - https://arxiv.org/pdf/2303.18223.pdf

QLoRA: Efficient Finetuning of Quantized LLMs - https://arxiv.org/pdf/2305.14314.pdf
