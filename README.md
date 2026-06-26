# Transformer-Based Conversational Chatbot

> A transformer-based chatbot trained on conversational and coding datasets, combining intent recognition with adaptive response generation.

---

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
  - [Datasets](#datasets)
  - [Preprocessing](#preprocessing)
  - [Model Architecture](#model-architecture)
  - [Chatbot Logic](#chatbot-logic)
  - [Optimization & Interface](#optimization--interface)
- [Results](#results)
- [Future Plans](#future-plans)
- [Dataset Licenses & Credits](#dataset-licenses--credits)

---

## Overview

This project presents the design and implementation of a transformer-based chatbot trained on conversational and coding datasets. The system combines **intent recognition** with **response generation** to produce contextually adaptive dialogue.

**Primary goals:**

- Gain hands-on experience building and training transformer models from scratch
- Explore data preprocessing pipelines and intent classification techniques
- Demonstrate the integration of NLP and generative AI into an interactive chatbot

**Long-term vision:** Expand the system to accept professor lecture recordings as input, enabling "virtual office hours" that imitate an instructor's conversational style — targeted initially at coding-based courses.

---

## Methodology

### Datasets

Three pre-existing datasets were selected to cover both conversational and programming domains:

| Dataset | Samples | Approx. Training Time (8 epochs, RTX 3080 Ti) | Purpose | License |
|:---|:---:|:---:|:---|:---|
| [Cornell Movie-Dialog Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) | ~80K | ~13 hrs | Movie-style conversations | Unspecified (research use) |
| [DailyDialog](http://yanran.li/dailydialog) | ~160K | ~26 hrs | Everyday text-based dialogue | CC BY-NC-SA 4.0 |
| [Project CodeNet](https://github.com/IBM/Project_CodeNet) | ~5.5M | N/A (code only) | Programming syntax & tasks | CDLA-Permissive 2.0 |

> **Note on CodeNet:** Training was scoped to **C and C++** only — not all languages present in the dataset.

---

### Preprocessing

Raw datasets were cleaned and prepared through the following steps:

1. **Tokenization & cleaning** — normalization of raw text
2. **Stopword removal** — filtering low-signal tokens
3. **Vectorization** — using `TfidfVectorizer` and `CountVectorizer` for transformer-compatible representations

---

### Model Architecture

A full transformer architecture was implemented from scratch, including:

- Scaled dot-product attention and multi-head attention
- Positional encoding
- Feedforward layers with layer normalization

**Training setup:**

- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Parameters are saved and reloaded for seamless reuse

---

### Chatbot Logic

User inputs are processed through the trained transformer to predict intent. Responses are generated via a layered fallback system:

```
User Input
    │
    ▼
Intent Classification (Transformer)
    │
    ├─ Recognized intent ──► Pre-written response map (randomized for variety)
    │
    └─ Unrecognized intent ──► DialoGPT fallback
    │
    ▼
Optional follow-up prompts + conversational memory
```

---

### Optimization & Interface

- Model saving and loading via **PyTorch** state dicts
- **Conversational memory** for improved multi-turn coherence
- Configurable dataset mixing with **adjustable training weights**
- Early-stage **UI groundwork** laid for future web-based deployment

---

## Results

| Metric | Result |
|:---|:---:|
| Intent Recognition Accuracy | ~80% |
| Response Generation | Adaptive, context-aware |

The model successfully classifies user intent and generates relevant responses. Some outputs were incoherent, indicating that further training and algorithmic refinement are needed before production use.

---

## Future Plans

- [ ] Enhance long-term conversational memory for better context tracking
- [ ] Integrate larger and more diverse conversational datasets
- [ ] Experiment with lower learning rates and adaptive optimizers to reduce overfitting
- [ ] Build a web-based UI for general-purpose deployment
- [ ] Train on Project CodeNet C/C++ data to support code understanding, error diagnosis, and code generation

---

## Dataset Licenses & Credits

### Cornell Movie-Dialog Corpus
- **License:** Not officially licensed — typically restricted to research and educational use. Confirm permissions before any commercial application.
- **Source:** http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
- **Citation:** Danescu-Niculescu-Mizil, C., & Lee, L. (2011). *Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs.*

---

### DailyDialog
- **License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Terms:** May be remixed or transformed, but derivatives must use the same license and remain **non-commercial**.
- **Source:** http://yanran.li/dailydialog
- **Citation:** Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). *DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset.*

---

### Topical-Chat
- **License:** [CDLA-Sharing 1.0](https://cdla.dev/sharing-1-0/)
- **Terms:** Any new versions or derivatives of this dataset must be shared under the same license (copyleft for data).
- **Source:** https://github.com/alexa/Topical-Chat
- **Citation:** Gopalakrishnan, K., Hedayatnia, B., Chen, Q., et al. *Topical-Chat: Towards Knowledge-Grounded Open-Domain Conversations.*

---

### Project CodeNet
- **License:** [CDLA-Permissive 2.0](https://cdla.dev/permissive-2-0/)
- **Terms:** Free to use, modify, and redistribute (including commercially) provided the **full license text** is included with any redistribution of the dataset.
- **Source:** https://github.com/IBM/Project_CodeNet
- **Citation:** Puri, R., Kung, D., Janssen, G., et al. (2021). *Project CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding Tasks.*
