# Transformer-Based Conversational Chatbot

## Project Goals
This project presents the design and implementation of a transformer-based chatbot trained on conversational datasets. Our system combines both intent recognition and response generation.

The primary goals included:
* Getting hands on experience with transformers.
* Exploring data preprocessing and intent recognition.
* Demonstrating the integration of NLP and generative AI into an interactive chatbot.

Future plans include expanding this project to allow individuals to input professor recordings, for the chatbot to train on and create "virtual office hours" which can imitate professors' way of talking, specifically for coding based classes.

## Methodology

### Data Preparation and Datasets

We selected pre-existing datasets including conversational datasets and coding datasets:

| Dataset | Number of Samples | Avg. Training Length\* | Purpose | License Used |
| :--- | :--- | :--- | :--- | :--- |
| **Cornell Movie-Dialog Corpus** | ~80k | ~13h per 8 epoches | Movie like conversations | Not Officially Licensed |
| **DailyDialog** | ~160k | ~26h per 8 epoches | Daily Text based dialogue | CC BY-NC-SA 4.0 |
| **Project CodeNet** | ~5.5 million | N/A (Code Only) | Programming | CDLA-Permissive 2.0 |

*\*Training statistics given 3080 TI GPU.*

**Note on CodeNet:** Although this dataset was used, we do not plan on training on all of the coding languages present within the dataset. We specifically focused on C and C++ for our model.

**Preprocessing Steps**:
* Tokenization and cleaning.
* Stopword removal.
* Vectorization using TfidfVectorizer and CountVectorizer Transformer-based.

### Model Architecture and Logic

We implemented a transformer architecture with:
* Scaled dot-product attention and multi-headed attention.
* Positional encoding.
* Feedforward layers and layer normalization.

The model was trained using:
* CrossEntropyLoss and Adam optimizer.
* Trained to classify user intents and save/load the trained parameters for seamless integration.

**Chatbot Logic**:
User inputs were processed through the trained transformer to predict intent, with responses generated using:
* Pre-written response mapping with random selection for variety.
* DialoGPT fallback for unrecognized intents.
* Optional follow-up prompts and basic memory for improved conversation flow.

### Optimization and Interface
Enhancements included:
* Model saving and loading mechanisms using PyTorch.
* Conversational memory.
* Initial steps toward a user interface, with potential for web-based deployment.
* Ability to choose which dataset to train on, or a mix of datasets, with adjusted weights.

## Results

The chatbot achieved:
* Accuracy of around ~80% on intent recognition.
* Using the trained model, generates an adaptive response in accordance with use prompt.

We noted that some results were incoherent, highlighting the need for more intensive training and improvements in our algorithms.

## Remarks and Future Plans

Although we are optimistic about the potential of our program, we still believe that much fine tuning is needed.

Future improvements may include:
* Enhancing memory for long term context recognition.
* Larger conversation dataset implementation.
* Experiment with lower learning rates and adaptive optimizers to reduce overfitting.
* Developing a web UI for general use.
* Training our model on the CodeNet C/C++ datasets, allowing for our model to understand coding syntax, errors in code, and how to generate code for specific prompts given by the user.

## Dataset Licenses & Credits

### DailyDialog
- **License:** **CC BY-NC-SA 4.0**
- **Terms:** You may remix, transform, or build upon this dataset, but must distribute derivatives under the same license and only for **non-commercial** use.
- **Source:** http://yanran.li/dailydialog
- **Full license text:** https://creativecommons.org/licenses/by-nc-sa/4.0/
- **Citation:** Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). *DailyDialog*.

---

### Topical-Chat
- **License:** **CDLA-Sharing-1.0**
- **Terms:** You must share any new versions or derivatives of the dataset you create under the same license (copyleft for data).
- **Source:** https://github.com/alexa/Topical-Chat
- **Full license text:** https://cdla.dev/sharing-1-0/
- **Citation:** Gopalakrishnan, K., Hedayatnia, B., Chen, Q., Gottardi, A., Kwatra, S., Venkatesh, A., Gabriel, R., Hakkani-Tür, D. *Topical-Chat: Towards Knowledge-Grounded Open-Domain Conversations*.

---

### Cornell Movie-Dialogs Corpus
- **License / Terms:** Not officially licensed; use is typically restricted to **research/educational purposes**. Confirm permissions for commercial use.
- **Source:** http://www.cs.cornell.edu/~cristian/Cornell\_Movie-Dialogs\_Corpus.html
- **Citation:** Danescu-Niculescu-Mizil, C., & Lee, L. (2011). *Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs*.

---

### Project CodeNet
- **License:** **CDLA-Permissive 2.0**
- **Terms:** You may use, modify, and redistribute the dataset (including commercially) as long as you include the **full license text** with any redistribution of the dataset.
- **Source:** https://github.com/IBM/Project\_CodeNet
- **Full license text:** https://cdla.dev/permissive-2-0/
- **Citation:** Puri, R., Kung, D., Janssen, G., Zhang, W., Domeniconi, G., Zolotov, V.,... Chen, T. (2021). *Project CodeNet: A large-scale Al for code dataset for learning a diversity of coding tasks*.
