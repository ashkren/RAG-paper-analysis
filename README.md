# Retrieval-Augmented Generation for Knowledge-Intensive Tasks

## Background
- Pre-trained large language models (LLMs) store large amounts of factual data in their parameters.  
- They perform well on downstream NLP tasks but have limitations:
  - Hard to update or revise memory  
  - Cannot cite sources or verify provenance  
  - Prone to hallucinations  
- **Solution:** Combine parametric memory (LLM) with non-parametric memory (retrieved documents).

### Previous Work
- **REALM** and **ORQA:** Encoder-based models for open-domain extractive QA.  

## RAG Overview
### **Why is RAG unique?**
- **RAG extends this hybrid parametric + non-parametric memory setup to sequence2seq encoder-decoder models.**
- Authors used BART-large as the parametric memory, and a dense vector index of Wikipedia, accessed with a pre-trained neural retriever, as the non-parametric memory.

### Main Components

- **Retriever:** $p_{\eta}(z \mid x)$  
  - Returns top-K text passages $z$ given the input query $x$  
  - Based on **Dense Passage Retrieval (DPR)**  
  - Uses a **bi-encoder** setup:  
    - $q(x)$: Query representation encoded by a BERT-based query encoder  
    - $d(z)$: Document representation encoded by a BERT-based document encoder
      - Builds the document index, which stores dense vector embeddings of all passages
  - Documents are ranked using **Maximum Inner Product Search (MIPS)** to find the most relevant passages.  

- **Generator:** $p_{\theta}(y_i \mid x, z, y_{1:i-1})$  
  - Generates each next token $y_i$ based on:  
    - The input query $x$  
    - The retrieved document $z$  
    - All previously generated tokens $y_{1:i-1}$  
  - Implemented using **BART-large**, a sequence-to-sequence (seq2seq) transformer.  


### Variable Definitions

| Symbol | Meaning |
|:-------|:---------|
| $x$ | Input query or question |
| $z$ | Retrieved document or passage |
| $y$ | Target output sequence |
| $y_i$ | Token generated at step *i* |
| $y_{1:i-1}$ | Sequence of all tokens generated before step *i* |
| $\eta$ | Retriever parameters |
| $\theta$ | Generator parameters |


---

## RAG Variants

### 1. RAG-Sequence
- Uses the same retrieved document to generate the entire output sequence.
- Overall probability:

$$
p(y \mid x) = \sum_{z \in \text{top-K}} p_{\eta}(z \mid x) \, p_{\theta}(y \mid x, z)
$$

This means the final prediction accounts for how likely each document was to be retrieved and how well each supports generating the output.

---

### 2. RAG-Token

- Allows a **different document** to be used for each token.
- For each token $y_i$, the model marginalizes across the retrieved documents to compute a single probability distribution for the next token:

$$
p(y_i \mid x, y_{1:i-1}) = \sum_{z \in \text{top-K}} p_{\eta}(z \mid x) \, p_{\theta}(y_i \mid x, z, y_{1:i-1})
$$

---

## Training
- Trained end-to-end on input/output pairs without supervision on which docs to retrieve.  
- Objective: maximize the likelihood of the correct output while marginalizing over top-K docs.  
- Optimization details:
- Updates query encoder and generator with SGD using **ADAM** optimizer  
- Keeps document encoder and index fixed

---

## Decoding

### RAG-Sequence Decoding

Since the full sequence probability does not factor per token, beam search is run separately for each retrieved document.  
Each sequence hypothesis is then weighted by its document retrieval probability $p_{\eta}(z \mid x)$ and combined to produce the final result.

Two decoding strategies are introduced:

- **Thorough Decoding** — recomputes probabilities for missing sequences to get exact results.  
- **Fast Decoding** — approximates missing sequences as zero to speed up inference.

### RAG-Token Decoding

For each token, the model marginalizes across the top-K retrieved documents and then selects the next token either by taking the highest probability or by sampling.  
This process repeats until the output sequence is complete.


---

# Impact

### Novel Architecture
- **First to combine learned end-to-end retrieval with seq2seq models** (BART)
- Prior work (REALM/ORQA) only used masked LMs for extractive QA
- **Single unified architecture** across multiple task types (QA, generation, classification)

### Strong Empirical Results
- Beat T5-11B (11B params) using only 626M trainable parameters
- Human evaluation: RAG more factual than BART (42.7% vs. 7.1%)
- 11.8% accuracy when answer not in retrieved docs (extractive = 0%)
- Index hot-swap: 70% accuracy with matched index, 4% with mismatched

## Broader AI Landscape

### Enabled Modern Systems
- **Direct precursor to:** ChatGPT with browsing, Bing Chat, Perplexity AI
- **Template for production RAG:** Legal, medical, enterprise knowledge systems

### Solved Core LLM Problems

| Problem | RAG Solution |
|:--------|:-------------|
| Hallucinations | Grounding in retrieved docs |
| Stale knowledge | Swap index (no retraining) |
| No provenance | Retrieved docs visible |




