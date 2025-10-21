# Retrieval-Augmented Generation for Knowledge-Intensive Tasks

## Background
- Pre-trained large language models (LLMs) store large amounts of factual data in their parameters.  
- They perform well on downstream NLP tasks but have limitations:
  - Hard to update or revise memory  
  - Cannot cite sources or verify provenance  
  - Prone to hallucinations  
- **Solution:** Combine parametric memory (LLM) with non-parametric memory (retrieved documents).

### Previous Work
- REALM and ORQA: Encoder-based models for open-domain extractive QA.  

## RAG Overview
### **Why is RAG unique?**
- **RAG extends this hybrid parametric + non-parametric memory setup to sequence2seq encoder-decoder models.**
- Authors used BART-large as the parametric memory, and a dense vector index of Wikipedia, accessed with a pre-trained neural retriever, as the non-parametric memory.

### Main Components

- **Retriever:** $p_{\eta}(z \mid x)$  
  - Returns top-K text passages $z$ given the input query $x$  
  - Based on Dense Passage Retrieval (DPR)
  - Uses a bi-encoder setup:  
    - $q(x)$: Query representation encoded by a BERT-based query encoder  
    - $d(z)$: Document representation encoded by a BERT-based document encoder
      - Builds the document index, which stores dense vector embeddings of all passages
  - Documents are ranked using Maximum Inner Product Search (MIPS) to find the most relevant passages.  

- **Generator:** $p_{\theta}(y_i \mid x, z, y_{1:i-1})$  
  - Generates each next token $y_i$ based on:  
    - The input query $x$  
    - The retrieved document $z$  
    - All previously generated tokens $y_{1:i-1}$  
  - Implemented using BART-large, a sequence-to-sequence (seq2seq) transformer.  


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

- Allows a different document to be used for each token.
- For each token $y_i$, the model marginalizes across the retrieved documents to compute a single probability distribution for the next token:

$$
p(y_i \mid x, y_{1:i-1}) = \sum_{z \in \text{top-K}} p_{\eta}(z \mid x) \, p_{\theta}(y_i \mid x, z, y_{1:i-1})
$$

---
## RAG Algorithm Pseudocode

### RAG-Sequence
```
INPUT: query x, document index D, top K documents
OUTPUT: generated sequence y

RETRIEVAL PHASE:
1. Encode query: q = BERT_query(x)
2. For each document d in D:
     Compute score = dot_product(q, BERT_doc(d))
3. Select top K documents with highest scores
4. Compute retrieval probabilities: p(z|x) = softmax(scores)

GENERATION PHASE:
5. For each of K documents z_i:
     a. Concatenate: context = [x, z_i]
     b. Pass to BART: generate full sequence y_i
     c. Compute sequence probability: P(y_i | x, z_i) using BART
     d. Weight by retrieval: prob_i = p(z_i|x) × P(y_i | x, z_i)

MARGINALIZATION:
6. Sum probabilities across all K documents for each unique sequence:
     P(y|x) = Σ prob_i
7. Return sequence with highest probability
```

---

### **RAG-Token**
```
INPUT: query x, document index D, top K documents
OUTPUT: generated sequence y

RETRIEVAL PHASE:
1. Encode query: q = BERT_query(x)
2. For each document d in D:
     Compute score = dot_product(q, BERT_doc(d))
3. Select top K documents with highest scores
4. Compute retrieval probabilities: p(z|x) = softmax(scores)

GENERATION PHASE (token-by-token):
5. Initialize: y = []
6. For each position i until END token:
     
     a. For each of K documents z_j:
          - Concatenate: context = [x, z_j]
          - Pass to BART: get next token distribution P(token | x, z_j, y_1:i-1)
          - Weight by retrieval: dist_j = p(z_j|x) × P(token | x, z_j, y_1:i-1)
     
     b. MARGINALIZATION - Sum weighted distributions element-wise:
          token_dist = Σ dist_j
     
     c. Select highest probability token from token_dist
     
     d. Append token to sequence: y.append(token)

7. Return y
```
---

## Questions

1) When would you choose RAG-Sequence over RAG-Token, or vice versa? Think about the nature of the task and where the information might come from.
2) 

---

## Critical Analysis

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
- **Direct precursor to** ChatGPT with browsing, Bing Chat, and Perplexity AI  
- **Template for enterprise RAG pipelines** (legal, medical, internal knowledge systems)

### Addressed Key LLM Challenges

| Problem | RAG Solution |
|:---------|:--------------|
| Hallucinations | Reduces hallucinations by grounding responses in retrieved evidence |
| Stale knowledge | Enables updates via index swaps — no retraining required |
| No provenance | Provides transparency through visible retrieved documents |


--- 
## Other Resources

- https://github.com/NirDiamant/RAG_Techniques/tree/main?tab=readme-ov-file




