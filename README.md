# Retrieval-Augmented Generation for Knowledge-Intensive Tasks

## Background
- Large pre-trained language models (LLMs) encode vast amounts of factual knowledge within their parameters.  
- **The challenge:** While they excel at many NLP tasks, LLMs have inherent limitations:  
  - Knowledge is static and difficult to update or correct  
  - Cannot reliably cite sources or provide provenance  
  - Susceptible to generating hallucinations  
- **The solution:** Augment LLMs with non-parametric memory by retrieving relevant documents, combining the model’s parametric knowledge with up-to-date external information.


### Previous Work
- REALM and ORQA: Encoder-based models for open-domain extractive QA.  

## RAG Overview
### **Why is RAG unique?**
- **RAG extends this hybrid parametric + non-parametric memory setup to sequence2seq encoder-decoder models.**
- RAG models take an input sequence, use it to retrieve text documents, and then uses the input and retrieved text documents as context to generate an output sequence.


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
- Uses **the same retrieved document** to generate the entire output sequence. Once a document is selected, the model conditions the full output on that document.  
- Marginalizes over the top-K retrieved documents to account for retrieval uncertainty, so the overall probability of generating a sequence is:  

$$
p(y \mid x) = \sum_{z \in \text{top-K}} p_{\eta}(z \mid x) \, p_{\theta}(y \mid x, z)
$$

- Suitable when a **single document contains most of the required information**, but may be less flexible if multiple sources are needed.

---

### 2. RAG-Token
- Allows **different documents to influence each token** in the output. For each token, the model marginalizes across the top-K documents to compute the probability of the next token.  
- Captures fine-grained information from multiple sources, making it more robust when no single document fully answers the query.  
- The probability for each token is:

$$
p(y_i \mid x, y_{1:i-1}) = \sum_{z \in \text{top-K}} p_{\eta}(z \mid x) \, p_{\theta}(y_i \mid x, z, y_{1:i-1})
$$

- Ideal for tasks where **information is scattered across multiple passages**, but computationally more expensive than RAG-Sequence.

---

## RAG Algorithm Pseudocode

### Algorithm 1: Retriever
**Dense Passage Retriever (DPR) component**

**Input:** $x \in V^{\*}$, query sequence  
**Output:** $\tilde{z} \in (V^{\*})^K$, top-K retrieved documents  
**Output:** $p_\eta \in \mathbb{R}^K$, retrieval probabilities  
**Parameters:** $\eta$ consisting of $\text{BERT}_q$ parameters for query encoder and $\text{BERT}_d$ parameters for document encoder (frozen)  
**Hyperparameters:** $K \in \mathbb{N}$, number of documents to retrieve  
**External:** $D$, document index with precomputed embeddings

1. $q \leftarrow \text{BERT}_q(x)$
2. **for** $i \in [|D|]$: $s[i] \leftarrow d[i]^T q$
3. $\tilde{z}, \text{indices} \leftarrow \text{top-K}(s, D)$
4. $p_\eta \leftarrow \text{softmax}(s[\text{indices}])$
5. **return** $\tilde{z}, p_\eta$

**Where:**
- $q$: query embedding
- $d[i]$: precomputed document embedding for document $i$
- $s[i]$: similarity score between query and document $i$

---

### Algorithm 2: Generator
**BART-based sequence generator**

**Input:** $x \in V^{\*}$, input sequence  
**Input:** $z \in V^{\*}$, retrieved document  
**Input:** $y_{1:i-1} \in V^{\*}$, previous tokens  
**Output:** $p_\theta \in \Delta(V)$, probability distribution over next token  
**Parameters:** $\theta$, BART generator parameters

1. $c \leftarrow \text{concatenate}(x, z)$
2. $p_\theta \leftarrow \text{BART}(c, y_{1:i-1} \mid \theta)$
3. **return** $p_\theta$

**Where:**
- $c$: concatenated context (input sequence and retrieved document)

---

### Algorithm 3: RAG-Sequence
**RAG-Sequence: same document for entire sequence**

**Input:** $x \in V^{\*}$, input sequence  
**Output:** $p \in \Delta(V^{\*})$, probability over output sequences

1. $\tilde{z}, p_\eta \leftarrow \text{Retriever}(x \mid \eta, D)$
2. **for** $k \in [K]$:
   1. $p_k \leftarrow 1$
   2. **for** $i \in [N]$:
      1. $p_{\theta,i} \leftarrow \text{Generator}(x, \tilde{z}[k], y_{1:i-1} \mid \theta)$
      2. $p_k \leftarrow p_k \cdot p_{\theta,i}[y_i]$
3. **return** $p = \sum_{k=1}^K p_\eta[k] \cdot p_k$

**Where:**
- $N$: length of output sequence $y$
- $p_k$: probability of generating sequence $y$ using document $k$
- $p_{\theta,i}$: probability distribution over tokens at position $i$

**Mathematical formulation:**

$$
p_{\text{RAG-Sequence}}(y|x) = \sum_{z \in \text{top-k}(p(\cdot|x), Z)} p_\eta(z|x) \prod_{i=1}^N p_\theta(y_i|x, z, y_{1:i-1})
$$

---

### Algorithm 4: RAG-Token
**RAG-Token: different document per token**

**Input:** $x \in V^{\*}$, input sequence  
**Output:** $p \in \Delta(V^{\*})$, probability over output sequences

1. $\tilde{z}, p_\eta \leftarrow \text{Retriever}(x \mid \eta, D)$
2. $p \leftarrow 1$
3. **for** $i \in [N]$:
   1. $p_i \leftarrow 0$
   2. **for** $k \in [K]$:
      1. $p_{\theta,i,k} \leftarrow \text{Generator}(x, \tilde{z}[k], y_{1:i-1} \mid \theta)$
      2. $p_i \leftarrow p_i + p_\eta[k] \cdot p_{\theta,i,k}[y_i]$
   3. $p \leftarrow p \cdot p_i$
4. **return** $p$

**Where:**
- $N$: length of output sequence $y$
- $p_i$: marginalized probability of token $y_i$ over all documents
- $p_{\theta,i,k}$: probability distribution over tokens at position $i$ using document $k$

**Mathematical formulation:**

$$
p_{\text{RAG-Token}}(y|x) = \prod_{i=1}^N \sum_{z \in \text{top-k}(p(\cdot|x), Z)} p_\eta(z|x) \; p_\theta(y_i|x, z, y_{1:i-1})
$$

---

## Questions

1) RAG combines retrieval and generation, but the generator’s output quality depends on the relevance of retrieved documents. How might errors in the retrieval step affect the final output, and what strategies could be used to mitigate these errors?
2) RAG-Sequence uses a single document for the entire output, while RAG-Token can use a different document for each token. What are the trade-offs between these approaches in terms of accuracy, flexibility, and computational cost?

---

## Critical Analysis

**Retrieval Quality Dependency**
- RAG's performance is fundamentally bounded by retrieval quality
- No strategy for handling scenarios where good documents aren't available or embeddings are poor
- Lacks fallback mechanisms for detecting and handling retrieval failures
- System becomes fragile in specialized domains with incomplete knowledge bases or poorly calibrated embedding spaces

**Conflicting Information Handling**
- Paper doesn't explore how RAG resolves contradictions when retrieved documents conflict
- Unclear which sources the model prioritizes (highest-scoring? attempts synthesis?)
- No mechanism to provide balanced views or flag contradictory evidence to users
- Risk of producing misleading answers based on arbitrary document rankings

---

## Impact

**Novel Architecture**
- **First model to combine learned retrieval with a seq2seq generator** in a single end-to-end system  
- Introduced a **hybrid design** that uses both parametric memory (model weights) and non-parametric memory (retrieved documents)  
- Extended retrieval methods beyond extractive QA to include **generation, reasoning, and classification tasks**

**Strong Empirical Results**
- Set **new state-of-the-art results** on multiple open-domain QA benchmarks  
- Achieved competitive performance with significantly fewer parameters than purely parametric models
- **Human evaluation showed RAG responses were more factual, specific, and diverse** than strong baselines
- Demonstrated **"index hot-swapping,"** allowing the model's knowledge to be updated without retraining

**Foundation for Modern Systems**
- Served as a **direct precursor** to retrieval-based assistants like ChatGPT with browsing, Perplexity AI, and enterprise RAG systems  
- Provided a **template for retrieval-augmented LLMs** now widely used across research and industry

**Addressed Key Model Limitations**
- **Reduced hallucinations** by grounding responses in real documents  
- **Kept knowledge current** through simple index updates rather than retraining
- **Improved interpretability** by making retrieved sources visible

--- 
## Other Resources

- https://github.com/NirDiamant/RAG_Techniques/tree/main?tab=readme-ov-file






