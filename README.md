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

## Overview
### **RAG: Retrieval Augmented Generation**
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
  - In the paper, the authors use a pre-trained neural retriever to access a dense vector index of Wikipedia 

- **Generator:** $p_{\theta}(y_i \mid x, z, y_{1:i-1})$  
  - Generates each next token $y_i$ based on:  
    - The input query $x$  
    - The retrieved document $z$ (concatenated with the query as context)  
    - All previously generated tokens $y_{1:i-1}$  
  - Implemented using **BART-large**, a sequence-to-sequence (seq2seq) transformer that encodes the combined input `[x ; z]` and decodes the output sequence.



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
- The model retrieves a set of relevant documents and uses **that set of documents to generate the entire output sequence**.  

**How it works:**
1. The model retrieves the top-K documents relevant to the query.  
2. Each retrieved document is treated as a latent variable $z$. The model then generates a full output sequence conditioned on that document and calculates the probability of producing that sequence $p(y∣x,z)$ given the input query and document. 
3. After generating K complete sequences, the model marginalizes over the documents by weighting each sequence according to the document’s retrieval probability $p(z | x)$ and summing these weighted probabilities to get the final probability of the complete output sequence $p(y | x)$.

**Mathematically:**

$$
p_{\text{RAG-Sequence}}(y|x) = \sum_{z \in \text{top-k}(p(\cdot|x), Z)} p_\eta(z|x) \prod_{i=1}^N p_\theta(y_i|x, z, y_{1:i-1})
$$

**Key insight:** One document conditions the entire generation; marginalization happens at the sequence level.

**Use case:** Works well when most of the required information is contained in one or a few documents.  
**Limitation:** Less effective when information is distributed across multiple sources.  
**Computation:** Requires K full sequence generations (one per retrieved document).  

---

### 2. RAG-Token
- The model allows **different retrieved documents to influence each token** in the generated output.  
- Instead of treating one document as the latent variable for the whole sequence, each token can draw from a different document.

**How it works:**
1. The model retrieves the top-K relevant documents for the query.  
2. The model generates the output token by token. At each token position $y_i$, the model computes the probability of that token for each document $p(y_i | x, z, y_{1:i-1})$ which is conditioned on the input query, the retrieved document, and the previously generated tokens.
3. Each document’s probability distribution is weighted by its retrieval probability $p(z | x)$ and summed to get the final probability distribution for the next token. The most likely token is selected, and the process repeats for the next token.  

**Mathematically:**

$$
p_{\text{RAG-Token}}(y|x) = \prod_{i=1}^N \sum_{z \in \text{top-k}(p(\cdot|x), Z)} p_\eta(z|x) \; p_\theta(y_i|x, z, y_{1:i-1})
$$

**Key insight:** Each token has its own latent document variable; marginalization happens at the token level.

**Use case:** Ideal when information is scattered across multiple passages and needs to be integrated throughout the answer.  
**Limitation:** Computationally expensive, since it performs K forward passes per token (K × N computations for a sequence of length N).  
**Computation:** Requires K probability computations at each of N token generation steps.

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

---

<details>
<summary>If you were tasked with building a RAG chatbot that is used for troubleshooting technical issues in a cloud infrastructure platform (like AWS or Azure), would you use RAG-Sequence or RAG-Token?</summary>

Because each step of a troubleshooting answer might depend on different documentation sources — such as error codes, CLI commands, or configuration guides — RAG-Token’s token-level retrieval lets the model dynamically pull the most relevant technical details as it generates the response.
</details>

---

## Empirical Results

| **Task** | **Evaluation Metric(s)** | **Key Takeaway** |
|-----------|---------------------------|----------------------------------|
| **Open-domain Question Answering** | Exact Match (EM) | RAG outperforms extractive retrievers like DPR and REALM, and generative seq2seq baselines such as T5 (closed-book). |
| **Abstractive Question Answering (MSMARCO)** | BLEU, ROUGE | RAG surpasses the BART seq2seq baseline, generating more factual and less hallucinatory responses. |
| **Jeopardy Question Generation** | Q-BLEU, Human Evaluation | RAG produces more factual and specific questions than the BART baseline, showing stronger knowledge-grounded generation. |
| **Fact Verification (FEVER)** | Classification Accuracy | RAG reaches performance comparable to state-of-the-art models, without relying on complex pipelines or additional supervision. |


---

## Critical Analysis

### What the Paper Did Well

- **Comprehensive Evaluation:** The paper tested RAG across diverse tasks (open-domain QA, abstractive QA, question generation, fact verification), demonstrating broad applicability rather than optimizing for a single benchmark.  
- **Human Evaluation:** Conducted human assessments showing RAG was more factual than BART in 42.7% of cases vs. 7.1%, with concrete evidence of reduced hallucinations.  
- **Explored Retrieval Settings:** The paper experimented with different numbers of retrieved documents during training and testing, showing that performance is robust across configurations and can improve when more documents are retrieved at test time.

### Key Limitations

- **Retrieval Quality Dependency:** RAG's performance is fundamentally bounded by retrieval quality. Experiments show a 12-point drop when swapping DPR for BM25 (44.0 → 31.8 EM on Natural Questions). No fallback mechanisms are provided for handling retrieval failures.  

- **Limited Exploration of Components:** All experiments use a single knowledge source (Wikipedia, December 2018), which may not always be reliable or complete. The generator also is fixed to BART-large, with no exploration of other seq2seq models or scaling effects.  

- **Conflicting Information Handling:** The paper does not explore how RAG resolves contradictions when retrieved documents contain conflicting information, and there is no mechanism to flag contradictory evidence to users.  

--- 

<details>
<summary> If you had to improve RAG's performance with limited resources, which component would you focus on — retrieval or generation — and why?</summary>

With limited resources, improving the retrieval component would yield the greatest impact.
RAG’s performance depends primarily on the quality and relevance of the retrieved documents — even a strong generator cannot produce accurate or factual answers without reliable context. Enhancing retrieval through better indexing, chunking, or embedding quality directly improves the grounding information the generator relies on.

Improving the generation component would be less efficient. Retraining or fine-tuning a generator is computationally expensive, involves large parameter updates, and typically produces only small gains. In contrast, improving retrieval quality is more cost-effective and leads to larger downstream improvements in factual accuracy.
</details>


---

## Impact

- **Novel Hybrid Architecture:** First end-to-end system combining learned retrieval with seq2seq generation, using both parametric (model weights) and non-parametric (retrieved documents) memory.  
- **Strong Results:** Achieved state-of-the-art performance on open-domain QA benchmarks like Natural Questions (44.5 EM), TriviaQA (56.8 EM), and WebQuestions (45.2 EM), with fewer parameters than models like T5-11B.  
- **Knowledge Updating Without Retraining:** Demonstrated "index hot-swapping" — knowledge can be updated by replacing the document index
- **Reduced Hallucinations & Interpretability:** Grounded generation in retrieved documents leads to more factual, specific outputs and provides visible sources.  
- **Foundation for Modern AI:** Inspired retrieval-augmented systems like ChatGPT with browsing, Perplexity AI, and enterprise RAG solutions; set the standard for hybrid parametric/non-parametric models.

--- 

## Citation

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., … Kiela, D. (2021). Retrieval-augmented generation for knowledge-intensive NLP tasks. Retrieved from https://arxiv.org/abs/2005.11401 


## Other Resources

- Code Demo: https://colab.research.google.com/drive/1J4vv3AuKSaLhY6fdgtIcbYEPmkJivJE-?usp=sharing
- https://github.com/NirDiamant/RAG_Techniques/tree/main?tab=readme-ov-file
- https://www.geeksforgeeks.org/nlp/retrieval-augmented-generation-rag-for-knowledge-intensive-nlp-tasks/#






