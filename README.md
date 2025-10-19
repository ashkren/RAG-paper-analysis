# Retrieval-Augmented Generation (RAG) for Knowledge-Intensive NLP Tasks

---

## Overview

Large pre-trained language models store vast knowledge but have critical limitations:

**Knowledge Management Issues:**
- Cannot easily expand or revise memory as the world changes
- Updating knowledge requires expensive retraining

**Reliability Issues:**
- Produce hallucinations
- Lag behind task-specific architectures on knowledge-intensive tasks
- Difficult to verify sources

**Interpretability Issues:**
- Operate as black boxes
- Cannot explain why a particular response is generated
- Cannot provide sources

**The Solution: RAG**

Retrieval-Augmented Generation (RAG) combines two types of memory:

- **Parametric Memory:** Pre-trained seq2seq transformer (BART-large, 400M parameters)
  - Encodes learned knowledge and reasoning
- **Non-Parametric Memory:** Dense vector index of Wikipedia (21M passages)
  - Accessed via Dense Passage Retrieval (DPR)
  - Provides external factual knowledge

**Key Advantages:**
- Knowledge can be updated by swapping document index
- Improved transparency and factual grounding
- Reduced hallucinations
- No retraining required to update knowledge

---

## How RAG Works

**Step 1: Retrieval**
- Encode input query using query encoder (BERT_q)
- Retrieve top-K relevant documents from Wikipedia using Maximum Inner Product Search (MIPS)
- Rank documents by similarity to query

**Step 2: Generation**
- Generator (BART) takes input + retrieved documents
- Concatenates input with each document
- Generates output conditioned on combined context

**Step 3: Marginalization**
- Treat documents as latent variables
- Compute output probability for each retrieved document
- Weight by retrieval probability \(p(z|x)\)
- Sum across all documents for final output probability

**Training**
- Both retriever and generator trained end-to-end
- No supervision on which documents to retrieve
- Model learns which documents improve generation

**Innovation Over Prior Work**
- Works with **any seq2seq task**, not just extractive QA
- Generates free-form text
- Uses pre-trained components; no expensive task-specific pre-training needed

---

## Architecture

### Core Components

**1. Retriever: DPR**
- Bi-encoder: Query encoder + Document encoder
- Similarity via dot product, top-K retrieval via MIPS
- Pre-trained on TriviaQA & Natural Questions
- Acts as **non-parametric memory**

**2. Generator: BART**
- Pre-trained seq2seq transformer
- Generates output autoregressively
- Acts as **parametric memory** (reasoning & language)

**3. Document Index**
- 21M Wikipedia passages (~100 words each)
- Encoded into 728-dimensional vectors
- Stored using FAISS for sub-linear retrieval

### RAG Variants

**RAG-Sequence (Document-Level Marginalization)**
- Uses same document for entire output sequence
- Best for single-source, factual answers

![equation](https://latex.codecogs.com/svg.image?p(y|x)=\sum_{z\in%20top-K}p_\eta(z|x)\cdot%20p_\theta(y|x,z)%20=%20\sum_{z\in%20top-K}p_\eta(z|x)\prod_{i=1}^{N}p_\theta(y_i|x,z,y_{1:i-1}))


**RAG-Token (Token-Level Marginalization)**
- Can use different documents for each token
- Best for multi-fact synthesis

![equation](https://latex.codecogs.com/svg.image?p(y|x)=\prod_{i=1}^{N}\sum_{z\in%20top-K}p_\eta(z|x)p_\theta(y_i|x,z,y_{1:i-1}))




### **Training Process**

- **Objective:** Minimize the *negative marginal log-likelihood* of generating the correct output given the retrieved documents.  
- **Optimizer:** ADAM — chosen for stable and efficient gradient updates.  
- **Parameter Updates:**  
  - **BERT₍d₎ (Document Encoder):** *Frozen* during training. Its weights are not updated, preserving its pre-trained representations to maintain retrieval consistency.  
  - **BERT₍q₎ (Query Encoder):** *Fine-tuned* so the query representations adapt to the specific downstream task.  
  - **BART (Generator):** *Fine-tuned* to improve sequence generation quality conditioned on the retrieved context.


---

## Questions

**Q1: Why are the parameters in the document encoder frozen?**
- Updating BERT_d requires re-encoding 21M passages each step (prohibitively expensive)
- Pre-trained DPR is strong enough for retrieval
- Empirical results show fine-tuning query encoder suffices
- Trade-off: computational feasibility vs potential performance gains

**Q2: When would we use RAG-Sequence vs RAG-Token?**
- **RAG-Sequence:** Short, factual answers from one source
- **RAG-Token:** Complex answers needing multiple sources
- Token-level marginalization is more computationally expensive

---

## Pseudocode

```text
Algorithm: Retrieval-Augmented Generation (RAG)

Input:
    x  ← input query
    D  ← document index (encoded passages)
    K  ← number of documents to retrieve
Output:
    ŷ ← generated output

---------------------------------------------------
1.  // Retrieval step
2.  q ← BERT_q.encode(x)                     // encode query
3.  z₁,...,z_K ← topK(MIPS(q, D))           // retrieve top-K docs using inner product
4.  for each document z_i:
5.      s_i ← similarity(q, z_i)            // compute similarity score
6.  p(z_i|x) ← softmax(s_i)                 // retrieval probability distribution

---------------------------------------------------
7.  // Generation step
8.  for each document z_i:
9.      context_i ← concat(x, z_i)
10.     ŷ_i ← BART.generate(context_i)       // conditional generation
11.     p(y|x,z_i) ← P_BART(ŷ_i|x,z_i)       // model likelihood

---------------------------------------------------
12. // Marginalization step
13. p(y|x) ← Σ_i [ p(z_i|x) * p(y|x,z_i) ]   // weighted marginal probability

---------------------------------------------------
14. // Training (optional)
15. minimize L = -log p(y|x)                 // marginal log-likelihood loss

```

## Results Snapshot

**Performance**
- State-of-the-art on Natural Questions, TriviaQA, WebQuestions, CuratedTrec
- Strong on Jeopardy, MS-MARCO abstractive QA, FEVER

**Human Evaluation**
- 42.7% more factual than BART baseline
- 37.4% more specific

**Knowledge Update Demo**
- Swapping Wikipedia index from 2016 → 2018 affects accuracy
- Demonstrates that knowledge can be updated without retraining

---

## Critical Analysis

**Strengths**
- Parameter-efficient: 626M trainable params outperform T5-11B
- Interpretability: Inspect retrieved documents
- Updatable knowledge
- Reduced hallucinations
- General-purpose: QA, generation, classification

**Limitations**
- Retrieval collapse in creative tasks
- Fixed document encoder limits task adaptation
- Single knowledge source (Wikipedia)
- Computational cost for large K

**Potential Improvements**
- Hybrid RAG-Sequence + RAG-Token strategies
- Iterative retrieval with partial generation
- Cross-task multi-task learning
- Failure analysis for complex tasks

---

## Impact

**Immediate (2020-2021)**
- Smaller + retrieval can beat much larger parametric models
- Inspired FiD, RETRO, Atlas

**Long-term (2021-Present)**
- Foundation for grounded LLMs in production (ChatGPT browsing, Claude)
- Vector DB ecosystem growth (LangChain, Pinecone, Weaviate)

**Paradigm Shift**
- Separates "how to think" (parametric) from "what to know" (non-parametric)
- Enables maintainable, trustworthy AI systems

---

## Resources

1. [Original Paper (arXiv)](https://arxiv.org/abs/2005.11401)
2. [HuggingFace Implementation](https://github.com/huggingface/transformers/tree/master/examples/rag)
3. [Interactive Demo](https://huggingface.co/rag/)
4. [DPR Explained](https://www.geeksforgeeks.org/nlp/what-is-dense-passage-retrieval-dpr/)
5. [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

## Code Demonstration

- See `rag_demo.ipynb` for an interactive demonstration

---

## Repository

- `README.md`: This overview  
- `rag_demo.ipynb`: Interactive demo  
- `presentation_slides.pdf`: Slide deck  
- `figures/`: Key diagrams from paper

---

## Citation

```bibtex
@article{lewis2020retrieval,
  title={Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and Riedel, Sebastian and Kiela, Douwe},
  journal={arXiv preprint arXiv:2005.11401},
  year={2020}
}

- Outperforms T5-11B on several benchmarks

**Key Results:**
- Natural Questions: 44.5 EM (vs T5-11B: 34.5)
- Human eval: 42.7% more factual than BART
- Knowledge update: 70% accuracy with correct index, 4-12% with wrong index
