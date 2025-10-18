# Retrieval-Augmented Generation (RAG) for Knowledge-Intensive NLP Tasks

---

## 1. Overview

### The Problem

Large pre-trained language models have shown impressive capabilities, storing vast amounts of factual knowledge in their parameters. However, they face critical limitations:

**Knowledge Management Issues:**
- Cannot easily expand or revise their memory as the world changes
- Updating knowledge requires expensive retraining on new data
- Knowledge is "baked in" during training and becomes stale

**Reliability Issues:**
- Produce "hallucinations" - plausible-sounding but factually incorrect information
- On knowledge-intensive tasks (those requiring external knowledge sources), they lag behind task-specific architectures
- Difficult to verify where information comes from

**Interpretability Issues:**
- Operate as black boxes with no insight into their decision-making process
- Cannot explain why they generated a particular response
- No ability to cite sources or provide provenance for claims

### The Solution: RAG

The authors propose Retrieval-Augmented Generation (RAG), a hybrid model that addresses these limitations by combining two types of memory:

**Parametric Memory:**
- A pre-trained seq2seq transformer (BART-large with 400M parameters)
- Encodes learned knowledge and language understanding
- Handles "how to generate" - the model's reasoning and language capabilities

**Non-parametric Memory:**
- A dense vector index of Wikipedia (21M passages, 100 words each)
- Accessed via a pre-trained neural retriever (Dense Passage Retrieval - DPR)
- Provides "what to know" - external factual knowledge

**Key Advantages of This Approach:**
- Knowledge can be directly revised by updating the document index
- Accessed knowledge can be inspected and interpreted (improved transparency)
- More factually grounded generation with reduced hallucinations
- Can update knowledge without retraining the model

### How RAG Works

**Step 1: Retrieval**
- Given an input query x (e.g., a question)
- Query encoder (BERT) converts query to embedding
- Retrieves top-K most relevant documents from Wikipedia using Maximum Inner Product Search
- Documents ranked by similarity to query

**Step 2: Generation**
- Generator (BART) receives original input + retrieved documents
- Concatenates input with each retrieved document
- Generates output conditioned on this combined context

**Step 3: Marginalization**
- Since we don't know which document is truly relevant, treat documents as latent variables
- Compute probability of output for each retrieved document
- Weight each by retrieval probability p(z|x)
- Sum (marginalize) across all documents to get final output probability

**Training:**
- Both retriever and generator trained end-to-end
- No explicit supervision on which documents to retrieve
- Model learns implicitly which documents help generate correct outputs

### Key Innovation Over Prior Work

**Previous Models (REALM, ORQA):**
- Only tackled extractive question answering
- Extract answer spans directly from retrieved text
- Limited to specific QA architectures

**RAG Innovation:**
- Works with any seq2seq task (generation, classification, QA)
- Generates free-form text rather than extracting spans
- General-purpose fine-tuning approach
- Uses existing pre-trained components (no expensive specialized pre-training needed)

### Results Snapshot

**Performance:**
- State-of-the-art on Natural Questions, TriviaQA, WebQuestions, CuratedTrec
- MS-MARCO abstractive QA, Jeopardy question generation
- FEVER fact verification (within 4.3% of SOTA)

**Human Evaluation:**
- 42.7% more factual than BART baseline (vs 7.1% less factual)
- 37.4% more specific responses

**Knowledge Update Demo:**
- Swapped Wikipedia index from 2016 → 2018
- 70% accuracy on 2016 world leaders with 2016 index
- 68% accuracy on 2018 world leaders with 2018 index
- Only 4-12% accuracy with mismatched indices
- Proves knowledge can be updated without retraining

---

## 2. Questions

### Question 1: Why do the authors freeze the document encoder during training?

**Question Context:**
The paper mentions keeping BERT_d (document encoder) frozen while only fine-tuning BERT_q (query encoder) and BART. This seems to limit the model's ability to adapt retrieval to specific tasks. Why make this choice?

**Answer Discussion Points:**
- **Computational cost**: Updating document encoder requires re-encoding all 21M Wikipedia passages and rebuilding the FAISS index after each gradient update - prohibitively expensive
- **Practical engineering decision**: Authors prioritize feasibility over theoretical optimality
- **Empirical validation**: Ablation studies (Table 6) show that fine-tuning just the query encoder provides significant gains over frozen retrieval
- **Pre-training quality**: DPR retriever already trained on Natural Questions and TriviaQA, so it's well-suited for knowledge-intensive tasks
- **Comparison to REALM**: REALM does update its index periodically during pre-training, but authors found this unnecessary for strong fine-tuning performance

**Implications:**
- Trade-off between computational resources and potential performance gains
- May leave some performance on the table, but enables practical training
- Future work could explore efficient index updating strategies

---

### Question 2: When should you use RAG-Sequence vs RAG-Token?

**Question Context:**
The paper introduces two variants with different marginalization strategies. RAG-Token outperforms on Jeopardy generation but RAG-Sequence wins on open-domain QA. What explains this?

**Answer Discussion Points:**
- **RAG-Sequence**: Uses same document for entire output sequence
  - Best for simple, factual answers requiring single source
  - More coherent when answer comes from one place
  - Example: "Who is the president of France?" → single fact

- **RAG-Token**: Can use different documents for each token
  - Best for complex answers requiring multiple facts
  - Can synthesize information from diverse sources
  - Example: Jeopardy about Hemingway needs "The Sun Also Rises" (doc 1) + "A Farewell to Arms" (doc 2)

**Practical Guidance:**
- Short factual answers → RAG-Sequence
- Complex multi-fact synthesis → RAG-Token
- Consider computational cost: RAG-Token is more expensive

---

## 3. Architecture

### Core Components

**1. Retriever: DPR (Dense Passage Retrieval)**
- **Architecture**: Bi-encoder with two BERT-base models
  - Query encoder: BERT_q(x) → converts query to dense embedding
  - Document encoder: BERT_d(z) → converts document to dense embedding
- **Function**: Computes similarity via dot product: d(z)ᵀ · q(x)
- **Retrieval**: Maximum Inner Product Search (MIPS) finds top-K documents
- **Pre-training**: Trained on TriviaQA and Natural Questions with contrastive learning
- **Role**: Non-parametric memory (external knowledge)

**2. Generator: BART**
- **Architecture**: Pre-trained seq2seq transformer with 400M parameters
- **Function**: Takes concatenated [input x, document z] → generates output tokens autoregressively
- **Pre-training**: Denoising objective with various corruption strategies
- **Role**: Parametric memory (reasoning and language generation)

**3. Document Index**
- 21M Wikipedia passages (100-word chunks from December 2018 dump)
- Each passage encoded to 728-dimensional vector
- Stored using FAISS with Hierarchical Navigable Small World approximation
- Enables sub-linear time retrieval

### Two RAG Variants

**RAG-Sequence: Document-Level Marginalization**

Mathematical formulation:
```
p(y|x) = Σ p_η(z|x) × p_θ(y|x,z)
     z∈top-K

where: p_θ(y|x,z) = ∏ p_θ(yᵢ|x,z,y₁:ᵢ₋₁)
                    i=1
```

Intuition:
- Pick one document, generate entire sequence using that document
- Do this for all K documents
- Weight each by how relevant the document is
- Sum to get final probability

Best for: Answers requiring consistent context from single source

**RAG-Token: Token-Level Marginalization**

Mathematical formulation:
```
p(y|x) = ∏ Σ p_η(z|x) × p_θ(yᵢ|x,z,y₁:ᵢ₋₁)
        i z∈top-K
```

Intuition:
- For each token position, consider all K documents
- Each token can come from a different document
- Marginalize at every token position
- Multiply token probabilities for sequence probability

Best for: Answers synthesizing information from multiple sources

### Pseudocode
```python
# RAG-Sequence Algorithm
def rag_sequence(query_x, target_y, K=5):
    # Step 1: Encode query
    q_embedding = BERT_query(query_x)
    
    # Step 2: Retrieve top-K documents
    top_K_docs = retrieve_topK(q_embedding, document_index, K)
    # Returns: [(doc₁, p(doc₁|x)), (doc₂, p(doc₂|x)), ..., (docₖ, p(docₖ|x))]
    
    # Step 3: Generate and marginalize
    total_prob = 0
    for doc, retrieval_prob in top_K_docs:
        context = concatenate(query_x, doc)
        
        # Generate full sequence autoregressively
        generation_prob = 1.0
        for i, token in enumerate(target_y):
            token_prob = BART(token | context, target_y[:i])
            generation_prob *= token_prob
        
        total_prob += retrieval_prob × generation_prob
    
    return total_prob

# Training loss
loss = -log(rag_sequence(x, y))
```
```python
# RAG-Token Algorithm
def rag_token(query_x, target_y, K=5):
    # Step 1: Encode query (same as RAG-Sequence)
    q_embedding = BERT_query(query_x)
    
    # Step 2: Retrieve top-K documents (same as RAG-Sequence)
    top_K_docs = retrieve_topK(q_embedding, document_index, K)
    
    # Step 3: Generate token-by-token, marginalizing at each step
    total_prob = 1.0
    
    for i, token in enumerate(target_y):
        # Marginalize over documents for this token
        token_prob = 0
        
        for doc, retrieval_prob in top_K_docs:
            context = concatenate(query_x, doc)
            p_token = BART(token | context, target_y[:i])
            token_prob += retrieval_prob × p_token
        
        total_prob *= token_prob
    
    return total_prob

# Training loss
loss = -log(rag_token(x, y))
```

**Key Difference:**
- RAG-Sequence: Marginalize **once** over documents (outer sum)
- RAG-Token: Marginalize **per token** (inner sum, outer product)

### Training Process

- Minimize negative marginal log-likelihood: -log p(y|x)
- Adam optimizer
- Document encoder BERT_d: **frozen**
- Query encoder BERT_q: **fine-tuned**
- Generator BART: **fine-tuned**

---

## 4. Critical Analysis

### Strengths

- **Parameter efficiency**: 626M trainable params achieves better results than T5-11B (11B params)
- **Interpretability**: Can inspect which documents were retrieved and how they influenced generation
- **Updatable knowledge**: Swap document index without retraining (proven with 2016→2018 experiment)
- **Reduced hallucination**: Grounding in retrieved documents improves factuality
- **General-purpose**: Single architecture handles QA, generation, classification tasks

### Limitations

**What the authors acknowledged but didn't deeply explore:**

1. **Retrieval collapse**: On creative tasks (story generation), model retrieves same documents regardless of input, then ignores them. No clear guidance on when RAG is appropriate vs inappropriate.

2. **Fixed document encoder**: Computational necessity, but limits task-specific adaptation. How much performance is lost? Not thoroughly investigated.

3. **Single knowledge source**: Only Wikipedia. What about:
   - Multiple knowledge sources with different reliability?
   - Domain-specific knowledge bases?
   - Temporal knowledge that changes?

4. **Computational cost**: RAG-Sequence with 50 documents is expensive. Limited discussion of production deployment trade-offs.

**What could have been developed further:**

1. **Hybrid approaches**: Could combine RAG-Sequence and RAG-Token advantages (e.g., adaptive switching)
2. **Iterative retrieval**: All documents retrieved at once. Could retrieve → generate partial → retrieve more based on partial output.
3. **Failure analysis**: Limited investigation into when and why RAG fails
4. **Cross-task learning**: Each task fine-tuned separately; multi-task learning could help

### How It Holds Up

**Strong aspects:**
- Results are reproducible (code released, widely replicated)
- Ablation studies are thorough (BM25 vs DPR, frozen vs learned retrieval)
- Human evaluations add credibility beyond automatic metrics

**Valid concerns:**
- Some hyperparameter choices lack justification (why K=5 or 10?)
- Limited guidance on task suitability
- Comparison to T5 somewhat unfair (different Wikipedia dumps, training procedures)

---

## 5. Impact

### How It Changed AI

**Immediate Impact (2020-2021):**
- Established retrieval-augmented models as viable alternative to pure parameter scaling
- Showed 626M + retrieval can beat 11B parameter-only models
- Inspired follow-up research: FiD (Fusion-in-Decoder), RETRO, Atlas

**Long-term Impact (2021-Present):**
- Foundation for modern production RAG systems
- Influenced: ChatGPT browsing, Claude with search, Perplexity AI
- Spawned ecosystem: LangChain, vector databases (Pinecone, Weaviate), embedding models
- Became standard approach for grounding LLMs in proprietary/current knowledge

### Importance Relative to Other Work

**Building on:**
- BERT (encoders), BART (generation), DPR (retrieval), REALM/ORQA (hybrid models)

**RAG's contribution:**
- Unified these into general-purpose framework
- Extended beyond extractive QA to full generation
- Demonstrated practical updatability

**Paradigm shift:**
- Separated "how to think" (parametric) from "what to know" (non-parametric)
- Showed knowledge doesn't need to be in parameters
- Enabled smaller, more maintainable, more trustworthy systems

---

## 6. Resource Links

1. **[Original Paper (arXiv)](https://arxiv.org/abs/2005.11401)** - Full paper with experiments and analysis
2. **[HuggingFace Implementation](https://github.com/huggingface/transformers/tree/master/examples/rag)** - Official code with examples
3. **[Interactive Demo](https://huggingface.co/rag/)** - Try RAG in browser
4. **[DPR Explained](https://www.geeksforgeeks.org/nlp/what-is-dense-passage-retrieval-dpr/)** - Understanding retrieval component
5. **[FAISS Documentation](https://github.com/facebookresearch/faiss)** - Similarity search library

---

## 7. Code Demonstration

*See rag_demo.ipynb for interactive demonstration*

---

## 8. Repository

This repository contains:
- **README.md**: This comprehensive overview
- **rag_demo.ipynb**: Code demonstration notebook
- **presentation_slides.pdf**: Slide deck
- **figures/**: Key diagrams from paper

---

## 9. Citation
```bibtex
@article{lewis2020retrieval,
  title={Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and Riedel, Sebastian and Kiela, Douwe},
  journal={arXiv preprint arXiv:2005.11401},
  year={2020}
}
```

**Authors**: Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela

**Affiliations**: Facebook AI Research, University College London, New York University

**Published**: NeurIPS 2020

---

## Quick Reference

**Model Stats:**
- 626M trainable parameters (110M query encoder + 406M BART; 110M doc encoder frozen)
- 21M Wikipedia passages × 728 dimensions = 15.3B values in index
- Outperforms T5-11B on several benchmarks

**Key Results:**
- Natural Questions: 44.5 EM (vs T5-11B: 34.5)
- Human eval: 42.7% more factual than BART
- Knowledge update: 70% accuracy with correct index, 4-12% with wrong index
