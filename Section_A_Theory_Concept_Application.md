# TOPS Technologies — Data Science Assessment (M16-A1)
## Section A: Concept Application (Theory)
**Modules Covered:** ML Deployment (Mod 13) | Generative AI (Mod 14) | Transformers & LLMs (Mod 15) | LangChain (Mod 16)  
**Domain:** Food Delivery Platform

---

### S1: Model Serialisation for Flask Production Deployment
**Scenario:** Deploying a scikit-learn delivery time prediction model to a production Flask service handling hundreds of requests per hour.  
**Question:** Compare Pickle and Joblib as serialisation choices for this deployment context. Which would you use and why? What risks does each carry when the model is loaded back in an environment that differs from the one it was trained in?

**Answer:**
For deploying scikit-learn models containing numerical arrays, fitted estimators, and pipeline weights, **Joblib** is the superior serialisation choice over standard Python `pickle`. Joblib is specifically optimized for Python objects with large NumPy array buffers, providing faster disk I/O, lower memory overhead, and internal memory-mapping during deserialization. This makes Joblib significantly more responsive when spawning or reloading worker processes in a high-throughput Flask web service.

However, both Pickle and Joblib carry critical risks when deserializing across disparate environments. First, they are tightly coupled to exact package versions; differing versions of `scikit-learn`, `numpy`, or `python` between the training notebook and production container can trigger unpickling crashes (`AttributeError`, `ModuleNotFoundError`, or silent numerical drift due to altered internal class structures). Second, both formats deserialize arbitrary Python bytecode, introducing severe remote code execution (RCE) vulnerabilities if an untrusted or compromised model file is loaded.

---

### S2: Generative Architecture for Marketing Image Generation
**Scenario:** Auto-generating high-quality promotional dish imagery using text descriptions and reference photos of plated meals.  
**Question:** Compare Variational Autoencoders (VAEs) and Stable Diffusion models for this task — address key architectural differences, output quality, and which is more appropriate for a production marketing pipeline.

**Answer:**
**Variational Autoencoders (VAEs)** employ an encoder-decoder architecture that compresses images into a continuous latent distribution using reconstruction loss and Kullback–Leibler (KL) divergence. Because VAEs optimize pixel likelihoods across continuous Gaussian priors, they inherently generate blurry, low-frequency images lacking fine culinary textures like grill marks, steam, or seasoning. 

In contrast, **Stable Diffusion** models operate by progressively reversing a latent diffusion process using a U-Net or Diffusion Transformer conditioned on CLIP/T5 text embeddings and spatial image adapters (such as ControlNet or IP-Adapter). This allows the model to faithfully preserve the plating geometry of the reference photo while rendering photorealistic, high-resolution textures. For a commercial food marketing pipeline where visual appetite appeal and fine detail are paramount, **Stable Diffusion is substantially superior and the industry-standard choice**.

---

### S3: Transformer Architectures for Conversational and Classification Tasks
**Scenario:** Food delivery AI assistant handling: (1) open-ended questions ("What cuisines are trending near me?"), and (2) classifying incoming customer reviews as Positive, Neutral, or Negative.  
**Question:** Compare GPT-style decoder-only models and BERT-style encoder-only models. Which architecture is better suited to each job, and what fundamental property of each architecture makes it the right choice?

**Answer:**
Job (1)—answering open-ended recommendations—is best suited to **GPT-style decoder-only models**. The fundamental property that makes GPT ideal is its **causal (unidirectional) masked self-attention**, which prevents tokens from attending to subsequent tokens and enables natural, autoregressive sequence generation token-by-token. This property allows the assistant to craft fluent, open-ended conversational replies tailored to varied queries.

Job (2)—sentiment classification of reviews—is best suited to **BERT-style encoder-only models**. The fundamental property here is **bidirectional self-attention**, which processes all tokens in the customer review simultaneously across left and right contexts. This full-context visibility produces rich, holistic pooled sentence representations (`[CLS]` embeddings) that can be fine-tuned into a compact, low-latency classification head, delivering higher accuracy and throughput for dashboard analytics than generative decoders.

---

### S4: Few-Shot vs. Zero-Shot Prompting for Customer Support
**Scenario:** Customer support bot handling unseen complaint categories ('wrong item delivered', 'packaging damaged', 'missing cutlery') with only 3 annotated pairs per category.  
**Question:** Explain the difference between zero-shot and few-shot prompting. Which strategy would you apply under these constraints, and how would you structure the prompt to maximise response quality?

**Answer:**
**Zero-shot prompting** provides the model with only high-level instructions and task descriptions without any concrete input-output demonstrations, relying strictly on the model's pre-trained associations. In contrast, **few-shot prompting** provides explicit exemplar pairs (input query and desired response) within the prompt context, demonstrating the required tone, boundary logic, and structure.

Given that the bot must handle previously unencountered complaint categories with only 3 annotated pairs each, **few-shot prompting is the definitive choice**. The prompt should be structured in three distinct sections:
1. **System Persona & Policy Instruction:** Define the agent role (empathic, concise QuickBite support agent) and refund/replacement policies.
2. **Exemplar Demonstrations:** Embed the 3 annotated examples per category formatted consistently:  
   `Customer: [Complaint text]` -> `Support Agent: [Apology + targeted resolution + ticket tag]`.
3. **Current Task Trigger:** Present the incoming customer query with explicit delimiters and an empty prompt marker (`Support Agent:`) to cue immediate, formatted response generation.

---

### S5: LangChain Agents vs. Simple LLM Chains
**Scenario:** Food ordering assistant needing to check live restaurant availability, calculate delivery time via API, and validate discount codes.  
**Question:** Describe what distinguishes a LangChain agent from a simple LLM chain. Justify which approach is more appropriate for this assistant, and identify two risks the agent approach introduces.

**Answer:**
A simple **LLMChain** follows a predetermined, static execution graph where inputs pass through fixed prompts and parsers in a rigid pipeline. Conversely, a **LangChain Agent** uses an LLM as a dynamic reasoning engine (typically via ReAct or Tool Calling) that inspects user queries, determines at runtime which external tools to call, parses tool outputs, and loops recursively until the task is complete.

Because the food ordering assistant must conditionally perform real-time actions—checking live table/dish availability, invoking an external delivery time calculation API, and validating discount coupon rules—a **LangChain Agent is far more appropriate**. 

However, the agent architecture introduces two prominent risks:
1. **Execution Non-determinism and Latency Loops:** The agent may enter repetitive reasoning loops, select inappropriate tools, or exceed token budgets, resulting in unpredictable response latency.
2. **Tool Invocation Errors and Security Risks:** The LLM may hallucinate tool arguments (e.g., passing invalid discount codes or coordinates) or be vulnerable to prompt injections that trigger unintended API actions.

---

### S6: Retrieval-Augmented Generation (RAG) vs. Model Fine-Tuning
**Scenario:** Food delivery chatbot answering questions about a 500-page policy document and a menu catalogue updated daily.  
**Question:** Compare fine-tuning and RAG for this scenario addressing update frequency, cost, hallucination risk, and data privacy. Which approach is more suitable, and what limitations does the recommended approach carry?

**Answer:**
For a daily-updating menu catalogue and 500-page policy manual, **Retrieval-Augmented Generation (RAG)** is substantially superior to fine-tuning across all four evaluative criteria:
- **Update Frequency:** RAG handles daily menu changes effortlessly by updating or upserting embeddings in a vector database in seconds, whereas fine-tuning requires continuous, costly re-training pipelines.
- **Cost:** Fine-tuning incurs high GPU compute and human annotation costs on every update; RAG incurs only marginal embedding generation costs.
- **Hallucination Risk:** Fine-tuned weights suffer from catastrophic forgetting and parametric hallucination; RAG explicitly grounds LLM answers in retrieved source chunks with verifiable citations.
- **Data Privacy:** RAG allows granular role-based access control (RBAC) at the retrieval chunk level, whereas fine-tuning bakes sensitive policies indelibly into model parameters.

**Remaining Limitations of RAG:**  
Despite being the recommended solution, RAG still carries key limitations: retrieval failures if semantic queries don't match dense policy legal terms, chunk fragmentation (where crucial discounts or conditions cross chunk boundaries), and increased inference latency due to vector search overhead and long context prompt processing.
