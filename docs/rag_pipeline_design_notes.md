# RAG Pipeline Components

### 1. Ingestion & Parsing
- Convert raw data (e.g PDFs) into structured text.
- Preserve metadata, e.g page numbers, section titles, source, date etc

    *Choose a parsing method & what metadata to preserve.*

### 2. Chunking
- Split documents into chunks to be embedded. 
- Chunks should be large enough to capture context, but small enough that they contain a distinct idea which could be retrieved to answer a question.

    *Choose a chunking method based on the types of documents.*

### 3. Embedding
- Convert document chunks & the query into vectors.

    *Choose an embedding model.*

### 4. Vector Store
- Where embeddings are stored and retrieved from.

    *Choose a vector store, indexing method and metadata filtering.*

### 5. Query processing
- Query re-writing
- HyDE: hypothetical document embeddings
- multi-query generation
- decomposition of complex questions into sub-questions

### 6. Retrieval
- Retrieve the k most relevant chunks

    *Choose k and a search technique.*

### 7. Re-ranking
- Cross-encoder or LLM that assigns a relevance score to each of the initial retrieved chunks
- Retain the n-most relevant chunks

    *Choose a re-ranking model and value of n*

### 8. Context assembly & prompting
- Construct the final prompt
- Include instructions for when the model doesn't know the answer

    *Choose chunk order and how metadata is injected.*

### 9. Generation
- Pass the query + context to the LLM

    *Choose which LLM is most suitable, temperature, response schema / structured output format*

By default, temperature is set to 0.8 by Ollama. I want to set the temperature to 0 to minimise the variation in ouputs from run to run.

### 10. Evaluation
- retrieval quality: precision, recall@k, MRR
    - these can be calculated by manually labelling a golden dataset of queries mapped to relevant document IDs
- generation quality: faithfulness/groundedness, answer relevance, hallucination rate
    - faithfulness: whether the generated answer is actually supported by the retrieved information (i.e not hallucinating). Using the RAGAS framework this is calculated as the `number of claims supported by retrieved info` divided by `the total number of claims` (the answer is first broken down into 'claims')
    - answer relevance: whether the answer actually addresses the question asked (penalises incomplete or off-topic answers, even if faithful). The RAGAS framework uses an LLM to generate synthetic questions that the answer could plausibly be answering, embeds them and calculates the cosine similarity between them and an embedding of the actual question.
- frameworks: RAGAS, DeepEval

#### Choosing an LLM judge for RAGAS

`RAGAS` (Retrieval Augmented Generation ASessment) is a library for evaluating RAG pipelines. It provides built-in metrics such as faithfulness and answer relevance. I will use these for the first version of my pipeline, but later on I will consider which metrics are most appropriate to my problem space.

Note that RAGAS uses an LLM judge to calculate metrics. A potential source of bias is using the same LLM as a judge and as the generation model, since the model is more likely to rate it's own outputs higher than those of another model. Bias could also be introduced by choosing an LLM judge from the same family as the generation model. Since I'm using Llama 3.2 as my generation model, I'll look outside of the Llama family for my judge model.

Since RAGAS involves multiple reasoning steps, a reasonably powerful model should be used as the judge.

Another constraint is compute power; I'm running models locally via Ollama. `gpt-oss:20b` meets all of these constraints, so I'll test this out first. I need to set `max_tokens` for the model. This needs to cover reasoning (which can be set to `low`, `medium` or `high`) and the JSONs that the judge outputs in order to calculate metrics. If `max_tokens` is too low, the JSONs might be incomplete, leading to a parsing error and `NaN` scores. Ideally we'd run some tests to see what range of token values RAGAS uses for our specific documents. For now, for the MVP, we'll set a high `max_tokens` value and revisit this if there are errors.

It's important to check that the LLM judge is aligned with what a human would say. To check for this alignment I'll need to manually label some examples and compare to the judge's decisions.
    
Another implication of using LLM-as-a-judge for evaluation is the cost and latency of extra LLM calls. However, latency may not be too much of an issue if evaluation is run offline.

##### Context window size

We've briefly discussed the token limit. With Ollama models, there's also a context window cap set by the `num_ctx` parameter. By default this is 4000.

The context window will contain:
- the prompt for the LLM judge: this is a fixed template written by RAGAS, containing an instruction, a JSON schema for the expected output, and few-shot examples (these are general, not specific to my domain).
- the question and answer generated by the generation LLM
- the 'context': the relevant chunks from the source docs that the generation LLM returns as supporting evidence
- the output: when calculating faithfulness, the output for each claim (the judge splits each answer into standalone claims) is a 'reason' (why the claim is or is not supported by the context) and a 'verdict' (a score between 0 and 1)

If, for example, some of the 'context' doesn't fit inside the context window, there is no error but the judge might mark a claim as 'unsupported' and the resulting faithfulness score will be lower than it should be.

#### Summary

- Build a golden dataset of queries mapped to relevant document IDs. Automatically calculate retrieval quality metrics on this test set each time the pipeline is updated.
- Use RAGAS to calculate generation-quality metrics, starting with faithfulness and answer relevancy. Choose which LLM to use as the judge and validate it against human-labelled samples.
- Consider calculating domain-specific metrics (e.g custom judge metrics) in the future.


### 11. Serving & Orchestration
- caching
- async / streaming
- latency budgets
- updates to the vectore store as new documents become available

### 12. Observability & Monitoring
- logging queries, retrieved chunks & generations
    - LangSmith?
- record latency at each stage
- capture user feedback to help find failure modes


# Breaking down the problem

First we want to load, parse, chunk and index the documents into the vector store. This can be considered an 'offline' phase.

Once we have a vector store, we can perform retrieval and generation, i.e get an answer to a query. 

So, we actually need to build 2 pipelines, one for each of the above stages.

In this project we'll use LangChain.

Once we have a basic RAG system working, we want to evaluate its performance. From there we can consider adding re-ranking, query processing, context assembly etc and see whether each change improves performance.