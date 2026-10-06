# Initial end-to-end tests

#### Test case 1

- **Question:** "How much ibuprofen should I take?"
- **Answer:** "I don't have enough information to answer that question confidently based on the available documents. The question asks for a specific amount of ibuprofen to take, but there is no clear answer provided in the context. The documents provide general dosage guidelines and potential side effects, but not a specific recommendation for an individual taking ibuprofen."

This is a good response; the model admits that it doesn't know instead of making something up or returning irrelevant information, and it provides details about why it can't answer the question. The system prompt specifically says `If the context does not contain enough information to answer the question confidently, say "I don't have enough information to answer that question based on the available documents."`

When I build an evaluation set I will include several examples like this.


#### Test case 2

- **Question:** "What are the side effects of paracetamol?"
- **Answer:** "According to the information provided in the document [1] Source: paracetamol_PIL.pdf, page 3, the possible side effects of paracetamol are:

    * Skin rashes or other signs of allergic reaction such as itching, swelling of the lips, tongue or throat, or difficulty in breathing
    * Unexpected bruising or bleeding, persistent tiredness, or increase in the number of infections (e.g. sore throats) which you get.
    * These are possible signs of changes in the blood.

    Additionally, some side effects are more likely if you have experienced them before when taking other painkillers such as ibuprofen and aspirin:

    * Breathing problems
    * Nausea

    The document also mentions that breathing problems may be more likely if you have a history of experiencing this symptom while taking other painkillers (section [3] on page 3).

    It is worth noting that the source explicitly states that side effects are rare with paracetamol, but not everybody gets them."


This answer is relevant to the question and the source document and page number cited are correct. The first 3 bullet points are copied word-for-word from the original document. 

However, the answer leaves out some information which I would consider important:
- "Stop taking the tablets and contact your doctor immediately if you notice...". In the original document this appears before the 3 bullet points that this answer quotes. This provides an important instruction for the patient.

Manually reviewing question-answer pairs highlights issues like this which might warrant adding additional details to the prompt. For example, to address this particular issue I could add something like "Pay attention to the severity of side effects listed in the document. If the document suggests an action for the patient to take given a specific side effect, please raise this with the patient". This change to the prompt would of course need to be carefully tested by re-running the evaluation pipeline.


## Evaluation

Ideas for queries to test:

- Queries asking about a medicine that doesn't exist in the dataset
- Queries in which the medicine is in the dataset, but is spelled incorrectly in the query

- Queries which can't be answered by the information in the leaflet
- Queries ranked by how risky an incorrect answer would be


### Evaluation metrics

#### Retrieval Quality

For a given query, we can identify the ID(s) of the chunk(s) containing relevant information to answer the query. The retriever returns the K chunks that are most semantically similar to the query. [Common metrics](https://www.evidentlyai.com/llm-guide/rag-evaluation) related to this include:

- Precision@k: of the top k retrieved items, how many are actually relevant?
- Recall@k: of all relevant items, how many were retrieved in the top k?
- Hit rate: Did at least one relevant item appear in the top k? (yes/no)
- NDCG@k (Normalized Discounted Cumulative Gain): rewards correct items appearing higher in rank
- MRR (mean reciprocal rank): inverse of the rank of the first relevant chunk among all retrieved chunks.

Since safety is a priority, recall@k is a good metric as it will reflect the number of relevant chunks that retrieval misses. Missing a chunk translates to missing potentially important information.

MRR tells us whether the most useful chunk was retrieved as one of the most important. This relates to ranking, so when making changes such as adding a re-ranker, changes in ranking will be reflected in MRR.



#### Answer relevancy 

The RAGAS prompt template for answer relevancy asks the judge to mark "evasive, vague" answers like "I don't know" as noncommittal: 1. In RAGAS's code, the score is then multiplied by zero when all 3 generated questions are marked noncommittal.

That means a correct refusal scores 0 on answer relevancy. For example, the ibuprofen test case, where the system rightly said it didn't have enough information, would score 0.

In this project, refusing is often the safe and correct behaviour, but the metric penalises this. 

When building the golden set, I should:

- Label which questions should be refused.
- Report answer relevancy only on questions that should be answered.
- Separately, measure refusal accuracy: did the system refuse when it should have, and answer when it should have?