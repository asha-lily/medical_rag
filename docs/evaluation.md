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


#### Test case 3

- **Question:** "Can I take ibuprofin while breastfeeding?"
- **Answer:** "I don't have enough information to answer that question based on the available documents.

The only relevant section regarding breastfeeding is [2] paracetamol_PIL.pdf, page 2, which states that "You can take paracetamol while breastfeeding." However, ibuprofen's safety during breastfeeding is not mentioned in the provided source documents."

```
Sources:
  [1] ibuprofen_PIL.pdf, page 4 (Pregnancy, breast-feeding and fertility)
  [2] paracetamol_PIL.pdf, page 2 (Pregnancy and breastfeeding)
  [3] cetirizine_PIL.pdf, page 2 (Pregnancy and breast-feeding)
  [4] motion_sickness_PIL.pdf, page 1 (Pregnancy and breast-feeding)
  [5] motion_sickness_PIL.pdf, page 2 (Pregnancy and breast-feeding)]
```

It's interesting that only one of the sources are the ibuprofen document. The referenced page is not the only part of the document that references breast feeding, so it seems that not all relevant chunks have been retrieved from this document, and chunks from other documents are deemed more relevant.

The relevant chunks in `ibuprofen_PIL.pdf` are:

```
{'chunk_index': 0, 'page': 1, 'source': 'ibuprofen_PIL.pdf', 'medicine_name': 'Ibuprofen', 'chunk_id': 'ibuprofen_PIL.pdf#5.0', 'heading': 'Do not takeIbuprofen', 'section_index': 5}
Medicine: Ibuprofen
Section: Do not takeIbuprofen

If you are allergic to ibuprofen or any of the other ingredients of this medicine (listed in section6).
if you have had allergic reactions such as asthma, runny nose, itchy skin rash or swelling of the lips, face, tongue, or throat after you have taken medicines containing acetylsalicylic acid (such as aspirin) or other medicines for pain and inflammation(NSAIDs)
If you have suffered from gastrointestinal bleeding or perforation related to previous use of drugs for pain and inflammation(NSAIDs)
If you are suffering from an ulcer or bleeding in the stomach or small intestine (duodenum) or if you have had two or more of these episodes in thepast
If you suffer from severe liver, kidney or heartproblems
If you pregnant, planning to become pregnant or breastfeeding
If you are suffering from significant dehydration (caused by vomiting, diarrhoea or insufficient fluid intake)
If you have any active bleeding (including in thebrain)
Have a condition which increases your tendency tobleeding
```

```
{'medicine_name': 'Ibuprofen', 'chunk_id': 'ibuprofen_PIL.pdf#6.0', 'page': 1, 'heading': 'Warnings and precautions', 'source': 'ibuprofen_PIL.pdf', 'chunk_index': 0, 'section_index': 6}
Medicine: Ibuprofen
Section: Warnings and precautions

Talk to your doctor before taking Ibuprofen
if you have Systemic Lupus erythematosus (SLE) or mixed connective tissuediseases
if you have inherited a disorder of the red blood pigment haemoglobin(porphyria)
if you have chronic inflammatory intestinal diseases such as inflammation of the colon with ulcers (ulcerative colitis), inflammation affecting the digestive tract (Crohn's disease) or other stomach or intestinal diseases
if you have disturbances in the formation of bloodcells
if you have problems with normal blood clottingmechanism
if you suffer from allergies, hay fever, asthma, chronic swelling of nasal mucosa, sinuses, adenoids, or chronic obstructive disorders of the respiratory tract because the risk for developing narrowing of the airways with difficulty in breathing (bronchospasm) isgreater
if you have liver, kidney, or heartproblems
if you have justhad majorsurgery
if you are in the first six months ofpregnancy
if you arebreast-feeding
```

A potential reason why these chunks weren't retrieved is that 'breast feeding' is referenced in only one of a long list of bullet points so it likely carries little weight in the embedding of the whole chunk.

There's also the misspelling of `ibuprofen` as `ibuprofin`. When this is spelled correctly in the query, the answer is almost identical. However, the sources are not. While `ibuprofen_PIL.pdf` only appeared once in the list of 5 sources for the question with the misspelling, this time it appeared 4 times.

```
Sources:
  [1] ibuprofen_PIL.pdf, page 4 (Pregnancy, breast-feeding and fertility)
  [2] paracetamol_PIL.pdf, page 2 (Pregnancy and breastfeeding)
  [3] ibuprofen_PIL.pdf, page 1 (Do not takeIbuprofen)
  [4] ibuprofen_PIL.pdf, page 1 (Do not takeIbuprofen)
  [5] ibuprofen_PIL.pdf, page 3 (Other precautions)
```


# Building a ground truth data set

I'll build a dataset of ground truth samples, where each sample contains the following:

- question
- answer: the correct answer I'd expect based on the information in the documents
- evidence: the relevant information in the documents

I will include a variety of questions which test different aspects of the system, e.g:

- Queries asking about a medicine that doesn't exist in the dataset
- Queries in which the medicine is in the dataset, but is spelled incorrectly in the query
- Queries which can't be answered by the information in the leaflet

This dataset will be split into 2 subsets, which can be thought of as `training` and `test` data sets:

- `training` data: I will run the evaluation pipeline on this data multiple times; at each iteration I'll manually review the results and make updates to the RAG pipeline in order to address any issues identified.
- `test` data: this is 'unseen' data in the sense that I'm not making changes to the pipeline based on how it performs on this data. This should give us a better indication of 'real-world' performance, provided the data set is large and diverse enough.

Given the manual work required to build this dataset, I'll build a small training set to start with, while acknowledging that if this system were to be deployed (especially in a medical setting), a large and diverse test set would be essential. I'll need to be systematic about building the test set with representation across all medicine types and real-world scenarios, without bias towards the scenarios that I know the system performs well on. I think using an AI coding tool will be beneficial here!

While building this dataset I decided to add some additional fields to each sample:

- medicine (list[str]): the names of the medicines that the question asks about. In future I could add an evaluation metric to check whether the sources cited in the answer come from the correct medicine document.
- category (enum): a label to help ensure good coverage of different topics. The categories currently include:      
    - `side_effects`
    - `out_of_scope_medicine`: i.e a medicine that we don't currently have a document for
    - `breastfeeding`
    - `max_dose`
    - `pregnancy`
    - `suitable_conditions`: i.e asking whether a medicine can help with a specific condition
    - `combinations`: questions asking about whether multuple different medications can be taken together
    - `not_suitable_for_patient`: the patient has a condition that is listed under 'do not take this medicine if you...' or similar
    - `general`
- answerable (bool): whether or not the question can be answered given the information in the documents

I've also broken down the `evidence` field into:

- source (enum): the name of the document containing the information needed to answer the question
- quote (str): a quote of the piece of information needed to answer the question

If there are multiple quotes relevant to a given sample, then the `evidence` field can contain multiple source-quote pairs.


# Evaluation metrics

## Retrieval Quality

For a given query, we can identify the ID(s) of the chunk(s) containing relevant information to answer the query. The retriever returns the K chunks that are most semantically similar to the query. [Common metrics](https://www.evidentlyai.com/llm-guide/rag-evaluation) related to this include:

- Precision@k: of the top k retrieved items, how many are actually relevant?
- Recall@k: of all relevant items, how many were retrieved in the top k?
- Hit rate: Did at least one relevant item appear in the top k? (yes/no)
- NDCG@k (Normalized Discounted Cumulative Gain): rewards correct items appearing higher in rank
- MRR (mean reciprocal rank): inverse of the rank of the first relevant chunk among all retrieved chunks.

Since safety is a priority, recall@k is a good metric as it will reflect the number of relevant chunks that retrieval misses. Missing a chunk translates to missing potentially important information.

MRR tells us whether the most useful chunk was retrieved as one of the most important. This relates to ranking, so when making changes such as adding a re-ranker, changes in ranking will be reflected in MRR.


## RAGAS: Answer relevancy & Faithfulness

`Answer relevancy` and `faithfulness` are defined in `notes/rag_pipeline_design_notes.md`, along with the reasoning behind the choice of RAGAS LLM judge and other parameters.

The RAGAS prompt template for answer relevancy asks the judge to mark "evasive, vague" answers like "I don't know" as non-committal, and assigns a score of 0. We don't want to penalise refusal, as it's important that this system refuses to answer when it can't find relevant information in the documents, instead of making something up or misinterpreting the documents.

This is why each ground truth sample has an `answerable` field, which is `false` when the question can't be answered using information in the documents. In such cases, the generation model should refuse to answer the question. The prompt (the baseline version, at least) addresses this specifically by saying:

```
If the context does not contain enough information to answer the question confidently, say "I don't have enough information to answer that question based on the available documents."
```

In the evaluation pipeline, currently only samples for which `answerable` is `true` are sent to RAGAS. We need to measure refusal rate separately, i.e for all of the unanswerable questions in the ground truth dataset, for what % does the system refuse to answer? 

We also want to measure how often the model is overly cautious and refuses to answer questions that are answerable; we'll call this `false refusal rate`. Since samples that are answerable currently get sent to RAGAS, if they're refused they'll get an answer relevancy score of 0, which is fair since it's a failure. When calculating faithfulness, RAGAS splits the answer into factual statements. When the model refuses, we expect the answer to look something like *I don't have enough information to answer that question...*, which would likely get a low faithfulness score due to the absence of similar meaning in the documents. If the output is empty then the faithfulness score is `NaN`. A `NaN` result can also happen due to an LLM judge call timing out or returning a JSON that can't be parsed (e.g due to the context window being too small); this is something to consider in the future.

The third case is when a question is unanswerable but the model answers. We could call these cases `missed refusals`, and calculate `missed refusal rate` as `1 - refusal rate`. We also don't want to send these to RAGAS.

If we can classify whether or not an answer is a refusal, we can make sure we only send samples that are both answerable and answered to RAGAS. With the other samples, we can calculate `refusal rate`, `false refusal rate` and `missed refusal rate`.


### Classifying refusals

So, how can we build such a classifier?

#### Option 1

I think the quickest method is for me to manually review answers and label whether or not they are refusals. This means I need to separate out running the RAG pipeline on my ground truth dataset, and calculating metrics. Between these two steps I'll add my manual labels, which will be a new `refusal` column that I'll label as true/false.

How should I handle ambiguous cases, which aren't clear refusals but also aren't clear answers? For example "the leaflet doesn't mention X, but it does say Y...". If the question were asking about X, then I think I'd label this as a refusal.

I think this method is the correct choice for the MVP phase where I'm working with a small, manageable dataset, but of course this method won't scale. See the next two options.

I also considered automatically checking whether "I don't have enough information to answer that" appears in the answer, however I've seen refusals that use slightly different wording.

#### Option 2

I could configure the generation model to return a structured output, with an `answered` field which must be set to true or false. The challenge with this is that the `answered` field might disagree with the answer itself, so I'd still need to do some manual labelling in order to measure this alignment.

#### Option 3

I could of course train a classifier, for example a linear probe on top of a text encoder, but this would require a lot of labelled data. I could revisit this type of method in the future.


# Baseline results

Now that I have a training set of 13 samples, I'll run my evaluation pipeline to calculate retrieval metrics (`run_retrieval_evaluation.py`) and RAGAS evaluation (`ragas_evaluation.py`) to get a set of baseline results. When I make changes to the system, I'll run evaluation again and compare the results to the baseline.

The results from this run are saved in `results/retrieval_baseline.csv`.


### Ideas for pipeline improvements

As mentioned above, the purpose of the training dataset is to surface issues with the pipeline. Issues identified in the latest iteration are listed below:

- To reduce the risk of retrieving information from the wrong PIL, I could add a classification step (i.e identify which medicine the query is asking about) and filter the chunks that can be retrieved from to those for the specific medicine.
    - Users may ask questions using brand names instead of the generic medicine name, so I could create mappings from brand names to medicine names to use in this classification step.
    - Users may misspell the medicine name. In test case 3 we saw how this can affect retrieval. The classification step could use fuzzy matching to identify the correct medicine name.


### Next steps

- Run evaluation pipeline to calculate metrics
- Run RAGAS evaluation
- Manually inspect what went wrong and propse changes to the RAG pipeline to address these issues