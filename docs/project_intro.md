# Background and motivation

I'm interested in how AI is being used in healthcare, and how it might be used in the future.

One such application is chatbots designed to answer health and medicine-related questions. These already exist.

This initially struck me as very risky, and in need of very thorough evaluation and strict guardrails to minimise potential harm to users. Returning the wrong dose, or hallucinating a seemingly minor detail could be deadly. As chatbots become agentic, with access to tools and the ability to autonomously carry out multiple actions, the risks multiply.

My curiosity about the potential benefits of using AI in healthcare, and the challenges of doing this safely, motivates this project. 

I'd like to build an agentic RAG system in which the retrieval step has access to patient information leaflets (PILs). These are the leaflets inside every pack of medicine sold in the UK. They have a fairly consistent structure, usually along the lines of:

- What the medicine is for
- Before you take it
- How to take it
- Side effects

These documents could be used to answer questions such as:
- How many paracetamol tablets can I take in 24 hours?
- What should I do if I miss a dose of my contraceptive pill?
- Is it safe to take ibuprofen while breastfeeding?
- What are the common side effects of sertraline?

A system like this could be useful to the average person who may have over-the-counter medicines at home or in their bag but without the original information leaflet.

It's crucial to note that the aim of this project is not to build an accurate, safe system that can be relied on, but rather to explore the challenges associated with applying RAG and agentic behaviour to a relatively simple and constrained health-related problem space.

I'm setting out with the following plan:
1. Build a simple RAG pipeline that can answer questions by retrieving relevant information from patient information leaflets.
2. Evaluate this pipeline on basic metrics, including using RAGAS with an LLM judge to evaluate faithfulness and answer relevancy.
3. Build a second retriever with access to a vector store of medical guideline documents. 
    - Where PILs address the question of 'how to take a medicine safely', guidelines address 'what's the right treatment or care'
    - Providers of guidelines include https://www.sign.ac.uk/guidelines/pharmacological-management-of-migraine/ & https://www.who.int/publications/who-guidelines
4. Give an agent access to both retrievers (PIL & guidelines) and let it choose which to use for a given question (i.e routing). Measure routing accuracy.
    - The user could then ask a question such as "My doctor prescribed amlodipine for high blood pressure. Why this medicine, and how should I take it?" The "why" is in the guideline; the "how" is in the PIL.
5. Add more complex agentic behaviour, e.g provide access to tools, enable the agent to ask clarifying questions, plan sub-queries etc.

A key thing to measure will be whether the agent admits when it doesn't have the relevant information to answer the question, since hallucinating an answer puts the user at great risk in this domain. The system should also return citations with every answer.


# Patient Information Leaflets

The website https://products.mhra.gov.uk/ provides access to these leaflets.

I'll start by creating a list of 10 common medicines. I found a list from a Manchester pharmacy, titled `Essential Over-the-Counter Medications: A Must-Have for Every Household`. Some of the items were categories, e.g 'antihistamines'. In these cases I manually chose a specific medicine to represent the group.

- paracetamol
- ibuprofen
- cetirizine
- antacid
- oral rehydration solution
- decongestant
- cough suppressant
- antiseptic cream
- hydrocortisone cream
- motion sickness medication

When searching for each medicine on the MHRA website, the results should be filtered such that `Type of document = Patient Information Leaflet (PIL)` and `Applicable to territory = United Kingdom (UK)`. Each leaflet is a PDF which is downloaded and saved in `PILs`.

An important note at this stage is that there's a lot of variety in the format of PILs from different providers. To build a robust RAG system I would want to sample from a diverse range of formats and make sure the parsing step works for all of them. For the MVP stage of this project I'm intentionally using a small dataset, and while I've tried to select a range of PIL formats, the small dataset size means that this diversity will be limited.

## Leaflet licences

The MHRA products site has no terms on reuse or redistribution. Neither the homepage nor the "About this service" page says anything about copyright, licensing or reuse of the documents.

To be safe, I won't upload the documents to my github repo. However, anyone wanting to reproduce the work in this repo can download similar documents for themselves.

# Parsing, cleaning and chunking

See `notebooks/parsing_and_chunking_exploration.ipynb`.
