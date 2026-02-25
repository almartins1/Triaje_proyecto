from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_db import retriever
import re
import pandas as pd
from bert_score import score
from datasets import Dataset
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_correctness, answer_relevancy, context_precision, context_recall
import os
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, ChatOllama, OllamaEmbeddings
from phoenix.otel import register

from deepeval.test_case import LLMTestCase

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import triage_class

# Set environment (Esto no creo que sea necesario)
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
os.environ["OPENAI_API_KEY"] = "none"


tracer_provider = register(
    project_name="triage_proyect", # sets a project name for spans (proyecto en arize)
    batch=True, # uses a batch span processor
    auto_instrument=True, # uses all installed OpenInference instrumentors
)
LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)

# Base de datos a probar para el RAG
data_test = pd.read_csv("MIETIC_test.csv")
data_handbook = pd.read_csv("handbook_cases.csv")

# # Esta línea sirve para poner el modelo local de Ollama
# # Para experimentar poner varios
# model = OllamaLLM(model="deepseek-r1:32b")

# Ponemos el modelo de para metricas
LLM_NAME = 'llama3:8b' #llama3:8b, deepseek-r1:32b
EMB_NAME = 'mxbai-embed-large'
ragas_llm = LangchainLLMWrapper(langchain_llm=ChatOllama(model=LLM_NAME, base_url="http://localhost:11434", temperature=0,
                                                          format="json"))
#ragas_embed = OllamaEmbeddings(model=EMB_NAME)
embed_model = OllamaEmbeddings(model=EMB_NAME)
ragas_embed = LangchainEmbeddingsWrapper(embed_model)

# Assign local models to all metrics
for metric in [answer_correctness, answer_relevancy, faithfulness, context_precision, context_recall]:
    metric.llm = ragas_llm
    metric.embeddings = ragas_embed


# # Este template es lo que quiero que haga el modelo
# template = """
# You are an experienced triage nurse in the Emergency Department. A patient just arrived at ED. 
# The user will provide you with the description of this patient's situation.

# Here are some relevant triage cases: {case}

# And here is the case of the new patient: {new}

# Instruction:
# Using the patient’s description provided:
# 1. Analyse patients condition.
# 2. Predict the exact number of resources the patient is likely to require during their ED visit to reach a 
# disposition.
# 3. Explain your reasoning briefly (less than 100 words), listing the anticipated resources.
# 4. Only include those resources most needed.
# """

# prompt = ChatPromptTemplate.from_template(template)

# # Primero se pasa todo al prompt (en este caso "case" y "new")
# # lo resultante de eso se pasa al modelo.
# chain = prompt | model          

# Caso nuevo (test de esi handbook)
question = """A 28-year-old male presents to the ED requesting to be checked. He has a severe shellfish allergy and
 mistakenly ate a dip that contained shrimp. He immediately felt his throat start to close so he used his EpiPen®. 
 He tells you he feels okay. No wheezes or rash noted. VS: BP 136/84, HR 108, RR 20, SpO2 97%, temperature (T) 97° F."""
# Referencia de la respuesta para metricas
reference = """ESI level 2: high-risk situation for
allergic reaction. The patient has used his
EpiPen but still requires additional medications
and close monitoring. """
# cases = retriever.invoke(question)

def think_remover(text: str) -> str:
    # Aqui quito el think del llm
    return re.sub(r'(?is)<think>.*?</think>', '', text).strip()

def triage_call(model_llm, question_llm, reference):
    model_run = OllamaLLM(model=model_llm)
    # template_llm = """
    # You are an experienced triage nurse in the Emergency Department. A patient just arrived at ED. 
    # The user will provide you with the description of this patient's situation.

    # Here are some relevant triage cases: {case}

    # And here is the case of the new patient: {new}

    # Instruction:
    # Using the patient’s description provided:
    # 1. Analyse patients condition.
    # 2. Predict the exact number of resources the patient is likely to require during their ED visit to reach a 
    # disposition.
    # 3. Explain your reasoning briefly, listing the anticipated resources.
    # 4. Only include those resources most needed.
    # 5. The reasoning MUST be fewer than 100 words.
    # """
    template_llm = """
    You are an experienced emergency department triage nurse with expert knowledge of patient assessment and ED workflows.

    You will be provided with:
    - Relevant historical triage cases retrieved from a knowledge base.
    - A description of a newly arrived patient in the Emergency Department.

    --------------------------------
    INPUT
    --------------------------------

    Relevant triage cases:
    {case}

    New patient case:
    {new}

    --------------------------------
    TASK
    --------------------------------

    Using ONLY the information provided:

    1. Assess whether the patient requires immediate life-saving intervention by evaluating:
    - Airway, breathing, circulation,
    - Severe physiological instability.
    Clearly state if immediate life-saving care is required and why.

    2. If no immediate life-saving intervention is required, evaluate whether the patient presents high-risk clinical features, including:
    - High-risk clinical situations,
    - New onset confusion, lethargy, or disorientation,
    - Severe pain or distress.
    Reference relevant patterns from the retrieved cases when applicable.

    3. Independently of the above assessments, predict the exact number of ED resources required to reach disposition:
    - Predict the exact number of ED resources required to reach disposition.
    - Always list only the most essential resources needed.
    - List only the most essential resources needed.

    4. Provide a concise clinical justification grounded in the patient’s condition and the retrieved cases.
    
    5. The total reasoning MUST be fewer than 100 words.

    Do NOT:
    - Explicitly assign an ESI level.
    - Add speculative or non-essential resources.
    - Use external medical knowledge beyond the provided inputs.

    --------------------------------
    OUTPUT FORMAT
    --------------------------------

    Immediate life-saving intervention required:
    <Yes / No>

    High-risk (ESI Level 2) indicators present:
    <Yes / No>

    Predicted essential resources:
    - <resource 1>
    - <resource 2>
    - ...

    Total number of resources: <integer>

    Reasoning (≤100 words):
    <concise, clinically grounded explanation>
    """
    prompt_llm = ChatPromptTemplate.from_template(template_llm)
    chain_llm = prompt_llm | model_run  
    cases_llm = retriever.invoke(question_llm)
    response = chain_llm.invoke({"case": cases_llm, "new": question_llm})
    response = think_remover(response)

    contexts_llm = [i.page_content for i in cases_llm]

    return [question_llm, contexts_llm, response, reference]

def verification_call(model_llm, patient_case, resources):
    model_run = OllamaLLM(model=model_llm)
    prompt = """
    You are an experienced emergency department triage nurse with expert knowledge of the Emergency Severity Index (ESI).
    
    You must strictly follow the ESI decision rules described below. Do NOT invent new criteria or use other triage scales.

    --------------------------------
    ESI SCALE (AUTHORITATIVE RULES)
    --------------------------------

    ESI Level 1:
    - Patient requires immediate life-saving intervention.
    - Examples: cardiac arrest, respiratory failure, severe shock, unresponsive patient.
    - Resource count is NOT considered.

    ESI Level 2:
    - Patient is high risk, confused/lethargic/disoriented, or in severe pain or distress.
    - No immediate life-saving intervention required, but condition is potentially unstable.
    - Resource count is NOT the deciding factor.

    ESI Level 3:
    - Patient is stable but is expected to require TWO OR MORE ED resources.
    - Examples of resources: labs, imaging, IV fluids, medications, specialty consults.
    - Vital signs should be within acceptable limits.

    ESI Level 4:
    - Patient is stable and expected to require ONE ED resource.

    ESI Level 5:
    - Patient is stable and expected to require NO ED resources beyond examination.

    --------------------------------
    INPUT
    --------------------------------

    You will be given:
    - A description of a patient’s clinical condition.
    - An analysis of the patient condition with list of anticipated resources predicted for this patient during their ED visit.

    Your task is to infer the most appropriate ESI level (1–5) based on:
    - The patient’s clinical presentation.
    - The urgency and severity of symptoms.
    - The number and type of resources required, following standard ESI guidelines.

    Patient description:
    {patient_case}

    Predicted resources:
    {resources}

    --------------------------------
    TASK
    --------------------------------

    Instructions:
    1. Assess whether ESI 1 or ESI 2 applies based on clinical severity.
    2. If not, determine ESI strictly from the NUMBER of predicted resources.
    3. Infer the most appropriate ESI level (1–5).
    4. Provide a concise justification grounded ONLY in the ESI rules above.
    5. The reasoning MUST be fewer than 100 words.

    --------------------------------
    OUTPUT FORMAT
    --------------------------------
    Output format:
    - Reasoning (free text, concise but complete)
    - Inferred ESI Level: X
    """

    prompt_llm = ChatPromptTemplate.from_template(prompt)
    chain_llm = prompt_llm | model_run
    response_esi_level = chain_llm.invoke({"patient_case": patient_case, "resources": resources})
    response_esi_level = think_remover(response_esi_level)

    return response_esi_level

# first_row = data_handbook.iloc[0]
# quest = first_row["cases"]
# answ = first_row["references"]

# results = triage_call("llama3.2:3b", question_llm=quest, reference=answ)

evaluator_triage = triage_class.triage_eval()
#evaluator_triage.test_call(question=results[0], answer=results[2], context=results[1], ground_truth=results[3])
print("Chao")
# print(results[1])

dataset_test = []   # Aqui creo un dataset para probar respuestas dadas por rag
deepeval_test = []
bert_responses = []
bert_references = []

excel_output = []

for index, row in data_test.iterrows(): #cambiar por: data_test.iterrows()
    hb_case = row['input']          #row['cases'](ESTO PARA HANDBOOK)  row['input'](ESTO PARA TEST)
    hb_reference = row['output']    #row['references'](ESTO PARA HANDBOOK) row['output'](ESTO PARA TEST)

    exe_model = "deepseek-r1:14b"  # deepseek-r1:32b llama3:8b deepseek-r1:14b

    results = triage_call(exe_model, question_llm=hb_case, reference=hb_reference)
    #esi_level = verification_call(exe_model, patient_case=hb_case, resources=results[2])
    bert_responses.append(results[2])
    bert_references.append(results[3])
    sample_rag = {
        'user_input':results[0],              #User input
        'retrieved_contexts':results[1],      #retrieved_contexts
        'response':results[2],                #response
        'reference':results[3]                #reference
    }
    # sample_rag_excel = {
    #     'user_input':results[0],              #User input
    #     'retrieved_contexts':results[1],      #retrieved_contexts
    #     'response':results[2],                #response
    #     'reference':results[3],               #reference
    #     'verification':esi_level
    # }

    dataset_test.append(sample_rag)
    # excel_output.append(sample_rag_excel)

# OUTPUT PARA PROBAR CON MÉDICO
# df_excel = pd.DataFrame(excel_output)
# df_excel.to_csv("output.csv", sep=",", index=False)

print("output")
# DESCOMENTAR TODOo ESTO AL MOMENTO DE REALIZAR EVAL
# Clase personalizada para realizar ragas
# errors_lst, metrics_lst = evaluator_triage.evaluate_metrics(dataset_test)

evaluation_dataset = EvaluationDataset.from_list(dataset_test) 

# Ragas importado
score_ragas=evaluate(evaluation_dataset, metrics=[answer_correctness, answer_relevancy, faithfulness, 
                                                  context_precision, context_recall])
print(f"Puntaje ragas: {score_ragas}")

# print("#####################################################")
# print("Puntaje NO RAGAS")
# print(f"ERRORES: {errors_lst}")
# print(f"METRICAS: {metrics_lst}")


# print("#####################################################")

P, R, F1 = score(bert_responses, bert_references, lang="en", verbose=True)



print(f"Precision: {P.mean().item():.4f}")
print(f"Recall:    {R.mean().item():.4f}")
print(f"F1:        {F1.mean().item():.4f}")

# Organizar los experimentos
# Cambiar el prompt inicial (o agregar otro agente)

#print(geval_metrics)

# Exportar puntajes
puntajes = [
    {
        "Model":exe_model,
        "Ragas":score_ragas,
        "Bert_Precision":P.mean().item(),
        "Bert_Recall":R.mean().item(),
        "Bert_F1":F1.mean().item()
    }
]

df_puntajes = pd.DataFrame(puntajes)
df_puntajes.to_csv("output_puntajes.csv", sep=",", index=False)