import json
import re
from statistics import mean
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
import numpy as np

class triage_eval:
    """
    Evaluador similar a RAGAS para probar el funcionamiento (Ollama)
    Calcula:
      - answer_correctness
      - answer_relevancy
      - faithfulness
      - context_precision
      - context_recall
    """

    def __init__(self, model="llama3.2:3b", temperature=0.2, num_predict=512, verbose=False, reasoning=None):
        self.model = ChatOllama(
            model=model, 
            validate_model_on_init=True,      # Verificar que el modelo está en ollama
            num_predict=num_predict,          # Numero de tokens a generar
            reasoning=reasoning,                   # caja de rasonamiento del LLm
            format="json",                    # Para que la respuesta sea json
            temperature=temperature)          # Modelo
        self.verbose = verbose                # Para debugging
        self.embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

    # ---------------- METRIC PROMPTS ---------------- #

    def answer_correctness(self, question, answer, ground_truth):
        try:
            prompt = f"""

    Classify statements from the answer and ground truth into the categories TP, FP, and FN.

    “TP”: A statement that appears in the answer and appears in the ground truth
    “FP”: A statement that appears in the answer but does not appear in the ground truth.
    “FN”: A statement that appears in the ground truth but not in the answer.

    Each statement can only belong to one of the categories. TP statements must match the ground truth directly, not by assumption.

    Expected JSON format:
    {{
        "TP": ["...", ...]
        "FP": ["...", ...]
        "FN": ["...", ...]
    }}

    Now process the following inputs:
    question: {question}
    answer: {answer}
    ground truth: {ground_truth}

                """
            results = self.model.invoke(prompt).content
            results = json.loads(results)
            # print("###############################################################")
            # print(results)
            # print("###############################################################")
            f1_score = len(results['TP']) / (len(results['TP']) + (0.5 * (len(results['FP']) + len(results['FN']))))
            emb_ans = self.embedding_model.embed_query(answer)
            emb_gt = self.embedding_model.embed_query(ground_truth)
            correctness = (f1_score * 0.75) + (self._cosine_similarity(emb_ans, emb_gt) * 0.25)

            print(f"score correctness: {correctness:.3f}")
            return (correctness, 1)
        except:
            return (0,0)    # Segunda posición 0 si falla algo

####################################################################################   
    def questions(self, answer):
        prompt = f"""
You are an assistant that generates evaluation questions based ONLY on the provided answer. Generate exactly 3 short questions. Questions must be directly derived from the content. Do NOT create more or fewer than 3 questions.

Expected JSON format:

{{
  "questions": [
    "<question_1>",
    "<question_2>",
    "<question_3>"
  ]
}}

Now process the following answer:

answer: {answer}

        """
        return self.model.invoke(prompt).content
####################################################################################   
      
    def _cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
####################################################################################     
    def answer_relevancy(self, answer):
        try:
            question_list = json.loads(self.questions(answer))
            emb_ans = self.embedding_model.embed_query(answer)
            sim_list = []
            for i in question_list['questions']:
                emb_quest = self.embedding_model.embed_query(i)
                sim = self._cosine_similarity(emb_ans, emb_quest)
                sim_list.append(sim)
            
            answer_rev = sum(sim_list)/len(sim_list)

            #print(f"score answer_relevancy: {answer_rev:.3f}")
            return (answer_rev, 1)
        except:
            return (0, 0) # Segunda posición 0 si falla algo
    

####################################################################################   

    def statements(self, question, answer):
        prompt = f"""
Given a question and answer, create one or more statements from each sentence in the given answer. Break down each sentence into one or more fully understandable statements.Ensure that no pronouns are used in any statement. Do NOT add, infer, or omit information. Format the outputs in JSON. DO NOT GENERATE MORE THAN 10 STATEMENTS

Expected JSON format:
{{
  "statements": [
    "<statement 1>",
    "<statement 2>",
    ...
  ]
}}

Now process the following input:

question: {question}
answer: {answer}
        """
        return self.model.invoke(prompt).content
    
####################################################################################   
    def faithfulness(self, question, context, answer):
        try:
            joined_context = "\n".join(context) if isinstance(context, list) else context
            statement_list = json.loads(self.statements(question, answer))
            response_list = []
            for state in statement_list['statements']:
                
                statement = state

                prompt = f"""
    You are a factuality evaluator.

    Consider the given context and following statement, then determine whether the statement is supported by the information
    present in the context. Provide a brief explanation for the statement before arriving at the verdict (Yes/No).
    Provide a final verdict for the statement in order at the end in the given format. Do not deviate from the specified
    format. Do not assume or infer information not present in the context.

    Expected JSON format:
    {{
    "label": "yes" | "no",
    "explanation": "<brief reasoning>"
    }}

    Now process the following input:

    context: {joined_context}
    statement: {statement}

                """
                response = self.model.invoke(prompt).content
                response_list.append(response)

            count = 0
            for i in response_list:
                if json.loads(i)['label'] == "yes":
                    count+=1

            faith = count/len(response_list)

            #print(f"score faithfulness: {faith:.3f}")
            return (faith, 1)
        except:
            return (0, 0)   # Segunda posición 0 si falla algo

#################################################################################### 
#   
    def context_precision(self, question, answer, context):
        try:
            count = 0
            for ctx in context:

                prompt = f"""
    Given a question, an answer, and a context, determine whether the context was actually used to produce the answer.

    1: The answer depends on information contained in the context.
    0: The answer does not rely on the context (e.g., it is generic, irrelevant, or guessable without context).

    Use only information explicitly present in the context. Do not infer or assume the model used context unless the answer clearly reflects it. Output must follow the format exactly.
                
    Expected JSON format:

    {{
    "useful": 1 | 0
    "reason": "<brief reasoning>"
    }}       

    Now process the following input:

    question: {question}
    answer: {answer}
    context: {ctx}
                """
                result = json.loads(self.model.invoke(prompt).content)
                if result['useful'] == 1:
                    count +=1
            context_pre = count/len(context)

            #print(f"score context_precision: {context_pre}")
            return (context_pre, 1)
        except:
            return (0, 0)   # Segunda posición 0 si falla algo

    
####################################################################################   

    def context_recall(self, answer, context):
        try:
            joined_context = "\n".join(context) if isinstance(context, list) else context

            prompt = f"""
    Given a context and an answer, determine whether each sentence in the answer is supported by / attributable to the context.
    Classify:

    1: The sentence’s meaning is directly supported by the context.
    0: The sentence is not supported, contradicted, or not present in the context.

    Use only information explicitly found in the context. Do not infer, paraphrase beyond clear semantic equivalence, or introduce external knowledge.Output must strictly follow the format below.
                
    Expected JSON format:
    {{
        "classifications": [
            {{
                "sentence": "...",
                "supported": 1
            }},
            {{
                "sentence": "...",
                "supported": 0
            }}
        ]
    }}   

    ------------------------------------
    FEW-SHOT EXAMPLES
    ------------------------------------

    Example 1
    Context:
    "The hospital uses a five-level triage system. Level 1 is the most urgent."

    Answer:
    1. "The hospital has a five-level triage system. Level 1 is the least urgent."

    Expected output:
    {{
        "classifications": [
            {{ 
                "sentence": "The hospital has a five-level triage system.", 
                "supported": 1 
            }},
            {{ 
                "sentence": "Level 1 is the least urgent.", 
                "supported": 0 
            }}
        ]
    }}

    Example 2
    Context:
    "The triage protocol states that vital signs must be measured upon patient arrival. 
    After this, the nurse assigns an acuity level based on the ESI scale. 
    Only patients categorized as Level 1 receive immediate life-saving interventions."

    Answer:
    "Vital signs are taken as soon as the patient arrives. The nurse assigns acuity levels according to the ESI scale. All patients receive immediate life-saving interventions."

    Expected output:
    {{
        "classifications": [
            {{ 
                "sentence": "Vital signs are taken as soon as the patient arrives.", 
                "supported": 1 
            }},
            {{ 
                "sentence": "The nurse assigns acuity levels according to the ESI scale.", 
                "supported": 1 
            }},
            {{ 
                "sentence": "All patients receive immediate life-saving interventions.", 
                "supported": 0 
            }}
        ]
    }}


    Now process the following input:

    answer: {answer}
    contexts: {joined_context}

                """
            # ESTE ESTA CON PROBLEMAS DEBIDO AL PROMPT REVISARRRRRRRRR
            bruh = self.model.invoke(prompt).content
            classification = json.loads(bruh)

            count = 0
            for i in classification['classifications']:
                count += int(i['supported'])
            context_re = count/len(classification['classifications'])

            #print(f"score context_recall: {context_re:.3f}")
            return (context_re, 1)
        except:
            return (0, 0)       # segunda posición 0 si falla algo

    # ---------------- BATCH EVALUATION ---------------- #

    def evaluate(self, examples):
        """
        examples: list of dicts with keys:
            question, context, answer, ground_truth, (optional) reference_contexts
        """
        results = []
        for ex in examples:
            if self.verbose:
                print(f"Evaluating question: {ex['question'][:60]}...")
            metrics = {
                #"answer_correctness": self.answer_correctness(ex["question"], ex["answer"], ex["ground_truth"]),
                #"answer_relevancy": self.answer_relevancy(ex["question"], ex["answer"]),
                "faithfulness": self.faithfulness(ex["context"], ex["answer"]),
                #"context_precision": self.context_precision(ex["question"], ex["context"]),
            }
            if "reference_contexts" in ex:
                metrics["context_recall"] = self.context_recall(
                    ex["question"], ex["context"], ex["reference_contexts"]
                )
            results.append(metrics)

        # Compute averages
        summary = {k: mean([r[k] for r in results if r[k] is not None]) for k in results[0].keys()}
        return summary, results
    
    def test_call(self, question, answer, context, ground_truth):
        self.faithfulness(question, context, answer)
        self.answer_relevancy(answer)
        self.answer_correctness(question, answer, ground_truth)
        self.context_recall(answer, context)
        self.context_precision(question, answer, context)

    def evaluate_metrics(self, dataset):

        results = []
        errors_lst = []
        for example in dataset:
            if self.verbose:
                print(f"Evaluando caso: {example['user_input'][:10]}")

            correctness = self.answer_correctness(example['user_input'], example['response'], example['reference'])
            ans_rev = self.answer_relevancy(example['response'])
            faithful = self.faithfulness(example['user_input'], example['retrieved_contexts'], example['response'])
            con_rec = self.context_recall(example['response'], example['retrieved_contexts'])
            con_prc = self.context_precision(example['user_input'], example['response'], example['retrieved_contexts'])

            metrics = {
                "answer_correctness" : correctness[0],
                "answer_revelancy" : ans_rev[0],
                "faithfulness" : faithful[0],
                "context_recall" :  con_rec[0],
                "context_precision" : con_prc[0]
            }
            errors = {
                "answer_correctness" : correctness[1],
                "answer_revelancy" : ans_rev[1],
                "faithfulness" : faithful[1],
                "context_recall" :  con_rec[1],
                "context_precision" : con_prc[1]
            }
            results.append(metrics)
            errors_lst.append(errors)           
        summary_results = {k: mean([r[k] for r in results if r[k] is not None]) for k in results[0].keys()}
        summary_errors = {k: mean([r[k] for r in errors_lst if r[k] is not None]) for k in errors_lst[0].keys()}
        return summary_errors, summary_results

    