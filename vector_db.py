# PARA CREAR LA VECTOR DATABASE PARA EL RAG

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from tqdm import tqdm


#Cargar la data ya procesada
data_loaded = pd.read_csv("MIETIC_train.csv")
# VOY A SAMPLEAR UNA PARTE PARA PROBAR QUE ESTO FUNCIONA
#data_loaded = data_loaded.sample(frac=0.5, random_state=42)
#Cargar el modelo de embeddings ya instalado
# Puse uno por default (COMPROBAR SI HAY OTROS MEJORES)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
print("hola")
#Lugar donde guardaré la database
db_loc = "./chrome_langchain_db"

#Mirar si la database ya existía antes para saber si hay 
#que crearla
add_doc = not os.path.exists(db_loc)

if add_doc:
    documents = []
    ids = []

    for i, row in data_loaded.iterrows():
        #print(row['instruction'])
        #print(i)
        document = Document(
            # TODo EL CONTENIDO QUE SE VA A VECTORIZAR
            # IMPORTANTE: QUE QUIERO VECTORIZAR??????
            # LO DE ABAJO POR EL MOMENTO LO DEJARÉ ASI:
            
            # REVISAR TIPOS DE DATOS DE CADA COLUMNA
            # 'Case: \n' + str(row['input']) + ' ' + str(row['output'])
            # Tengo que cambiar como creo el RAG?
            page_content=f"Case: {str(row['input'])} \nRecommended actions: {str(row['output'])}",
            # Metadata es información adicional que se añade pero no se vectoriza
            metadata = {
                "id" : str(i),
                "instruction" : str(row["instruction"])
                # Que más debería incluir en la metadata?
                },
            #Index
            id = str(i)
        )

        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="triage_cases",
    # Esto es recomendado para no recrear siempre la DB
    persist_directory=db_loc,
    embedding_function= embeddings
)

def add_documents_in_batches(vector_store, documents, batch_size=5000):
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vector_store.add_documents(
            documents=batch,
            ids=batch_ids
        )


if add_doc:
    add_documents_in_batches(vector_store, documents)
    # vector_store.add_documents(documents=documents, ids = ids)    # ESTA LÍNEA NO GENERA BIEN EL RAG

# CUANTOS DOCS RELACIONADOS VA A AGARRAR
retriever = vector_store.as_retriever(
    search_kwargs={"k":3}
)