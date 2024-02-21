import streamlit as st
import os 
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.storage.storage_context import StorageContext
from llama_index.retrievers import QueryFusionRetriever
from llama_index.query_engine import RetrieverQueryEngine, PandasQueryEngine
import nest_asyncio
from llama_index.vector_stores import ChromaVectorStore
from llama_index import StorageContext, load_index_from_storage, VectorStoreIndex, VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.schema import Document, TextNode
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta # to add days or years
import pandas as pd
from llama_index.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
import sys
from llama_index import ServiceContext
sys.path.append("/mount/src/hanwhaqcells/pages/utils")
sys.path.append("utils")
from web_surf import GoogleSearchToolSpec
from chromadb.config import Settings

nest_asyncio.apply()
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://qcells-us-test-openai.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "70d67d8dd17f436b9c1b4e38d2558d50"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ['ACTIVELOOP_TOKEN'] = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwNTIxMjk0MCwiZXhwIjoxNzM2ODM1MzM1fQ.eyJpZCI6Imt5b3VuZ3N1cDg4MDMifQ.KAo14SA3CNMkK68YG9pFiIrShZBqoK9ElOMfyQh8HiBfn9rsEdZneTLQOBQi1kHBjzndbYtOju-FceXx_Rv83A'


st.session_state.embedding = AzureOpenAIEmbeddings(azure_deployment="embedding_model")
st.session_state.llm = AzureChatOpenAI(temperature = 0, deployment_name="test_gpt")
service_context = ServiceContext.from_defaults(llm=st.session_state.llm,embed_model=st.session_state.embedding,)


web_surfer = GoogleSearchToolSpec(key = 'AIzaSyBZaepCCskamC_j3aBLnUNfOTRcpBgNteU',
                        engine = 'd18770dfc867442e9',
                        num = 5)

st.set_page_config(page_title="RAG",  layout="wide",  page_icon="⛅")    

@st.cache_data
def get_UN_data():
    df = pd.read_csv("data/20240207.csv")
    return df

def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs."):
        db2 = chromadb.HttpClient(host='4.242.8.48', port=8000, settings=Settings(chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",chroma_client_auth_credentials="qcells:qcells"))
        # db2 = chromadb.PersistentClient(path="../db/pvmagazine_db")
        chroma_collection = db2.get_or_create_collection("pv_magazine")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store,service_context=service_context, similarity_top_k=10, use_async=True)        
        chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", 
                                        filters = st.session_state.filters,
                                            context_prompt=(
                                                "Hello assistant, we are having a insightful discussion about documents."
                                                "Here are the relevant documents for the context:\n"
                                                "{context_str}"
                                                "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
                                            ),
                                            verbose=False,
        )
        return chat_engine
    
def get_filter_data():
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="release_date_year", operator=FilterOperator.GTE, value=st.session_state.slider[0].year),
            MetadataFilter(key="release_date_year", operator=FilterOperator.LTE, value=st.session_state.slider[1].year),
        ]
    )
    return filters

def reset_conversation():
    st.session_state.filters = get_filter_data()    
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about news!"}]    
    st.session_state.chat_engine = load_data()
    st.session_state.chat_engine.reset()
        
df = get_UN_data()
start_date = dt.date(year=2021,month=1,day=1)-relativedelta(years=1)  #  I need some range in the past
end_date = dt.datetime.now().date()-relativedelta(years=0)

with st.sidebar.form(key='Search'):
    st.session_state.slider = st.slider('New release period', min_value=start_date, value=(start_date ,end_date) ,max_value=end_date, format='YYYY MMM DD', )
    st.form_submit_button(label='Search', on_click = reset_conversation)

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about news!"}]

if "filters" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.filters = get_filter_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.chat_engine = load_data()


st.markdown("# PV News chat powered by LLM")
st.button('Reset Chat', on_click=reset_conversation)

with st.chat_message("assistant"):
    st.write('''☀️ Hanwha QCELLS CTO RAG☀️ 
             \n 👉🏼 Sample query case 1. How is solar panel degrading rate? (기존 ChatGPT)
             \n 👉🏼 Sample query case 2. 한화큐셀 관련된 최신 뉴스 찾아줘 (RAG index)
             \n 👉🏼 Sample query case 3. 브라질의 현재 태양광 발전 용량은 어떻게 되나요? -> 2024년도 기준으로 알려줘 -> 분산형 발전과 중앙형 발전 용량은? (Meta filter RAG index)
             \n 👉🏼 Sample query case 4. 태양광 패널 관련 HOA 규제에 대한 기사 혹은 문서 찾아줘. (Meta filter RAG index)
             ''')
    
if prompt := st.chat_input("Your question", key = 'chat_input_query'): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            if len(response.source_nodes) >= 1:
                for n in response.source_nodes:
                    url = n.metadata['url']
                    st.sidebar.write(url)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history


# st.write(st.session_state)
# deep memory training
# window parsing index vector