import streamlit as st
import os 
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import nest_asyncio
from llama_index import set_global_service_context, ServiceContext
import sys
sys.path.append("./utils")
from web_catch import BeautifulSoupWebReader
from web_surf import GoogleSearchToolSpec
from web_crack import RemoteDepthReader
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, Document
script_dir = os.path.dirname(__file__)
st_abs_file_path = os.path.join(script_dir, "static/")

import fitz
from llama_hub.youtube_transcript import YoutubeTranscriptReader
from llama_index import download_loader
from llama_index.node_parser import SentenceWindowNodeParser
from pathlib import Path
import json


wiki_loader = download_loader("WikipediaReader", custom_path='./wikipedia')
wiki_loader = wiki_loader()
web_cracker = RemoteDepthReader()
web_surfer = GoogleSearchToolSpec(key = 'AIzaSyBZaepCCskamC_j3aBLnUNfOTRcpBgNteU',
                        engine = 'd18770dfc867442e9',
                        num = 5)

web_loader = BeautifulSoupWebReader()
youtube_loader = YoutubeTranscriptReader()

st.set_page_config(page_title="Multi_Docs",  layout="wide", page_icon="📚")
nest_asyncio.apply()
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://qcells-us-test-openai.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "70d67d8dd17f436b9c1b4e38d2558d50"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ['ACTIVELOOP_TOKEN'] = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwNTIxMjk0MCwiZXhwIjoxNzM2ODM1MzM1fQ.eyJpZCI6Imt5b3VuZ3N1cDg4MDMifQ.KAo14SA3CNMkK68YG9pFiIrShZBqoK9ElOMfyQh8HiBfn9rsEdZneTLQOBQi1kHBjzndbYtOju-FceXx_Rv83A'

embedding = AzureOpenAIEmbeddings(azure_deployment="embedding_model")
llm = AzureChatOpenAI(temperature = 0, deployment_name="test_gpt")
service_context = ServiceContext.from_defaults(llm=llm,embed_model=embedding,)
set_global_service_context(service_context)

st.sidebar.header("Parallel Demo")
st.sidebar.write("https://docs.llamaindex.ai/en/stable/index.html")
st.sidebar.write("https://www.youtube.com/watch?v=-uT-9ZBvF44&t=463s")

st.title("Parallel Multi-Document RAG")

if 'analyst_prompt_output' not in st.session_state:
    st.session_state['analyst_prompt_output'] = []

wiki_data = st.text_input("Wiki Keyword(Anchor)")
web_crack_data = st.text_input("Web cracker(reference)")
# web_search_data = st.text_input("Google search(reference)")
web_data = st.text_input("Web Single page (reference)") 
youtube_data = st.text_input("Youtube List(reference)")

if "save_btn" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.save_btn = False

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about news!"}
    ]

if "engineYN" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.engineYN = False


if "external_docs" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.external_docs = []

# @st.cache_resource(show_spinner=True)
def create_db_chat(_docs):
    index_name = GPTVectorStoreIndex.from_documents(_docs, show_progress=True)
    chat_engine = index_name.as_chat_engine(
                        chat_mode="condense_plus_context",
                        context_prompt=(
                            "Hello assistant, we are having a insightful discussion about solar news."
                            "Here are the relevant documents for the context:\n"
                            "{context_str}"
                            "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."),
                        verbose=False)  
    st.session_state.chat_engine = chat_engine
    st.session_state.engineYN = True
    # return chat_engine

def pdf_load_data(multiple_files):
    docs = []    
    if len(multiple_files) > 0:
        for file in multiple_files:
            file.seek(0,0)
            extra_info = {}
            doc = fitz.open(stream=file.read(), filetype="pdf")
            extra_info["total_pages"] = len(doc)
            docs.append([
                    Document(text=page.get_text().encode("utf-8"),extra_info=dict(extra_info,**{"source": f"{page.number+1}"}))
                    for page in doc])
        return docs[0]

def flatten(xss):
    return [x for xs in xss for x in xs]

# @st.cache_resource(show_spinner=True)
def load_pptx(multiple_files):
    PptxReader = download_loader("PptxReader")
    loader = PptxReader()
    documents = loader.load_data(file=multiple_files[0])
    return documents

# @st.cache_resource(show_spinner=True)
def load_web(multiple_files):
    web_crack_documents = web_cracker.load_data(url=multiple_files)
    web_crack_documents = web_loader.load_data(urls=web_crack_documents[:10])
    return web_crack_documents

def change_name():
    with st.spinner(text="Creating response"):
        st.session_state.analyst_prompt_output = []
        for i in st.session_state.engines.items():
            response = i[1].query(st.session_state.query_text)
            st.session_state.analyst_prompt_output.append(response.response)

def reset_conversation():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about external data source!"}]
    chat_engine = create_db_chat()        
    return chat_engine

def cache_clear():
    st.cache_resource.clear()
    st.session_state.clear()

multiple_files = st.file_uploader("Drop files(reference):", accept_multiple_files=True, key="file_uploader")
if multiple_files:
    if st.session_state.engineYN == False:
        st.write('pdf read!')
        st.session_state.pdf_documents= pdf_load_data(multiple_files)
        # pdf_documents = load_pptx(multiple_files)
        for i in st.session_state.pdf_documents:
            st.session_state.external_docs.append(i)  

if wiki_data:
    wiki_documents = wiki_loader.load_data(pages=wiki_data)
    st.session_state.external_docs.append(wiki_documents[0])

if web_crack_data:
    if st.session_state.engineYN == False:
        web_crack_documents = load_web(web_crack_data)
        for i in web_crack_documents:
            st.session_state.external_docs.append(i)  
    # web_crack_documents = web_cracker.load_data(url=web_crack_data)
    # web_crack_documents = web_loader.load_data(urls=web_crack_documents)
    # st.write(web_crack_documents)
    # external_docs.append(web_crack_documents)

# if web_search_data:
#     web_data_documents = web_surfer.google_search(web_search_data)
#     links = [i['link'] for i in json.loads(web_data_documents[0].to_dict()['text'])['items']] 
#     st.write(links)
    # external_docs.append(web_data_documents[0])

if web_data:
    if st.session_state.engineYN == False:
        web_data_documents = web_loader.load_data(urls=[web_data])
        st.session_state.external_docs.append(web_data_documents[0])

if youtube_data:
    if st.session_state.engineYN == False:
        youtube_data_documents = youtube_loader.load_data(ytlinks=[youtube_data], languages=['en', 'ko'])
        st.session_state.external_docs.append(youtube_data_documents[0])

col1, col2, col3 = st.columns([1, 1, 13])
with col1:
    btn = st.button("RESET", on_click = cache_clear)  
with col2:
    btn2 = st.button("SAVE", on_click = create_db_chat, args=[st.session_state.external_docs])  
    if btn2:
        st.session_state.save_btn = True

st.write('total docs count: ', len(flatten(st.session_state.external_docs)))
if len(st.session_state.external_docs) > 0:    
    if st.session_state.save_btn==True:
        if prompt := st.chat_input("Your question", key = 'chat_input_query'): # Prompt for user input and save to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    
                    response = st.session_state.chat_engine.chat(prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message) # Add response to message history

# pdf table parsing
# deep memory training
# window parsing index vector

