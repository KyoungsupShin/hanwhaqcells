import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import datetime as dt
from dateutil.relativedelta import relativedelta # to add days or years
import pandas as pd
import os 
import wikipedia
from llama_index.core import download_loader
from random import randint
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, set_global_service_context, VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.memory import ChatMessageHistory
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
wiki_loader = download_loader("WikipediaReader", custom_path='./wikipedia')
wiki_loader = wiki_loader()

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://qcells-us-test-openai.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "70d67d8dd17f436b9c1b4e38d2558d50"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ['ACTIVELOOP_TOKEN'] = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwNTIxMjk0MCwiZXhwIjoxNzM2ODM1MzM1fQ.eyJpZCI6Imt5b3VuZ3N1cDg4MDMifQ.KAo14SA3CNMkK68YG9pFiIrShZBqoK9ElOMfyQh8HiBfn9rsEdZneTLQOBQi1kHBjzndbYtOju-FceXx_Rv83A'

st.session_state.embedding = AzureOpenAIEmbeddings(azure_deployment="embedding_model")
st.session_state.llm = AzureChatOpenAI(temperature = 0, deployment_name="test_gpt")

service_context = ServiceContext.from_defaults(llm=st.session_state.llm,embed_model=st.session_state.embedding,)


if "company_desc" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.company_desc = wikipedia.summary("q_cells", sentences=5)

if "department_desc" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.department_desc = wikipedia.summary("Chief_technology_officer", sentences=5)

template = f"""
You are an AI agent currently employed at "Hanwha Solution Corporation". 
{st.session_state.company_desc}

Your department at Hanwha Solution is the CTO, and the meaning of CTO is as follows.
{st.session_state.department_desc}

Positions explanations at Hanwha Solution of CTO department
- Data Engineers : Architect and orchestrate data infrastructure, implementing scalable data ingestion, storage, and processing systems using ETL methodologies, big data technologies, and cloud platforms, ensuring high data quality and integrity for Hanwha QCELLS' diverse analytical needs.
- Reinforcement Machine Learning Engineers : Design and refine state-of-the-art reinforcement learning models, employing advanced algorithmic solutions for optimizing predictive analytics and decision-making processes in renewable energy systems and smart grid technologies.
- Time Series Machine Learning Engineers at Hanwha QCELLS: Specialize in statistical analysis and predictive modeling of sequential data, utilizing time series algorithms to forecast trends and anomalies, enhancing operational intelligence in energy production and market dynamics.
- Image Machine Learning Engineers : Develop and fine-tune sophisticated computer vision algorithms, leveraging deep learning and neural networks for image recognition and analysis, contributing to technological advancements in photovoltaic cell inspection and automated quality control.
- cloud service engineers : Engineer robust, scalable cloud architectures, focusing on cloud-native solutions, containerization, and orchestration technologies to facilitate agile development, deployment, and management of Hanwha QCELLS' cloud-based applications and services.
- web service developers : Craft dynamic and responsive web applications, utilizing modern web frameworks and front-end technologies, emphasizing user experience (UX) design principles, to provide intuitive interfaces for Hanwha QCELLS' digital platforms.
- Sales Managers : Devise and execute comprehensive sales strategies, leveraging market analysis and customer relationship management (CRM) techniques to expand market penetration and revenue growth in the renewable energy and chemical sectors.
- Marketing Managers : Lead integrated marketing campaigns, harnessing digital marketing, brand management, and market research methodologies, to amplify Hanwha QCELLS' presence in the global sustainable energy and materials market.
- Legal Managers : Oversee legal compliance and risk management, providing expert legal counsel on corporate law, intellectual property, and regulatory matters, aligning legal strategies with Hanwha QCELLS' business objectives in energy, chemicals, and real estate.
- Chemical Researchers :  Conduct cutting-edge research in chemical engineering and materials science, focusing on innovative synthesis, characterization, and application of novel compounds, driving forward Hanwha QCELLS' commitment to sustainable and advanced chemical solutions.


When I provide an news article, you evaluate the direct relevance or interest for each position and provide a Relevance & interest score and summary about reason of scores (if there direct relevance, just say "No relevance.")
If there is no direct relevance or interest between the article and a specific position, the Relevance & interest score will be close to 0. Conversely, score will be close to 1. 
"""

jsonfy_txt_business = '''
please, evaluate relevance and interest between hanwha QCELLS business and the article. 
    - "Relevance score": from 0 to 1 as float type
    - "Summary": summarize about reasons of scores. if no relevance, just say no relevance.
'''

jsonfy_txt = '''
Evaluate each positions :
1.Relevance score : 0 - 1 as float
2.Interest score : 0 - 1 as float
3.Summary : briefly summarize
'''

output_template = '''
                    extract from given text.
                    "position names": ["Relevance score", "Interest score", "summary"]
                    text: ```{text}```
                    {format_instructions}'''

output_template2="""
                extract from given text.
                "company name": ["Relevance score", "Summary"]
                text: ```{text}```
                {format_instructions}
"""

schema = ResponseSchema(name='output', description="json object, key=Position name, value= [Relevance score, Interest score, summary]")
output_parser = StructuredOutputParser.from_response_schemas([schema])
format_instructions = output_parser.get_format_instructions()
prompt_output = ChatPromptTemplate.from_template(output_template)

schema2 = ResponseSchema(name='output', description="list object, value= [Relevance score, summary]")
output_parser2 = StructuredOutputParser.from_response_schemas([schema2])
format_instructions2 = output_parser2.get_format_instructions()
prompt_output2 = ChatPromptTemplate.from_template(output_template2)


def get_json_parse(texts):
    messages = prompt_output.format_messages(text=texts, format_instructions=format_instructions)
    response = st.session_state.llm(messages)
    output = output_parser.parse(response.content)
    return output['output']

def get_json_parse2(texts):
    messages = prompt_output2.format_messages(text=texts, format_instructions=format_instructions2)
    response = st.session_state.llm(messages)
    output = output_parser2.parse(response.content)
    return output['output']

if "chat_history" not in st.session_state.keys(): # Initialize the chat mess        ge history
    st.session_state.chat_history = ChatMessageHistory()
    st.session_state.chat_history.add_ai_message(template) # assign job to evaluate job relavance
    st.session_state.chat_history.add_ai_message(jsonfy_txt) # formatting

def org_news_to_summary(org_news_txt):
    st.session_state.chat_history = ChatMessageHistory()
    st.session_state.chat_history.add_ai_message(template) # assign job to evaluate job relavance
    st.session_state.chat_history.add_ai_message(jsonfy_txt) # formatting
    st.session_state.chat_history.add_user_message(org_news_txt) # article data
    org_news_txt_summary = st.session_state.llm(st.session_state.chat_history.messages)
    scores = get_json_parse(org_news_txt_summary.content) #output parse

    st.session_state.chat_history.add_user_message(jsonfy_txt_business) # assign job to evaluate business relavance
    org_news_txt_summary2 = st.session_state.llm(st.session_state.chat_history.messages)
    scores2 = get_json_parse2(org_news_txt_summary2.content) #output parse
    return org_news_txt_summary.content, scores, scores2




if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about news!"}]
    
if "selection" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.selection = pd.DataFrame()

if "firstChatYN" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.firstChatYN = True

if "AnalyYN" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.AnalyYN = False

if "origin_text" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.origin_text = ''

if "selected_idx" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.selected_idx = []

if "analysis_output" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.analysis_output1 = None

if "analysis_output2" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.analysis_output2 = None

if "analysis_output3" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.analysis_output3 = pd.DataFrame()

@st.cache_data
def get_UN_data():
    df = pd.read_csv("./data/20240207.csv")
    return df

@st.cache_resource(show_spinner=True)
def load_chat_engine():
    with st.spinner(text="Loading and indexing the Streamlit docs."):
        db2 = chromadb.PersistentClient(path="../db/pvmagazine_db")
        chroma_collection = db2.get_or_create_collection("pv_magazine")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store,service_context=service_context, similarity_top_k=10, use_async=True)        
        chat_engine = index.as_chat_engine(chat_mode="condense_plus_context",
                                            # memory=memory,
                                            context_prompt=(
                                                "Hello assistant, we are having a insightful discussion about documents."
                                                "Here are the relevant documents for the context:\n"
                                                "{context_str}"
                                                "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
                                            ),
                                            verbose=False
        )        
        return chat_engine


def search_dataframe(df:pd.DataFrame, column:str, search_str:str) -> pd.DataFrame:
    results = df.loc[df[column].str.contains(search_str, case=False)]
    return results

def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "ID", False)    
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        disabled=df.columns,
        column_order=("ID","news_national_type","release_date", "title"),
        on_change=reset_dataframe_selector
    )
    selected_rows = edited_df[edited_df.ID]
    if len(selected_rows) >= 1:     
        st.session_state.selection = selected_rows[-1:]
    else:
        pass

def reset_dataframe_selector():
    st.session_state.AnalyYN = False

def reset_conversation():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about news!"}]
    st.session_state.chat_engine.reset()
    st.session_state.firstChatYN = True

def get_semantic_analysis(texts):
    _, st.session_state.analysis_output1, st.session_state.analysis_output2 = org_news_to_summary(texts)
    df = pd.DataFrame.from_dict(st.session_state.analysis_output1, orient='index', columns=['Relevance_Score', 'Interesting_Score', 'Summary'])
    df.index.name = 'Job_Title'
    df.reset_index(inplace=True)
    st.session_state.analysis_output3 = df
    

st.set_page_config(layout="wide", page_title="NEW ANALYSIS", page_icon="📰")
st.markdown("# News Analysis powered by LLM")
st.sidebar.header("RAG Demo")

df = get_UN_data()
st.session_state.chat_engine = load_chat_engine()
start_date = dt.date(year=2021,month=1,day=1)-relativedelta(years=1)  #  I need some range in the past
end_date = dt.datetime.now().date()-relativedelta(years=0)

with st.sidebar.form(key='Search'):
    slider = st.slider('New release period', min_value=start_date, value=(start_date ,end_date) ,max_value=end_date, format='YYYY MMM DD', )
    countries = st.multiselect("Choose countries", df.news_national_type.drop_duplicates().values.tolist(), [])
    text_query = st.text_input(label='Enter text to search')
    submit_button = st.form_submit_button(label='Search')
    if countries:
        df = df[df['news_national_type'].isin(countries)]
    if text_query:
        df = search_dataframe(df, "contents", text_query)
    df_filterd = df[(pd.to_datetime(df['release_date']).dt.date>=slider[0]) & (pd.to_datetime(df['release_date']).dt.date<=slider[1])]

col1, col2 = st.columns([1, 1])
with col1:
    st.write('News dataset')
    dataframe_with_selections(df_filterd)
    
with col2:
    st.write('General ChatAgent')
    st.button('Reset Chat', on_click=reset_conversation)
    container = st.container(height=300)    
    if prompt := st.chat_input("Say something"):
        if st.session_state.firstChatYN == True:
            if len(st.session_state.selection)>0:
                prompt = prompt + "\n\ndocument title: " + st.session_state.selection.title.values[0]
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.firstChatYN = False
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})                    
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        container.chat_message(message['role']).write(message['content'])
    
    if st.session_state.messages[-1]["role"] != "assistant":
        response = st.session_state.chat_engine.chat(prompt)
        container.chat_message("assistant").write(f"Echo: {response.response}")
        message = {"role": "assistant", "content": response.response}
        st.session_state.messages.append(message) # Add response to message history

if len(st.session_state.selection) > 0:
    with st.expander("News info"): 
        st.subheader(st.session_state.selection.title.values[0])
        st.write(st.session_state.selection.url.values[0])
    with st.expander("News contents"): 
        st.write(st.session_state.selection.contents.values[0])

    if st.session_state.AnalyYN == False:
        get_semantic_analysis(st.session_state.selection.contents.values[0])
        # for message in st.session_state.messages: # Display the prior chat messages
        #     container.chat_message(message['role']).write(message['content'])
        st.session_state.AnalyYN = True

    with st.expander("Analyzed"):    
        st.write('Hanwha QCELLS Impact: ', st.session_state.analysis_output2[0])
        st.write('Summary: ', st.session_state.analysis_output2[1])
        st.data_editor(
            st.session_state.analysis_output3,
            column_config={
                "Interesting_Score": st.column_config.ProgressColumn("Interesting_Score",format="%f",min_value=0,max_value=1),
                "Relevance_Score": st.column_config.ProgressColumn("Relevance_Score",format="%f",min_value=0,max_value=1),
            },
            hide_index=True,
            use_container_width = True
        )
    st.success('Done!')

# st.write(st.session_state)
# st.write(type('abc'))