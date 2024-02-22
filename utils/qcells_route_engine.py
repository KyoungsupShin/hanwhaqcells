from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, SemanticSplitterNodeParser
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import nest_asyncio
from llama_index.core.schema import IndexNode
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.chat_engine.condense_question import CondenseQuestionChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import PromptTemplate
from llama_index.core import SummaryIndex, get_response_synthesizer, StorageContext, load_index_from_storage, VectorStoreIndex, set_global_service_context, SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex, Document
import Phoenix as px
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter,FilterOperator
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import set_global_handler
from pydantic import Field, BaseModel
from typing import List
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core.query_engine import CustomQueryEngine, BaseQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.readers.file import PyMuPDFReader
from pathlib import Path
from typing import List, Optional
import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.agent.openai import OpenAIAgent
from langchain_community.document_loaders import PyPDFLoader
from llama_index.core.node_parser import SentenceSplitter
from chromadb.config import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
# from llama_index.readers.web import NewsArticleReader

from llama_index.core.query_engine import CustomQueryEngine, BaseQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize, CompactAndRefine, Refine
from llama_index.core.schema import NodeWithScore
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer


import json
nest_asyncio.apply()

llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="test_gpt",
    temperature = 0,
    api_key="70d67d8dd17f436b9c1b4e38d2558d50",
    azure_endpoint="https://qcells-us-test-openai.openai.azure.com/",
    api_version="2023-07-01-preview",
)

embedding = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="embedding_model",
    api_key="70d67d8dd17f436b9c1b4e38d2558d50",
    azure_endpoint="https://qcells-us-test-openai.openai.azure.com/",
 api_version="2023-07-01-preview",
)
service_context = ServiceContext.from_defaults(llm=llm,embed_model=embedding)
set_global_service_context(service_context)

class VectordbSearchToolSpec():
    def __init__(self):
        self.filters = MetadataFilters(
            filters=[
                MetadataFilter(key="release_date_year", operator=FilterOperator.GTE, value=2024),
                MetadataFilter(key="release_date_year", operator=FilterOperator.LT, value=2025),
            ]
        )
        
        self.filters1 = MetadataFilters(
            filters=[
                MetadataFilter(key="release_date_year", operator=FilterOperator.GTE, value=2023),
                MetadataFilter(key="release_date_year", operator=FilterOperator.LT, value=2024),
            ]
        )
        
        self.filters2 = MetadataFilters(
            filters=[
                MetadataFilter(key="release_date_year", operator=FilterOperator.GTE, value=2022),
                MetadataFilter(key="release_date_year", operator=FilterOperator.LT, value=2023),
            ]
        )
        
        self.filters3 = MetadataFilters(
            filters=[
                MetadataFilter(key="release_date_year", operator=FilterOperator.GTE, value=2021),
                MetadataFilter(key="release_date_year", operator=FilterOperator.LT, value=2022),
            ]
        )
    def connect_db(self):
        db = chromadb.HttpClient(host='4.242.8.48', port=8000, settings=Settings(chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",chroma_client_auth_credentials="qcells:qcells"))
        chroma_collection = db.get_collection("pv_magazine_sentence_split")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.index = VectorStoreIndex.from_vector_store(vector_store,service_context=service_context, similarity_top_k=5, use_async=True)       

    def create_indexes(self):
        self.vector_queryengine = self.index.as_query_engine(vector_store_query_mode="hybrid", alpha = 0.2 , similarity_top_k=4, filters = self.filters, verbose = True)
        self.vector_queryengine1 = self.index.as_query_engine(vector_store_query_mode="hybrid", alpha = 0.2 , similarity_top_k=4, filters = self.filters1, verbose = True)
        self.vector_queryengine2 = self.index.as_query_engine(vector_store_query_mode="hybrid", alpha = 0.2 , similarity_top_k=4, filters = self.filters2, verbose = True)
        self.vector_queryengine3 = self.index.as_query_engine(vector_store_query_mode="hybrid", alpha = 0.2 , similarity_top_k=4, filters = self.filters3, verbose = True)

    def create_tools(self):
        self.query_engine_tools = [
            QueryEngineTool(
                query_engine=self.vector_queryengine,
                metadata=ToolMetadata(
                    name="news_retriever in 2024",
                    description="Provides Renewable Energy industry related news documents that are released in 2024. this documents are stored in vector database."
                ),
            ),
        
            QueryEngineTool(
                query_engine=self.vector_queryengine1,
                metadata=ToolMetadata(
                    name="news_retriever in 2023",
                    description="Provides Renewable Energy industry related news documents that are released in 2023. this documents are stored in vector database."
                ),
            ),
        
            QueryEngineTool(
                query_engine=self.vector_queryengine2,
                metadata=ToolMetadata(
                    name="news_retriever in 2022",
                    description="Provides Renewable Energy industry related news documents that are released in 2022. this documents are stored in vector database."
                ),
            ),
            QueryEngineTool(
                query_engine=self.vector_queryengine3,
                metadata=ToolMetadata(
                    name="news_retriever in 2021",
                    description="Provides Renewable Energy industry related news documents that are released in 2021. this documents are stored in vector database."
                ),
            ),
        ]        
    def create_query_engine(self):
        self.connect_db()
        self.create_indexes()
        self.create_tools()
        news_agent = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=self.query_engine_tools,
            service_context=service_context,
            use_async=True,
            response_synthesizer = get_response_synthesizer(response_mode="tree_summarize",)
        )   
        return news_agent

# class WebUrlSearchToolSpec(BaseToolSpec):
#     spec_functions = ["url_indexing"]
#     def __init__(self)-> None:
#         self.reader = NewsArticleReader(use_nlp=False)
        
#     def url_indexing(self, url):
#         "extract web url link from {query_str}, then use this function in order to create indexes"        
#         documents = self.reader.load_data([url])
#         for doc in documents:
#             doc.metadata['publish_date'] = json.dumps(doc.metadata['publish_date'],default=str)
#         index_name = GPTVectorStoreIndex.from_documents(documents, show_progress=True)
#         self.get_engine = index_name.as_query_engine()  
        
#     def url_query(self, query_str: str):
#         "useful question and answer function about web page documents."
#         return self.get_engine.query(query_str)

class GoogleSearchToolSpec(BaseToolSpec):
    spec_functions = ["google_patent_search"]
    def __init__(self):
        self.my_api_key = "AIzaSyBZaepCCskamC_j3aBLnUNfOTRcpBgNteU"
        self.my_cse_id = "d18770dfc867442e9"
        self.url = 'https://www.googleapis.com/customsearch/v1'

    def _google_request(self, search_query):
        params = {
            'q' : search_query,
            'key' : self.my_api_key,
            'cx' : self.my_cse_id
        }
        response = requests.get(self.url, params=params)
        return response.json()['items']
    
    def google_patent_search(self, query: str):
        response = self._google_request(search_query=query)
        sliced_response = [f'''1. title: {i['title']} \n2. url: {i['link']} \n3. pdf_document: {i['pagemap']['metatags'][0]['citation_pdf_url']}''' for i in response[:10]]
        return sliced_response

class PdfUrlSearchToolSpec(BaseToolSpec):
    spec_functions = ["url_pdf_indexing", "url_pdf_query"]
    def __init__(self)-> None:
        self.splitter = SentenceSplitter(chunk_size=256,chunk_overlap=20)
        self.pdf_url = ''
        
    def url_pdf_indexing(self, url:str, query:str):
        "extract url link from {query_str}, then use this function in order to create indexes"
        if url:
            self.pdf_url = url
            
        loader = PyPDFLoader(url)
        documents = loader.load_and_split()
        texts = ' '.join([i.page_content for i in documents])
        documents = [Document(text=texts)]
        self.nodes = self.splitter.get_nodes_from_documents(documents)
        self.index = VectorStoreIndex(nodes = self.nodes, service_context=service_context,show_progress=True)
        self.get_engine = self.index.as_query_engine()
        return self.get_engine.query('give summary')
    
    def url_pdf_query(self, query_str: str):
        "useful question and answer function about pdf documents. if not defined query_engine, use url_pdf_indexing function first."
        if not self.get_engine:
            self.get_engine = self.url_pdf_indexing(self.pdf_url, query_str)
        return self.get_engine.query(query_str)

class BingSearchToolSpec(BaseToolSpec):
    spec_functions = ["bing_news_search"]
    def __init__(self, api_key: str, lang: Optional[str] = "en-US", results: Optional[int] = 5) -> None:
        self.api_key = api_key
        self.lang = lang
        self.results = 5

    def _bing_request(self, endpoint: str, query: str, keys: List[str]):
        response = requests.get(
            "https://api.bing.microsoft.com/v7.0/" + endpoint,
            headers={"Ocp-Apim-Subscription-Key": self.api_key},
            params={"q": query, "mkt": self.lang, "freshness" : 'Month', "count": 5},
        )
        response_json = response.json()
        return [[result[key] for key in keys] for result in response_json["value"]]

    def bing_news_search(self, query: str):
        return self._bing_request("news/search", query, ["name", "description", "url"])

    def _query(self, query:str):
        """Answer a query."""
        response = self._bing_request("news/search", query, ["name", "description", "url"])
        response_from_bing = [' '.join(r) for r in response]
        nodes = [NodeWithScore(node=Node(text=t,score=1.0)) for t in response_from_bing]
        response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)        
        response = response_synthesizer.synthesize(query=query_str,nodes=nodes)
        return response

class Answer(BaseModel):
    "Represents a single choice with a reason."
    choice: int
    url: str
    task_input: str
    reason: str

class Answers(BaseModel):
    """Represents a list of answers."""
    answers: List[Answer]

class RouterQueryEngine(CustomQueryEngine):
    """Use our Pydantic program to perform routing."""
    query_engines: List[BaseQueryEngine]
    choice_descriptions: List[str]
    verbose: bool = False
    
    router_prompt: PromptTemplate
    llm: llm
    summarizer: TreeSummarize = Field(default_factory=TreeSummarize)

    url_path = ''
    query_engine= ''
    def custom_query(self, query_str: str):
        """Define custom query."""
        program = OpenAIPydanticProgram.from_defaults(
            output_cls=Answers,
            prompt=router_prompt1,
            verbose=self.verbose,
            llm=self.llm,
        )
        choices_str = get_choice_str(self.choice_descriptions)
        output = program(context_list=choices_str, query_str=query_str)
        if self.verbose:
            print(f"Selected choice(s):")
            for answer in output.answers:
                print(f"Query: {query_str}, Choice: {answer.choice-1}, task: {answer.task_input} , Url: {answer.url}, Reason: {answer.reason}")
        responses = []
        for answer in output.answers:
            choice_idx = answer.choice - 1
            if choice_idx == 0:
                response = news_agent.query(answer.task_input)
                responses.append(response)

            elif choice_idx == 1:
                res = bing_agent.query(answer.task_input)
                responses.append(res)
                
            elif choice_idx == 2:
                res = pdf_agent.chat(answer.task_input)
                responses.append(res)

            elif choice_idx == 3:
                res = patent_agent.query(query_str)
                responses.append(res)            
            break
            
        if len(responses) == 1:
            return responses[0]
        else:
            response_strs = [str(r) for r in responses]
            result_response = self.summarizer.get_response(query_str, response_strs)
            return result_response

def get_choice_str(choices):
    choices_str = "\n\n".join([f"{idx+1}. {c}" for idx, c in enumerate(choices)])
    return choices_str

choices = [
    "Provides Renewable Energy industry news documents from vector database. the vector database is stroing news released in 2021, 2022, 2023 and 2024, which manually gathered from web previously, therefore data quantity is limited.",
    "Provides latest news documents from Bing web browser search. if user indicates using Bing search or wants the latest news, use bing search and provide documents.",
    "Provides url pdf extractor and analyzer. this is to extract from given url(extension .pdf). in addition to, question and answering from user's query when there is url(extension .pdf) in previous conversation.",
    "Provides patent documents from Google patent search engine. Search patent documents when user want to get patents documents on web browser(this is not for analyzer)"
]

router_prompt0 = PromptTemplate(
    "Some choices are given below. It is provided in a numbered list (1 to {num_choices}), where each item in the list corresponds to a summary."
    "\n---------------------\n{context_list}\n---------------------\n"
    "Using only the choices above and not prior knowledge, return the top choices (no more than {max_outputs}, but only select what is needed) that are most relevant to the question : '{query_str}'\n"
    "Additionally, extract url link from {query_str} if exists. do not make up any url."
    "Must provide task_input that is refined and simplified from query_str. task_input should be answerable by web browser search"
    "for example, simplify from 'using bing search, find hanwha qcells news related to new product' to 'hanwha qcells new product.'"
    
)

def main():
    router_prompt1 = router_prompt0.partial_format(
        num_choices=len(choices),
        max_outputs=len(choices),
    )
    vector_search = VectordbSearchToolSpec()
    news_agent = vector_search.create_query_engine()
    
    bing_tool_spec = BingSearchToolSpec(api_key="a94a4c9cc6ad49ce9fcb4e2a4db7c4e5")
    bing_agent = OpenAIAgent.from_tools(bing_tool_spec.to_tool_list(), llm = llm)
    
    pdf_url_tool = PdfUrlSearchToolSpec()
    pdf_agent = OpenAIAgent.from_tools(pdf_url_tool.to_tool_list(), llm = llm, verbose = True)
    
    # web_url_tool = WebUrlSearchToolSpec()
    # web_agent = OpenAIAgent.from_tools(web_url_tool.to_tool_list(), llm = llm, verbose = True)
    
    patent_tool_spec = GoogleSearchToolSpec()
    patent_agent = OpenAIAgent.from_tools(patent_tool_spec.to_tool_list(), llm = llm)
    
    router_query_engine = RouterQueryEngine(
        query_engines=[news_agent, bing_agent, pdf_agent, patent_agent],
        choice_descriptions=choices,
        verbose=True,
        router_prompt=router_prompt1,
        llm=llm,
    )
    
    custom_prompt = PromptTemplate(
    """\
    Given a conversation (between Human and Assistant) and a follow up message from Human, \
    rewrite the message to be a standalone question that captures all relevant context \
    from the conversation.
    
    <Chat History>
    {chat_history}
    
    <Follow Up Message>
    {question}
    
    <Standalone question>
    """
    )
    memory = ChatMemoryBuffer.from_defaults(token_limit=20000)
    chat_engine = CondenseQuestionChatEngine(query_engine = router_query_engine, condense_question_prompt = custom_prompt, llm = llm, memory = memory, verbose = True)
    return chat_engine

if __name__:
    chat_engine = main()