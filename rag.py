##  定义RAG接口
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from customllm import QwenCustomLLM
from llama_index.core.vector_stores.simple import SimpleVectorStore
import json
# from llama_index.postprocessor.flag_embedding_reranker import (
#     FlagEmbeddingReranker,
# )
from chTokenizer import split_by_re
default_file_path = '../index'
class RAG():
    def __init__(self,):
        # data_lib:['file','localdata']
        embeddingpath = './embedding'
        embedding_model = HuggingFaceEmbedding(embeddingpath)
        Settings.embed_model = embedding_model

        #self.data_lib = data_lib
        self.index = self.mk_vectory()
        self.llm = QwenCustomLLM()
        Settings.llm = self.llm

        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
            sentence_splitter=split_by_re()
        )
        text_qa_template_str = (
            "Context information is"
            " below.\n---------------------\n{context_str}\n---------------------\nUsing"
            " both the context information and also using your own knowledge, answer"
            " the question: {query_str}\nIf the context isn't helpful, you can also"
            " answer the question on your own.\n"
        )
        self.text_qa_template = PromptTemplate(text_qa_template_str)
        refine_template_str = (
            "The original question is as follows: {query_str}\nWe have provided an"
            " existing answer: {existing_answer}\nWe have the opportunity to refine"
            " the existing answer (only if needed) with some more context"
            " below.\n------------\n{context_msg}\n------------\nUsing both the new"
            " context and your own knowledge, update or repeat the existing answer.\n"
        )
        self.refine_template = PromptTemplate(refine_template_str)
    def mk_vectory(self):
        # if self.data_lib == 'file':
        #documents = SimpleDirectoryReader(input_files=['./localdata/纸包装-头豹词条报告系列2023.txt']).load_data()
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
            sentence_splitter=split_by_re()
        )
        storage_context = StorageContext.from_defaults(persist_dir="./local_index")
        index = load_index_from_storage(storage_context)
        #nodes = node_parser.get_nodes_from_documents(documents)
        #index = VectorStoreIndex(nodes, show_progress=True)
        return index
        # if self.data_lib == 'localdata':
        #     storage_context = StorageContext.from_defaults(persist_dir=default_file_path)
        #     index = load_index_from_storage(storage_context)
        #     return index

    def get_answer(self,prompt):
        #self.index = self.mk_vectory()
        query_engine = self.index.as_query_engine(similarity_top_k=3,
                                             #node_postprocessors=[
                                                 #MetadataReplacementPostProcessor(target_metadata_key="window")],
                                             text_qa_template=self.text_qa_template,
                                             refine_template=self.refine_template)
        response = query_engine.query(prompt)
        return response.response


