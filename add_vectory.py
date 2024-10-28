##  定义RAG接口
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser,SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

from chTokenizer import split_by_re

def creat_local_vectory():
    path = '/home/workspace/ps/zdy/rag/localdata/PDF-TXT/XNY/'
    documents = SimpleDirectoryReader(path).load_data(show_progress=True) # 读文档
    embedding = HuggingFaceEmbedding('./embedding')
    Settings.embed_model = embedding # 设置全局模型
    #node_parser = SentenceWindowNodeParser.from_defaults(
            #window_size=3,
            #window_metadata_key="window",
            #original_text_metadata_key="original_text",
            #sentence_splitter=split_by_re()
        #)
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist('./local_index')
creat_local_vectory()