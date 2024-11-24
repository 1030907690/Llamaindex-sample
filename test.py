

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
import logging
import sys

# 增加日志信息
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
# 配置HuggingFaceEmbeddings嵌入模型，这里我们用BAAI/bge-small-zh-v1.5
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.ollama import OllamaEmbedding
Settings.embed_model = OllamaEmbedding(model_name="quentinz/bge-small-zh-v1.5")
# 配置ollama的LLM模型，这里我们用qwen2.5:14b
Settings.llm = Ollama(model="qwen2.5-coder", request_timeout=600.0)




data_file = ['D:/work/self/Llamaindex-sample/data/a.txt']
documents = SimpleDirectoryReader(input_files=data_file).load_data()
index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=256)])

query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("介绍一下CSDN博主愤怒的苹果ext")
print(response)

