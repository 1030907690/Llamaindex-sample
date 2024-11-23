# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.llms.ollama import Ollama
#
# documents = SimpleDirectoryReader("data").load_data()
#
# Settings.embed_model = OllamaEmbedding(model_name="quentinz/bge-small-zh-v1.5")
#
# Settings.llm = Ollama(model="qwen2.5:7b", request_timeout=360.0)
#
# index = VectorStoreIndex.from_documents(
#     documents,
# )
#
# query_engine = index.as_query_engine()
# response = query_engine.query("淘宝是什么?")
# print(response)


# 检索上下文进行对话
# from llama_index.core.memory import ChatMemoryBuffer
# memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
#
# chat_engine = index.as_chat_engine(
#     chat_mode="context",
#     memory=memory,
#     system_prompt=(
#         "You are a chatbot, able to have normal interactions."
#     ),
# )
#
# response = chat_engine.chat("Datawhale是什么？")
# print(response)

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
import logging
import sys

# 增加日志信息
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# 配置ollama的LLM模型，这里我们用qwen2.5:14b
Settings.llm = Ollama(model="qwen2.5:7b", request_timeout=600.0)

# 配置HuggingFaceEmbeddings嵌入模型，这里我们用BAAI/bge-small-zh-v1.5
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.ollama import OllamaEmbedding
Settings.embed_model = OllamaEmbedding(model_name="quentinz/bge-small-zh-v1.5")


data_file = ['D:/work/self/Llamaindex-sample/data/a']
documents = SimpleDirectoryReader(input_files=data_file).load_data()
index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=256)])

query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("阿里巴巴是什么？")
print(response)

