import nest_asyncio
nest_asyncio.apply()
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTVectorStoreIndex
from langchain import OpenAI
import os

os.environ["OPENAI_API_KEY"] = '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'

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=-1, streaming=True))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
documents = SimpleDirectoryReader(input_files=["paper.pdf"]).load_data()
index = GPTVectorStoreIndex.from_documents(documents)
gpt_engine = index.as_query_engine(indsimilarity_top_k=-1)

response = gpt_engine.query("chatgpt 답변의 길이가 짧은데 늘리는 법 알려줘")
print(response)
