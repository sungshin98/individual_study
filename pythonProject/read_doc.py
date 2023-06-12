import nest_asyncio
nest_asyncio.apply()
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, GPTVectorStoreIndex
from langchain import OpenAI
import sys
import openai
import os

openai.api_key = sys.argv[1]
os.environ["OPENAI_API_KEY"] = sys.argv[1]
'''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'''
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=-1, streaming=True))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
documents = SimpleDirectoryReader(input_files=[sys.argv[2]]).load_data()
index = GPTVectorStoreIndex.from_documents(documents)
gpt_engine = index.as_query_engine(indsimilarity_top_k=-1)

while True:
    query = input("Enter your question (or leave blank to exit): ")
    if not query:
        break

    response = gpt_engine.query(query)
    print(response)