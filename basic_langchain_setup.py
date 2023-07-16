from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# local_path = './models/gpt4all-converted.bin' 
local_path = 'ggml-model-gpt4all-falcon-q4_0.bin'  # Updated model path
callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler()])

template = """Question: {question}

Answer: Let's think step by step.

"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = GPT4All(model=local_path, callback_manager=callback_manager, verbose=True)
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year 1994?"

# question = input("Enter your question: ")

llm_chain.run(question)
