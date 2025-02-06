import mysecrets
import pandas as pd
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import ArgumentsGeneration

database = pd.read_csv('Data/article10_cases_.csv')
case_ind = 104
case = database['facts'].iloc[case_ind]
case_summary = database['summary'].iloc[case_ind]
similar_cases = ArgumentsGeneration.findSimilarCases(4,case)
mask = database.index.isin(similar_cases)
similar_case_database = database[mask]
similar_case_database = similar_case_database[['summary2','violation']]
similar_case_database = similar_case_database.rename(columns={"summary2": "case_info"})
similar_case_database.to_csv('Results/similarcases_'+str(case_ind)+'.csv')

openai_api_key = mysecrets.deepinfra_key2['key']

IA_prompt = ChatPromptTemplate([
    ("system", "You are an Intelligen Assistant who is provided a case. Provide arguments and counterarguments why the case is a violation of article 10 of the ECHR and why not. Only supply 1 argument at the time and adjust to what the judge wants to know. Give legal advice if asked. "),
    ("user", "The judge said {input}, respond to guide the decision process")
])

'''
# Define Lawyer's LLM and Prompt
IA_prompt = PromptTemplate(
    input_variables=["judge_message"],
    template="""
    You are an Intelligen Assistant. Provide arguments why the case is a violation of article 10 of the ECHR and why not.
    Only supply 1 argument at the time and adjust to what the judge wants to know.  
    The judge said: "{judge_message}"
    """
)
'''
model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
#model_name = "tiiuae/falcon-7b-instruct"

'''
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
IA_pipeline = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        #do_sample=False,
        temperature = 0.7,
        repetition_penalty=0.8,
    ),
)

IA_llm = ChatHuggingFace(llm=IA_pipeline)
'''
IA_llm = ChatOpenAI(model =model_name, api_key= openai_api_key,base_url="https://api.deepinfra.com/v1/openai",temperature=0.3, frequency_penalty=0.8 )

IA_chain = LLMChain(llm=IA_llm, prompt=IA_prompt, memory=ConversationSummaryMemory(llm=IA_llm) )
''' 
IA_agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0.2, model=model_name, api_key= openai_api_key,base_url="https://api.deepinfra.com/v1/openai"),
    similar_case_database,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS, # CHAT_CONVERSATIONAL_REACT_DESCRIPTION?
    # MEMORY?????
)
'''
# Define Judge's LLM and Prompt
'''
judge_prompt = PromptTemplate(
    input_variables=["IA_input"],
    template="""
    You are a judge who has to decide if a case is a violation of article 10. Try to keep your responses short. Use the assistant to form an opinion. 
    Only use the input given here, do not use any knowledge from pretraining or looking on the internet.
    The assistant said: " + IA_input
    """
)
'''
judge_prompt = ChatPromptTemplate([
    ("system", "You are a judge who has to decide if a case is a violation of article 10.  Use the assistant to form an opinion with questions and arguments. Only use the input given here, do not use any knowledge from pretraining or looking on the internet. Try to keep your responses short."),
    ("user", "The assistant said {input}, engage in the discussion")
])

judge_llm = ChatOpenAI(model =model_name, api_key= openai_api_key,base_url="https://api.deepinfra.com/v1/openai", temperature = 0.5, frequency_penalty=0.6)
'''
judge_pipeline = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        #do_sample=False,
        temperature = 0.5,
        repetition_penalty=0.8,
    ),
)

judge_llm = ChatHuggingFace(llm=judge_pipeline)
'''
judge_chain = LLMChain(llm=judge_llm, prompt=judge_prompt, memory=ConversationSummaryMemory(llm=judge_llm))

myfile = open('Results/case'+ str(case_ind)+'.txt', 'w')
    
# Start interaction
initial_input_judge = 'This is the case: '+ case + ' What is your initial ruling, a violation or not? Just answer yes or no.'
initial_verdict = judge_chain.invoke({"input": initial_input_judge})
print(f"Judge: {initial_verdict['text']}")
myfile.write("Initial verdict: " + initial_verdict['text'] + "\n")

initial_input_IA = 'Identify the main points and supporting evidence in the document that support that is a violation. Do the same for supporting it is not a violation. This is the case: ' + case
IA_message = IA_chain.invoke({"input": initial_input_IA})
print(f"IA: {IA_message}")
myfile.write("IA: " + IA_message['text'] + "\n")

# Conversation Loop
for _ in range(4):  # Adjust for more dialogue turns
    judge_message = judge_chain.invoke({"input": IA_message['text']})
    print(f"Judge: {judge_message['text']}")
    myfile.write("JUDGE: " + judge_message['text'] + "\n")

    #IA_message = IA_chain.invoke(judge_message['text']) # , case, similar_case_database['law'].iloc[0]
    IA_message = IA_chain.invoke({"input": judge_message['text']})
    print(f"IA: {IA_message['text']}")
    myfile.write("IA.: " + IA_message['text'] + "\n")

final_verdict_q = IA_message['text'] +'Now we had this conversation, what is your final verdict on the case? Was it a violation or not? Just answer yes or no.'
final_verdict = judge_chain.invoke({"input": final_verdict_q})
print(f"Judge: {final_verdict['text']}")
myfile.write("Final verdict: "+ final_verdict['text'])

myfile.close()