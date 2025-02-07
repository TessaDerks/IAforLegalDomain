
from langchain.agents import  Tool
import mysecrets
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import ArgumentsGeneration
from typing import List, Callable
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import OpenAI

from langchain.schema import (AIMessage,HumanMessage,SystemMessage,BaseMessage,) 

# insert you key
my_api_key = mysecrets.deepinfra_key4['key']
#llm = OpenAI(api_key = my_api_key,base_url="https://api.deepinfra.com/v1/openai")

model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# load dataset 
database = pd.read_csv('Data/article10_cases_.csv')
case_ind = 699
case = database[['facts', 'law','violation']] 
case_summary = database['summary'].iloc[case_ind]

# find similar cases in the dataset
similar_cases = ArgumentsGeneration.findSimilarCases(4,case['facts'].iloc[case_ind])
mask = database.index.isin(similar_cases)
similar_case_database = database[mask]
similar_case_database = similar_case_database[['summary','summary2','violation']]
similar_case_database = similar_case_database.rename(columns={"summary2": "case_arguments", 'summary': 'case_facts'})
similar_case_database.to_csv('Results/similarcases_'+str(case_ind)+'.csv')

# simulate law expert
class DialogueAgent:
    def __init__(self, name: str, system_message: SystemMessage,model: ChatOpenAI) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]
    
    def send(self) -> str:
        """
        Applies the chatmodel to the message history and returns the message string
        """
        prompt = PromptTemplate.from_template("{input}")
                   
        agent_chain = LLMChain(llm=self.model, prompt=prompt) 

        content = agent_chain.invoke(input="\n".join(
                    [self.system_message.content] + self.message_history + [self.prefix]))

        message = AIMessage(content=content['text'])
        
        return message.content
    

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")

# Intelligent Assistant agent
class DialogueAgentWithTools(DialogueAgent):
    def __init__(self,name: str,system_message: SystemMessage,model: ChatOpenAI,tool_names) -> None:
        super().__init__(name, system_message, model)
        self.tools = tool_names

    def send(self) -> str:
        """
        Applies the chatmodel to the message history and returns the message string
        """
        agent_chain = initialize_agent(
            self.tools,
            self.model,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            ),
        )
        
        content = agent_chain.invoke(
                input="\n".join(
                    [self.system_message.content] + self.message_history + [self.prefix]
                ))
        message = AIMessage(content=content['output'])

        return message.content

# interactive conversation between (simulated) law expert and IA
class DialogueSimulator:
    def __init__(self,agents: List[DialogueAgent],selection_function: Callable[[int, List[DialogueAgent]], int]) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function
        self.myfile = open('Results/case'+ str(case_ind)+'_agent.txt', 'w')

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, sender: str, message: str, agent_receiver):
        """
        Initiates the conversation with a {message} from {name}
        """
        print(f"{sender}: {message}")
        print("\n")
        self.myfile.write(sender+":  "+ message + "\n")
        agent_receiver.receive(sender, message)

    def step(self) -> None:
        # choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]
        print(speaker.name)
        # next speaker sends message
        message = speaker.send()
        
        # everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # increment time
        self._step += 1
        print(f"{speaker.name} : {message}")
        self.myfile.write(speaker.name+":  "+ message + " \n")
    
# create tools for IA to use
# custom tool for retrieving similar cases
def get_df_tool():
    def get_argument(input: str):
        df_agent = create_pandas_dataframe_agent(
        llm = OpenAI(temperature=0, model=model_name, api_key= my_api_key ,base_url="https://api.deepinfra.com/v1/openai"),
        df = similar_case_database,
        verbose=True,
        allow_dangerous_code=True,
        max_iterations = 4
        )
        return df_agent.invoke(input)
    
    df_tool = Tool.from_function(
        name="dataframe_search_tool",
        func=get_argument,
        description="Searches a dataframe with similar cases for relevant arguments"
        )
    return df_tool

# custom tool to retrieve more detailed facts of discussed case
def get_facts_tool():
    def get_facts(input: str):
        df_agent = create_pandas_dataframe_agent(
        llm = OpenAI(temperature=0, model=model_name, api_key= my_api_key ,base_url="https://api.deepinfra.com/v1/openai"),
        df = case,
        verbose=True,
        allow_dangerous_code=True,
        max_iterations=4
        )
        return df_agent.invoke(input)
    
    case_tool = Tool.from_function(
        name="casefacts_search_tool",
        func=get_facts,
        description="Searches a dataframe with facts of the case"
        )
    return case_tool

# tool for searching questions on the internet
search = DuckDuckGoSearchRun()
search_tool = Tool(name = 'search tool',func= search.run, description='useful for providing extra information')

conversation_description = f"""Here is the case for the judge to decide if it is a violation of article 10 of the ECHR with the help of the assistant: {case_summary}"""

# instruction for IA
IA_message = f"""{conversation_description} 
You are the (Intelligent) Assistant.
Your goal is to help the judge.
Do this by providing arguments and counterarguments why the given case is a violation of article 10 of the ECHR and why not.
You can use the get_df_tool() for this, it contains similar cases and their arguments.
You can use the get_facts_tool() to get more detailed facts of the case.
You can use the search_tool to look up information.
Only supply 1 argument at the time.
Answer the questions of the judge, you can use the tools for this. 

DO NOT fabricate fake information.

Do not add anything else.
"""

 #instruction for law expert
judge_message = f"""{conversation_description}

You are the Judge. 

You want to decide if the given case is a violation of article 10 of the ECHR.
Use the assistant to form an opinion with questions and arguments.
Have an argumentative interaction with the assistant to do this.
Let the assistant help you to form an opinion by asking for arguments, counter-arguments or clarifications.

ONLY use the case information and input given by the assistant, do not use any knowledge from pretraining or looking on the internet. 
Keep your responses short.
DO NOT use tools
"""

# create IA
IA_agent = DialogueAgentWithTools(
            name= 'Assistant',
            system_message= SystemMessage(content=IA_message),
            model=ChatOpenAI(model =model_name, api_key= my_api_key,base_url="https://api.deepinfra.com/v1/openai",temperature=0),
            tool_names = [search_tool,get_df_tool(), get_facts_tool()])
# create law expert
judge_agent = DialogueAgent(
                name = 'Judge',
                system_message= SystemMessage(content=judge_message),
                model=ChatOpenAI(model =model_name, api_key= my_api_key,base_url="https://api.deepinfra.com/v1/openai",temperature=0.3))
agents = [ judge_agent, IA_agent]


def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx

# length of interaction (has to be even number)
max_iters = 4
n = 0

simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)
simulator.reset()
# initiate conversation
simulator.inject("Moderator", 'Is this a violation of article 10 only based on the case presented now? Answer yes or no. Then ask the assistant something to form a better opinion on this case', judge_agent)

while n < max_iters:
    simulator.step()
    n += 1

# get final conclusion of law expert
simulator.inject("Moderator", 'Is this a violation of article 10 based on the given case AND the interaction you just had? Answer yes or no.', judge_agent)
simulator.step()