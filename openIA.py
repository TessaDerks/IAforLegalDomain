
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
from tenacity import (retry, stop_after_attempt, wait_random_exponential)

from langchain.schema import (AIMessage,HumanMessage,SystemMessage,BaseMessage,) 

# insert you key
my_api_key = mysecrets.deepinfra_key['key']
model_name_tool = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# SELECT MODEL (also change folder name below!)
model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
#model_name = "microsoft/phi-4"
#model_name = 'mistralai/Mistral-Small-24B-Instruct-2501'


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

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
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
    def __init__(self,agents: List[DialogueAgent],selection_function: Callable[[int, List[DialogueAgent]], int], case_ind:int, folder:str) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function
        self.myfile = open('Results/'+folder+'/case'+ str(case_ind)+'_agent.txt', 'w', encoding="utf-8")
        

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, sender: str, message: str, agent_receiver):
        """
        Initiates the conversation with a {message} from {name}
        """
        print(f"{sender}: {message}")
        print("\n")
        try:
            self.myfile.write(sender+":  "+ message + "\n")
        except UnicodeEncodeError as e:
            print('ERROR OCCURRED')
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
        try:
            self.myfile.write(speaker.name+":  "+ message + "\n")
        except UnicodeEncodeError as e:
            print('ERROR OCCURRED')

    def stop_writing(self)-> None:
        self.myfile.close()

    def write_appno(self,appno: str) -> None:
        self.myfile.write("APPNO: "+ appno)

def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx
    
# create tools for IA to use
# custom tool for retrieving similar cases
def get_df_tool():
    def get_argument(input: str):
        df_agent = create_pandas_dataframe_agent(
        llm = OpenAI(temperature=0, model=model_name_tool, api_key= my_api_key ,base_url="https://api.deepinfra.com/v1/openai"),
        df = similar_case_database,
        verbose=True,
        allow_dangerous_code=True,
        max_iterations = 4
        )
        return df_agent.invoke(input)
    
    df_tool = Tool.from_function(
        name="dataframe_search_tool",
        func=get_argument,
        description="Searches a dataframe with cases similar to the given case, providing their facts and law arguments."
        )
    return df_tool

# custom tool to retrieve more detailed facts of discussed case
def get_facts_tool():
    def get_facts(input: str):
        df_agent = create_pandas_dataframe_agent(
        llm = OpenAI(temperature=0, model=model_name_tool, api_key= my_api_key ,base_url="https://api.deepinfra.com/v1/openai"),
        df = case,
        verbose=True,
        allow_dangerous_code=True,
        max_iterations=4
        )
        return df_agent.invoke(input)
    
    case_tool = Tool.from_function(
        name="casefacts_search_tool",
        func=get_facts,
        description="Searches a dataframe with all the facts of the case. Can be used for more details or information."
        )
    return case_tool


# length of interaction (has to be even number)
max_iters = 10

# use folder of selected model
folder = 'Lama_instruct'
#folder = 'Mistral'
#folder = 'Microsoft'
database = pd.read_excel('Data/cases_.xlsx')

for i in range(717,725):
    n = 0
    print('ITERATION: ', i)
    # load dataset 
    case_ind = i
    appno = database['appno'].iloc[i]
    print('CASE: ', appno)
    case = database[['facts', 'law','violation']] 
    case_summary = database['summary'].iloc[case_ind]
    

    # find similar cases in the dataset
    similar_cases = ArgumentsGeneration.findSimilarCases(5,case['facts'].iloc[case_ind])
    mask = database.index.isin(similar_cases)
    similar_case_database = database[mask]
    similar_case_database = similar_case_database[['summary','summary2','violation']]
    similar_case_database = similar_case_database.rename(columns={"summary2": "case_arguments", 'summary': 'case_facts'})
    #similar_case_database.to_csv('Results/'+ folder+ '/similarcases_'+str(case_ind)+'.csv')

    conversation_description = f"""Here is the case for the judge to decide if it is a violation of article 10 of the ECHR with the help of the assistant: {case_summary}"""

    # instruction for IA
    IA_message = f"""{conversation_description} 
    You are the (Intelligent) Assistant.
    Your goal is to help the judge.
    Do this by providing arguments and counterarguments why the given case is a violation of article 10 of the ECHR and why not.
    You can use the get_df_tool() for this, it contains similar cases and their arguments.
    You can use the get_facts_tool() to get more detailed facts of the case and how laws can be applied to the case.

    Only supply 1 argument at the time.
    Answer the questions of the judge, you can use the tools for this. 

    DO NOT fabricate fake information.

    Do not add anything else.
    """

    #instruction for law expert
    judge_message = f"""{conversation_description}

    You are the Judge. 

    You want to decide if the given case is a violation of article 10 of the ECHR.
    Use the assistant help you to form an opinion by asking for arguments, counter-arguments or other questions.
    You can also give arguments yourself and ask for a response on your input.
    Have an argumentative interaction.

    ONLY use the case information and input given by the assistant, do not use any knowledge from pretraining or looking on the internet. 
    Keep your responses short.
    """

    # create IA
    IA_agent = DialogueAgentWithTools(
                name= 'Assistant',
                system_message= SystemMessage(content=IA_message),
                model=ChatOpenAI(model =model_name, api_key= my_api_key,base_url="https://api.deepinfra.com/v1/openai",temperature=0), #, frequency_penalty=0.5
                tool_names = [get_df_tool(), get_facts_tool()]) #search_tool,
    # create law expert
    judge_agent = DialogueAgent(
                    name = 'Judge',
                    system_message= SystemMessage(content=judge_message),
                    model=ChatOpenAI(model =model_name, api_key= my_api_key,base_url="https://api.deepinfra.com/v1/openai",temperature=0.3))
    agents = [ judge_agent, IA_agent]

    

    simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker, case_ind=i, folder = folder)
    simulator.write_appno(appno)
    simulator.reset()
    # initiate conversation
    simulator.inject("Moderator", 'Is this a violation of article 10 only based on the case presented now? Answer yes or no. Then ask the assistant for an argument or counter argument to form a better opinion on this case', judge_agent)

    while n < max_iters:
        simulator.step()
        n += 1

    # get final conclusion of law expert
    simulator.inject("Moderator", 'The conversation is done. Is this a violation of article 10 based on the given case AND the interaction you just had? Answer yes or no.', judge_agent)
    simulator.step()
    simulator.stop_writing()