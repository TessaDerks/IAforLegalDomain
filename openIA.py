
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
import mysecrets
#from langchain.agents import load_tools
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import ArgumentsGeneration
from typing import List, Dict, Callable
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
#from langchain_community.tools import DuckDuckGoSearchTool
from langchain_community.llms import OpenAI


from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
) 
#DialogueAgentWithTools class that augments DialogueAgent to use tools.
my_api_key = mysecrets.deepinfra_key3['key']
llm = OpenAI(
    api_key = my_api_key,
    base_url="https://api.deepinfra.com/v1/openai")


model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# retrieve similar cases in the ECHR database 
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

class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]
    
    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        agent_chain = initialize_agent(tools = [],
            llm = self.model,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            ),
        )
        message = AIMessage(
            content=agent_chain.invoke(
                input="\n".join(
                    [self.system_message.content] + self.message_history + [self.prefix]
                )
            )
        )

        return message.content
    
    '''
    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content
    '''
    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
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
        # increment time
        #self._step += 1

    #def step(self) -> tuple[str, str]:
    def step(self) -> None:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]
        print(speaker.name)
        # 2. next speaker sends message
        message = speaker.send()
        
        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1
        print(f"{speaker.name} : {message}")
        self.myfile.write(speaker.name+":  "+ message + " \n")
    
class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
        tool_names,
        #**tool_kwargs,
    ) -> None:
        super().__init__(name, system_message, model)
        #self.tools = load_tools(tool_names, **tool_kwargs)
        self.tools = tool_names

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
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
        message = AIMessage(
            content=agent_chain.run(
                input="\n".join(
                    [self.system_message.content] + self.message_history + [self.prefix]
                )
            )
        )

        return message.content



def get_df_tool():
    def get_argument(input: str):
        df_agent = create_pandas_dataframe_agent(
        OpenAI(temperature=0, model=model_name, api_key= my_api_key ,base_url="https://api.deepinfra.com/v1/openai"),
        similar_case_database,
        verbose=True,
        allow_dangerous_code=True,
        )
        return df_agent.invoke(input)
    
    df_tool = Tool.from_function(
        name="dataframe_search_tool",
        func=get_argument,
        description="Searches a dataframe with similar cases for relevant arguments"
        )
    return df_tool
'''
df_tool = df_agent.as_tool(
    name="pet_expert",
    description="Get information about pets.",
)
'''
#duck_tool = load_tools(["ddg-search"], top_k_results = 2)

#search_tool = DuckDuckGoSearchRun()
search = DuckDuckGoSearchRun()
search_tool = Tool(name = 'search tool',func= search.run, description='useful for providing extra information')

#topic = "The current impact of automation and artificial intelligence on employment"
#word_limit = 50  # word limit for task brainstorming

conversation_description = f"""Here is the case for the judge to decide on if it is a violation of article 10 of the ECHR with the help of the IA: {case}"""
'''
names = {
    "IA": ["ddg-search", "df_tool"],
    "Judge": ["arxiv", "ddg-search", "wikipedia"],
}

agent_system_messages = {
    name: generate_system_message(name, description, tools) # 
    for (name, tools), description in zip(names.items(), agent_descriptions.values()) #
}
'''

IA_message = f"""{conversation_description}
    
You are the Intelligent Assistant (IA).

Your goal is to help the judge.
Do this by providing arguments and counterarguments why the given case is a violation of article 10 of the ECHR and why not.
You can use the dataframe search tool for this, it contains similar cases and their arguments.
Only supply 1 argument at the time.
Answer the questions of the judge, you can use the tools for this. 

DO NOT fabricate fake information.

Do not add anything else.
"""

judge_message = f"""{conversation_description}

You are the Judge. 

You want to decide if the given case is a violation of article 10 of the ECHR.
Have an argumentative interaction with the assistant to do this.
Let the assistant help you to form an opinion.

ONLY use the case information and input given by the assistant, do not use any knowledge from pretraining or looking on the internet. 
Keep your responses short.
DO NOT use tools
"""
IA_agent = DialogueAgentWithTools(
            name= 'IA',
            system_message= SystemMessage(content=IA_message),
            model=ChatOpenAI(model =model_name, api_key= my_api_key,base_url="https://api.deepinfra.com/v1/openai",temperature=0),
            tool_names = [search_tool,get_df_tool()], #, 
            
            )
judge_agent = DialogueAgent(
                name = 'Judge',
                system_message= SystemMessage(content=judge_message),
                model=ChatOpenAI(model =model_name, api_key= my_api_key,base_url="https://api.deepinfra.com/v1/openai",temperature=0.3)
                )
agents = [IA_agent, judge_agent]
'''
agents = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-4", temperature=0.2),
        tool_names=tools,
        top_k_results=2,
    )
    for (name, tools), system_message in zip( #
        names.items(), agent_system_messages.values()
    )
]
'''

def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx

max_iters = 4
n = 0

simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)
simulator.reset()
#simulator.inject("Moderator", 'Is this a violation of article 10 only based on the case presented now? Answer yes or no.', judge_agent)
simulator.inject("Moderator",'Can you give me a first argument on this case?', IA_agent )

while n < max_iters:
    #name, message = simulator.step()
    simulator.step()
    #print(f"({name}): {message}")
    #print("\n")
    n += 1

simulator.inject("Moderator", 'Is this a violation of article 10 based on the given case AND the interaction you just had? Answer yes or no.', judge_agent)
simulator.step()