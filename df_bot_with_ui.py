import pandas as pd
from langchain_community.llms import Ollama
from langchain_experimental.agents import create_pandas_dataframe_agent

DATA = 'Data/synth.xlsx'
MODEL = 'phi3:mini'

prefix = '''
    You are a chatbot for a Hotel, you are given data of Hotels and have to process a query from user by analyzing the data.
    Each row corresponds to a single Hotel and each column is a feature of that Hotel.
    All values in the data are in lowercase.
    You have access to the following tools:
'''

suffix = '''
    Below is an example of how the formatting should be:
    <start>
    Question: how many rows are there?
    Thought: I need to count the rows
    Action: python_repl_ast
    Action Input: df.shape[0]
    Observation: 1000
    Thought: I now know the final answer
    Final Answer: There are 1000 rows.
    <end>
    
    This is the result of `print(df.head())`:
    {df_head}

    Strictly follow the steps in format instructions in order : Question, Thought, Action, Action Input, Observation, Final Answer

    Begin!
    Question: {input}
    {agent_scratchpad}
'''

agent = create_pandas_dataframe_agent(
    llm = Ollama(model = MODEL, temperature = 0, top_p = 0),
    df = pd.read_excel(DATA).iloc[ : , : 12].drop(columns = 'description').map(lambda x: x.lower() if isinstance(x, str) else x),
    prefix = prefix,
    suffix = suffix,
    include_df_in_prompt = None,
    max_iterations = 20,
    verbose = True,
    agent_executor_kwargs = {"handle_parsing_errors" : True},
    allow_dangerous_code = True
)

# UI 
import gradio as gr, time

CSS ="""
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""

with gr.Blocks(css = CSS) as demo:
    gr.Markdown("# KOGO BOT 🦜 (Pandas Agent)")

    chatbot = gr.Chatbot(label="Chat history", elem_id="chatbot")
    message = gr.Textbox(label="Ask me a question!")
    clear = gr.Button("Clear")

    def user(user_message, chat_history):
        return gr.update(value="", interactive=False), chat_history + [[user_message, None]]

    def bot(chat_history):
        user_message = chat_history[-1][0]
        llm_response = agent.invoke(input = user_message)
        bot_message = llm_response["output"]
        chat_history[-1][1] = ""
        for character in bot_message:
            chat_history[-1][1] += character
            time.sleep(0.005)
            yield chat_history

    response = message.submit(user, [message, chatbot], [message, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    response.then(lambda: gr.update(interactive=True), None, [message], queue=False)

demo.queue()
demo.launch(share = True)