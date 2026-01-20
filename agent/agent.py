from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_openai import ChatOpenAI
from langchain.agents import AgentType,initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from functools import wraps
from queue import PriorityQueue
import threading
from dotenv import load_dotenv
from output_parsers import MultipartResponse, ResponseWithSentiment,WeatherResponse
from tools import WeatherTool, SearchTool, CapDescTool,TimeTool
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load environment variables
load_dotenv("../env")

NUM_LLM_WORKERS = 1
DAEMON = True

class LLM:
    def __init__(self, no_workers=NUM_LLM_WORKERS, daemon=True):
        self.no_workers = no_workers
        self.tasks_commands = None
        self.daemon = daemon
        self.threads: list[threading.Thread] = []
        # === Global task queue for LLM ===
        self.llm_task_queue = PriorityQueue()
       
        # self.chain = self.prompt | self.llm | self.output_parser
        self.running = threading.Event()
        self.running.set()

    def set_comands(self, commands):
        self.tasks_commands = commands

    # === LLM Worker Thread Function ===
    def llm_worker(self):
        pass
       
    def close(self):
        self.running.clear()

    # === Decorator to Queue LLM Tasks ===
    def llm_wrapper(self, func):
        @wraps(func)
        def wrapper(msg, priority):
            self.llm_task_queue.put((priority, (msg, func)))
        return wrapper

    def start(self):
        for _ in range(self.no_workers):
            th = threading.Thread(target=self.llm_worker, daemon=self.daemon)
            th.start()
            self.threads.append(th)

    def wait(self):
        if not self.daemon:
            for th in self.threads:
                th.join()

class AssistantAgent(LLM):
    def __init__(self):
        super().__init__(NUM_LLM_WORKERS,DAEMON)
        # Initialize language model
        # self.llm = ChatOpenAI(
        #     model="gpt-3.5-turbo-0125",
        #     temperature=0.3
        # )
        self.llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1",
         temperature=0.3,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
                              )

        # Initialize tools
        self.tools = [
            WeatherTool(),
            TimeTool(),
            SearchTool(),
            CapDescTool()
        ]
        response_parser = PydanticOutputParser(pydantic_object=ResponseWithSentiment)
        multipart_parser = PydanticOutputParser(pydantic_object=MultipartResponse)
        weather_parser = PydanticOutputParser(pydantic_object=WeatherResponse)
        # Get a prompt from LangChain hub
        # self.agent_prompt = hub.pull("langchain-ai/openai-tools-agent")
        self.agent_prompt = PromptTemplate.from_template("""
            You are a helpful agent. Use tools to answer questions.

            Tools:
            {tools}

            Question: {input}
            {agent_scratchpad}
            """)

        # Create the agent
        # self.agent = create_openai_tools_agent(
        #     self.llm,
        #     self.tools,
        #     self.agent_prompt
        # )
        # Set up memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Create the agent executor
        self.agent_executor = initialize_agent(
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True
        )

        # Create output formatting chain
        self.format_prompt = PromptTemplate(
            template="""
            You are a voice assistant that needs to provide clear, concise responses.
            
            Below is the raw response to the user's query:
            {raw_response}
            
            Please format this as a response with sentiment analysis, following this format:
            {format_instructions}
            
            Ensure the response is conversational and suitable for speech output.
            """,
            input_variables=["raw_response"],
            partial_variables={
                "format_instructions": response_parser.get_format_instructions()}
        )

        self.formatting_chain =  self.format_prompt | self.llm | response_parser


    def process_query(self, query):
        """Process user query and return structured response"""
        # try:
        # Get raw response from agent
        raw_response = self.agent_executor.invoke({"input": query})
        print("Raw response :",raw_response)

        # Format the response using the output parser
        formatted_response = self.formatting_chain.run(
            raw_response=raw_response["output"])

        # For debugging
        print(f"Structured Response: {formatted_response}")

        # Return formatted response for speech
        # return formatted_response.format_for_speech()
        return formatted_response
        # except Exception as e:
        #     return f"I encountered an error: {str(e)}"

    def llm_worker(self):
        while self.running.is_set():
            # if self.tasks_commands is None:
            #     print(f"❌ LLM Worker Error: Commands are not set!")
            #     self.close()
        # try:
            priority, (msg, func) = self.llm_task_queue.get()
            print("Msg :",priority, (msg, func))
            response = self.process_query(msg)
            print("Res :",response)
            func(msg,response.answer, response.sentiment)
        # except Exception as e:
        #     print(f"❌ LLM Worker Error: {e}")
        # finally:
            self.llm_task_queue.task_done()

    
def test_assistant():
    import time
    import sys
    from tasks import TASK_PRIORITY

    llm = AssistantAgent()

    @llm.llm_wrapper
    def handle_message(input_message,llm_response, sentiment):
        print(
            f"Input Message : {input_message} LLM Response : {llm_response} Task : {sentiment} Priority : {TASK_PRIORITY[sentiment]}")

    try:
        handle_message("How are you?", 1)
        handle_message("Can you record the task that I am doing?", 2)
        llm.start()
        time.sleep(3)
        handle_message("Stop the recording", 2)
        time.sleep(2)
        handle_message("Start processing the task", 2)
        time.sleep(2)
        handle_message("Start execution of the task", 2)
        # time.sleep(2)
        handle_message("Stop the execution its emergency!", 2)
        time.sleep(5)
    except KeyboardInterrupt:
        sys.exit(0)
    finally:
        llm.close()


if __name__ == "__main__":
    test_assistant()
