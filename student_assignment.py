import base64
import json
import requests
import time
import traceback
import openai

from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import AzureChatOpenAI
from mimetypes import guess_type
from openai import AzureOpenAI
from model_configurations import get_model_configuration
from pydantic import BaseModel, Field
from typing import List

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)


class hw4_item(BaseModel):
    score: int
    
class hw4_response(BaseModel):
    Result: hw4_item

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []
        
store = {}
def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    ) 

prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
        ("system", "{output_format}")
    ])
    
chain = prompt | llm | RunnableLambda(lambda x: x.content)
chain_with_history = RunnableWithMessageHistory(
    chain,
    # Uses the get_by_session_id function defined in the example
    # above.
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
    system_messages_key="output_format"
)


 # Homework at below
def generate_hw01(question):
    messages = [
        SystemMessage(content="""以Json格式回答紀念日的日期，並確保輸出格式為:
            {
                "Result": [
                    {
                        "date": "xxxx-xx-xx",
                        "name": "holiday name"
                    }
                ]
            }
            請給我純文字就好，不要有```json"""
        ),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    return response.content
    
def generate_hw02(question):
    messages = [
        SystemMessage(content="""回覆成以下的Json格式，解析使用者訊息中會帶的參數，格式為:
            {
                "year": xxxx,  //四碼數字
                "month": x,    //數字
                "country": "xx"  //ISO國家代碼
            }
            請給我純文字就好，不要有```json"""
        ),
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)

    parameters = json.loads(response.content)

    url = 'https://calendarific.com/api/v2/holidays'
    params = {
        'api_key': 'XxAsIXIu1YRJqhWZ4KHx8SC8d4dt1Y0P',
        'country': parameters["country"],
        'year': parameters["year"],
        'month': parameters["month"]
    }

    response = requests.get(url, params=params)
    holidays = ""
    if response.status_code == 200:
        holidays = response.json()

    messages = [
        SystemMessage(content="""以Json格式回答紀念日的日期，並確保輸出格式為:
            {
                "Result": [
                    {
                        "date": "xxxx-xx-xx",
                        "name": "(請用中文的節日名稱)"
                    }
                ]
            }
            請給我純文字就好，不要有```json"""
        ),
        HumanMessage(content=("幫我整理以下這些紀念日: " + str(holidays['response'])))
    ]
    
    response = llm.invoke(messages)
    return response.content
    
    
def generate_hw03(question2, question3):
    response = chain_with_history.invoke(
        {
            "question": question2, 
            "output_format": """回覆成以下的Json格式，解析使用者訊息中會帶的參數，格式為:
                {
                    "year": xxxx,  //四碼數字
                    "month": x,    //數字
                    "country": "xx"  //ISO國家代碼
                }
                請給我純文字就好，不要有```json"""
        },
        config={"configurable": {"session_id": "hw3"}},
    )
    
    parameters = json.loads(response)

    url = 'https://calendarific.com/api/v2/holidays'
    params = {
        'api_key': 'XxAsIXIu1YRJqhWZ4KHx8SC8d4dt1Y0P',
        'country': parameters["country"],
        'year': parameters["year"],
        'month': parameters["month"]
    }

    response = requests.get(url, params=params)
    holidays = ""
    if response.status_code == 200:
        holidays = response.json()
    
    response = chain_with_history.invoke(
        {
            "question": (question3 + " 我取得了以下的節日清單，請以這個清單作為回答依據:" + str(holidays['response'])),
            "output_format": """回覆成以下的Json格式，回答一個boolean與reason，格式為:
            {
                "Result": 
                    {
                        "add": boolean,  // 表示是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 true；否則為 false。
                        "reason": "xxxx"  // 描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，另外請一起印出當前清單中已有的節日。請使用中文描述。
                    }
            }
            請給我純文字就好，不要有```json"""
        },
        config={"configurable": {"session_id": "hw3"}}
    )
    
    return response

    
def generate_hw04(question):
    start_time = time.time()
    image_path = "baseball.png"
    data_url = local_image_to_data_url(image_path)
    '''
    time_1 = time.time()
    total_time = time_1 - start_time
    print(f"Load image to data url time: {total_time:.4f}s")
    '''
    vllm = AzureOpenAI(
        api_key=gpt_config['api_key'],  
        api_version=gpt_config['api_version'],
        base_url=f"{gpt_config['api_base']}openai/deployments/{gpt_config['deployment_name']}",
    )
    '''
    time_2 = time.time()
    total_time = time_2 - time_1
    print(f"Open AzureOpenAI time: {total_time:.4f}s")
    '''
    response = vllm.chat.completions.create(
        model=gpt_config['model_name'],
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
    )
    '''
    time_3 = time.time()
    total_time = time_3 - time_2
    print(f"Image analysis time: {total_time:.4f}s")
    '''
    image_analysis = str(response.choices[0].message.content)
    response = chain_with_history.invoke(
        {
            "question": image_analysis,
            "output_format": f"輸出格式為：{hw4_response.schema_json(indent=4)} 請給我純文字就好，不要有```json"
        },
        config={"configurable": {"session_id": "hw4"}}
    )
    '''
    end_time = time.time()
    total_time = end_time - time_3
    print(f"Output parser time: {total_time:.4f}s")
    total_time = end_time - start_time
    print(f"Total time time: {total_time:.4f}s")
    '''
    return response
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
