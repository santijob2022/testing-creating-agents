from langchain_groq import ChatGroq
from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(find_dotenv())


def draft_email(user_input, name="Valentin"):
    # chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)
    chat = ChatGroq(model="Gemma2-9b-It")

    template = """
    
    You are a helpful assistant that drafts an email reply based on an a new email.
    
    Your goal is to help the user quickly create a perfect email reply.
    
    Keep your reply short and to the point and mimic the style of the email so you reply in a similar manner to match the tone.
    
    Start your reply by saying: "Hi {name}, here's a draft for your reply:". And then proceed with the reply on a new line.
    
    Make sure to sign of with {signature}.
    
    """

    signature = f"Kind regards, \n{name}"
    messages = [
        SystemMessage(content=template.format(name=name, signature=signature)),
        HumanMessage(content=f"Here's the email to reply to and consider any other comments from the user: {user_input}")
    ]

    # Create the chat prompt
    chat_prompt = ChatPromptTemplate(messages=messages)

    # Define and execute the chain
    chain = chat_prompt | chat
    response = chain.invoke({
        "user_input": user_input, 
        "signature": signature, 
        "name": name
    })
    return response.content if isinstance(response, AIMessage) else response
    # return response