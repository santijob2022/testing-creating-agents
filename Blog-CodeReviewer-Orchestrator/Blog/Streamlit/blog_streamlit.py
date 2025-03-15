# ## Defining the Model to use
import streamlit as st

import os
import json
from dotenv import load_dotenv
load_dotenv() ## aloading all the environment variable

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "langgraph-Blog-Generator"

# os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langsmith import traceable

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
# llm=ChatGroq(model="Llama3-8b-8192")
# llm=ChatGroq(model="qwen-2.5-32b")

# ## Define the Initial State

from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState

# This imports will let me redefine the state of my graph
# from typing_extensions import TypedDict
# from typing import List
# from langchain_core.messages import AnyMessage

# class MessagesState(TypedDict):
#     """State definition for the graph"""
#     messages: List[AnyMessage]
#     articles: List[dict]  # Stores retrieved Arxiv articles
#     arxiv_query: str  # Stores the user input query

# System message
sys_msg = SystemMessage(content=
                        """You are a helpful assistant tasked with creating a blog.
                        Your first commitment is:
                        1. Use the arxiv tool to search for the top 2 articles related to the user's question.

                        2. Give the user the option to use wikipedia in case they want to research a term in the summaries.

                        3. Wait for user's feedback

                        4. If rejected try again from point 1.

                        5. If accepted create a blog with the articles information.
                     
                        """)                      


def assistant(state:MessagesState):
    return {"messages":[llm_with_tools.invoke([sys_msg] + state["messages"])]}


# ## Defining the tools

# #### Arxiv

from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# listed_articles = [] # global variables. This should be saved into a vector database

@traceable
def arxiv_search(state: MessagesState):
    """
    Search for the top 2 results according to the user query that you will see
    in the state["messages"][-1].content
    Use ArxivAPIWrapper to make the search.
    The retrieved articles are stored in the state and returned.
    """
    arxiv = ArxivAPIWrapper(
        top_k_results=2,
        ARXIV_MAX_QUERY_LENGTH=100,
        load_all_available_meta=True,
        doc_content_chars_max=50
    )
    query = state["messages"][-1].content  # Get user query from last message
    # print("Query definition",query)
    # input()
    results = arxiv.load(query)  # Load articles
    listed_articles = []  # Initialize list

    for article in results:
        listed_articles.append({
            "Title": article.metadata.get('Title', 'N/A'),
            "Published": article.metadata.get('Published', 'N/A'),
            "Authors": article.metadata.get('Authors', 'N/A'),
            "Summary": article.metadata.get('Summary', 'N/A')[:250],  # Limit summary length
            "PDF url": article.metadata.get('entry_id', 'N/A')
        })

    # Update state with retrieved articles
    return {**state, "arxiv_query": listed_articles}


# #### Wikipedia

from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun

api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

# ## Binding the tools

tools=[wiki,arxiv_search]

llm_with_tools=llm.bind_tools(tools,parallel_tool_calls=True)


# ## Defining the graph

# #### First I define a node that will be helpful for the human feedback

# def create_blog_entry(state: MessagesState):
#     """Indicates what arxiv articles should be selected to create a post."""

#     # Simulating user input (In real use, take input dynamically)
#     user_input = "Create a blog post about the second article"

#     # Append user input to state
#     new_message = HumanMessage(content=user_input)

#     # Return the updated state with all messages
#     return {"messages": state["messages"] + [new_message]}

def create_blog_entry(state: MessagesState):
    """This function is called when the user has accepted the blog post creation.
    It will be created manually durin execution"""
    pass

from langgraph.graph import START, StateGraph, END

@traceable
def human_feedback(state: MessagesState):
    """ Return the next node to execute """
    
    user_input = state["messages"][-1].content.lower()
    if "yes" in user_input:
        return "tools"
    elif "no" in user_input or "end" in user_input:
        return create_blog_entry 
    return create_blog_entry


@traceable
def custom_condition(state: MessagesState):
    """ Manually decide whether to go to tools or end. """
    decision = human_feedback(state)  # Use the updated function

    if decision == "tools":
        return "tools"
    return "create_blog_entry"  

# #### Then I define the graph

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display

builder = StateGraph(MessagesState)

## Define the node
builder.add_node("assistant",assistant)
builder.add_node("human_feedback",human_feedback)
builder.add_node("tools",ToolNode(tools))
builder.add_node("create_blog_entry",create_blog_entry)
# Define edges
builder.add_edge(START,"assistant")
builder.add_edge("assistant","tools")
builder.add_edge("tools","human_feedback")
builder.add_conditional_edges(
    "human_feedback", custom_condition, {"tools": "tools", "create_blog_entry": "create_blog_entry"}
)
builder.add_edge("create_blog_entry",END)   

# Set up memory
memory = MemorySaver()
# Compile the graph with memory
scienceBlogCreator = builder.compile(interrupt_before=["human_feedback"],checkpointer=memory)

# Show
# display(Image(scienceBlogCreator.get_graph().draw_mermaid_png()))

####################################################################################################
####################################################################################################

# Here starts the Streamlit UI
st.set_page_config(page_title="AI Blog of Science Generator", page_icon="🔬")
st.title("🔬 AI Blog of Science Generator")

# User input
user_input_topic = st.text_input("What do you want to research about in arxiv?", 
                      placeholder="The AI model will return the top 2 articles related to your query.")


if user_input_topic:
    with st.spinner("Researching on arxiv... ⏳"):        
        # Input        
        messages = {"messages": HumanMessage(content=user_input_topic)}
        thread={"configurable":{"thread_id":"arxiv_call_1"}}

        for event in scienceBlogCreator.stream(messages,thread,stream_mode="values"):
            event['messages'][-1].pretty_print()

        ### Ask the user if they want to research in wikipedia


        # Ask user if they want to research in Wikipedia
        st.subheader("🔍 Do you want to clarify any term using Wikipedia?")

        # Initialize session state variables
        if "researching" not in st.session_state:
            st.session_state.researching = True
        if "wiki_queries" not in st.session_state:
            st.session_state.wiki_queries = []

        # Layout buttons
        col1, col2 = st.columns(2)
        
        with col1:            
            if st.button("📖 Research"):
                wiki_term = st.text_input("Enter the term you want to research:")
                user_input = f"{st.button('📖 Research')} Research {wiki_term}"                
                st.session_state.wiki_queries.append(wiki_term)
                st.success(f"✅ Researching: {wiki_term}")

        with col2:
            if st.button("📝 Create Blog"):
                wiki_term = st.text_input("Enter the term you want to research:")
                user_input = f"st.button('📝 Create Blog')"                                
                st.success(f"✅ Creating Blog...")
                st.session_state.researching = False  # Stop research phase
        
        scienceBlogCreator.update_state(
            thread,
            {"messages": [
                SystemMessage(content="""Based on the next user answer, decide if you should return to the tools 
                            node to make a wikipedia search or if you should proceed to create  the blog entry."""),
                HumanMessage(content=user_input)
            ]},
            #  as_node="human_feedback"
        )

        for event in scienceBlogCreator.stream(messages,thread,stream_mode="values"):
            event['messages'][-1].pretty_print()

        # # Process Wikipedia research while still in research mode
        # if st.session_state.researching and st.session_state.wiki_queries:
        #     for query in st.session_state.wiki_queries:
        #         st.write(f"🔍 Searching Wikipedia for: **{query}**...")
        #         wiki_result = wiki.run(query)  # Assuming `wiki` is your Wikipedia tool
        #         st.write(f"📄 **Wikipedia Result:** {wiki_result}")

        # # Transition state in `scienceBlogCreator`
        # if not st.session_state.researching:
        #     scienceBlogCreator.update_state(
        #         thread,
        #         {"messages": [
        #             SystemMessage(content="""Based on the next user answer, decide if you should return to the tools 
        #                         node to make a Wikipedia search or if you should proceed to create the blog entry."""),  
        #             HumanMessage(content="No, create the blog.")  # Simulating user decision
        #         ]}
        #     )

        #     # Stream execution with updated state
        #     for event in scienceBlogCreator.stream(messages, thread, stream_mode="values"):
        #         event['messages'][-1].pretty_print()

            st.subheader("📌 Generating Blog from Arxiv Articles...")



#######################################################################################################

        # user_input=input("Do you want to clarify any term by using Wikipedia?",
        #                  placeholder="Respond by saying: Yes, Research... or No, create the blog.")
        

        # user_input="yes, research what is a network in machine learning using wikipedia"
        # scienceBlogCreator.update_state(
        #     thread,
        #     {"messages": [
        #         SystemMessage(content="""Based on the next user answer, decide if you should return to the tools 
        #                     node to make a wikipedia search or if you should proceed to create  the blog entry."""),
        #         HumanMessage(content=user_input)
        #     ]},
        #     #  as_node="human_feedback"
        # )

        # for event in scienceBlogCreator.stream(messages,thread,stream_mode="values"):
        #     event['messages'][-1].pretty_print()

        
        print("\n\nHEllooooooooooooooooooooooooooooo\n\n")  

        new_state = scienceBlogCreator.get_state(thread).values
        for m in new_state['messages']:
            m.pretty_print()

        print("\n\nHEllooooooooooooooooooooooooooooo\n\n")        
        

        print(scienceBlogCreator.get_state(thread).next)

        print(scienceBlogCreator.get_state(thread).tasks)

        st.success("✅ Articles found!")
        st.subheader("📌 Generated Blog:")
        # for article in listed_articles:
        #     st.write(article)
        

else:
    st.warning("Please enter a topic to research!")


