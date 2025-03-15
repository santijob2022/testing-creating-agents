import streamlit as st

import os
import json
from dotenv import load_dotenv
load_dotenv() 

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

# ## Define the Initial State
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

##### Wikipedia
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=100)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

# ## Binding the tools
tools=[wiki,arxiv_search]

llm_with_tools=llm.bind_tools(tools,parallel_tool_calls=True)

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

#### Then I define the graph

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

# User input for Arxiv research
user_input_topic = st.text_input(
    "What do you want to research about in Arxiv?", 
    placeholder="The AI model will return the top 2 articles related to your query."
)

# Initialize session state variables
if "researching" not in st.session_state:
    st.session_state.researching = True
if "wiki_queries" not in st.session_state:
    st.session_state.wiki_queries = []
if "blog_ready" not in st.session_state:
    st.session_state.blog_ready = False

# If user has entered a topic, start the research
if user_input_topic and not st.session_state.blog_ready:
    with st.spinner("Searching on arxiv... ⏳"):        
        # Input        
        messages = {"messages": [HumanMessage(content=user_input_topic)]}
        thread = {"configurable": {"thread_id": "arxiv_call_1"}}

        for event in scienceBlogCreator.stream(messages, thread, stream_mode="values"):
            event['messages'][-1].pretty_print()

    tool_message = scienceBlogCreator.get_state(thread)[0]['messages'][-1]
    print("Tool message: ",tool_message)
    researched_articles = eval(tool_message.content)['arxiv_query']
    print("Updated state", researched_articles)
    
    # Ensure two articles exist
    if len(researched_articles) == 2:
        col1, col2 = st.columns(2)  # Create two columns

        # Display first article in column 1
        with col1:
            st.subheader(f"📄 {researched_articles[0]['Title']}")
            st.write(f"**Published:** {researched_articles[0]['Published']}")
            st.write(f"**Authors:** {researched_articles[0]['Authors']}")
            st.write(f"**Summary:** {researched_articles[0]['Summary']}")
            st.markdown(f"[📄 Read Full Paper]({researched_articles[0]['PDF url']})")

        # Display second article in column 2
        with col2:
            st.subheader(f"📄 {researched_articles[1]['Title']}")
            st.write(f"**Published:** {researched_articles[1]['Published']}")
            st.write(f"**Authors:** {researched_articles[1]['Authors']}")
            st.write(f"**Summary:** {researched_articles[1]['Summary']}")
            st.markdown(f"[📄 Read Full Paper]({researched_articles [1]['PDF url']})")
    else:
        st.warning("⚠️ No Arxiv articles found. Please try a different topic.")

    ### Ask the user if they want to research in Wikipedia
    st.subheader("🔍 Do you want to clarify any term using Wikipedia?")

    # Single column layout for text area and buttons
    with st.container():
        # Text area for Wikipedia research input
        wiki_term = st.text_area("Enter the term you want to research:", key="wiki_input")

        # Create a row for both buttons
        col1, col2 = st.columns([1, 1])  # Equal width for both buttons

        user_input_wiki = None  # Initialize user input

        with col1:
            if st.button("📖 Research", key="wiki_btn"):
                if wiki_term.strip():  # Ensure user has entered something
                    user_input_wiki = f"Use wikipedia tool to research {wiki_term}"  # Append button state to user_input
                    st.session_state.wiki_queries.append(user_input_wiki)  # Store action
                    st.success(f"✅ Research on Wikipedia: {wiki_term}")

        with col2:
            if st.button("📝 Create Blog", key="blog_btn"):
                st.session_state.researching = False
                st.session_state.blog_ready = True  # Stop research, proceed to blog creation
                user_input_wiki = "Create Blog"

        # Pass action to the graph only if a button was clicked
        if user_input_wiki and ("Use wikipedia tool" in user_input_wiki):
            scienceBlogCreator.update_state(
                thread,
                {"messages": [
                    SystemMessage(content="""Based on the next user answer, decide if you should return to the tools 
                                node to make a Wikipedia search or if you should proceed to create the blog entry."""),  
                    HumanMessage(content=user_input_wiki)  # Append action to user message
                ]}
            )
            # Stream execution with updated state
            print("\n\nShowing wiki\n\n")
            for event in scienceBlogCreator.stream(messages, thread, stream_mode="values"):
                event['messages'][-1].pretty_print()
            
            # Displaying wikipedia result on the UI
            tool_message = scienceBlogCreator.get_state(thread)[0]['messages'][-1].content
            _, rest = tool_message.split("Page: ", 1)
            print("\n\nRest", rest)
            page_value, summary_value = rest.split("Summary: ")            
            st.subheader(f"📄 {page_value}")
            st.write(f"**Summary:** {summary_value}")

            # wiki_result = eval(tool_message.content) #['arxiv_query']
            # print("Wiki result", wiki_result)

        
        # elif user_input_wiki == "Create Blog":
        #     user_input = "Create a blog post about the second article"
        #     scienceBlogCreator.update_state(
        #             thread,
        #             {"messages": [            
        #                 # HumanMessage(content=user_input),
        #                 SystemMessage(content=f"""The arxiv articles are ordered here: {listed_articles}.
        #                             Pick the appropriate article in the list according to the number requested by the user
        #                             here {user_input}.
        #                             For instance, if the user wants to create a blog post about the first article,
        #                             then you should create a blog post about {listed_articles[0]}.
        #                             If the user wants to create a blog post about the second article,
        #                             then you should create a blog post about {listed_articles[1]}, and so on                       
        #                             It should be an engaging post of at most 200 words.
        #                             The title of the post should be the title of the article.
        #                             """)
        #             ]}, as_node="create_blog_entry"
        #         )

        #     for event in scienceBlogCreator.stream(messages,thread,stream_mode="values"):
        #         event['messages'][-1].pretty_print()



            # # Stream execution with updated state
            # print("\n\nShowing blog\n\n")
            # for event in scienceBlogCreator.stream(messages, thread, stream_mode="values"):
            #     event['messages'][-1].pretty_print()
            # # Display message if transitioning to blog creation
            # if scienceBlogCreator.get_state(thread).next == "create_blog_entry":
            #     st.success("✅ Blog created successfully!")



        # new_state = scienceBlogCreator.get_state(thread).values
        # for m in new_state['messages']:
        #     m.pretty_print()

        # print(scienceBlogCreator.get_state(thread).next)

        # print(scienceBlogCreator.get_state(thread).tasks)

        # st.success("✅ Articles found!")
        # st.subheader("📌 Generated Blog:")
        # # for article in listed_articles:
        # #     st.write(article)
        

else:
    st.warning("Please enter a topic to research!")


