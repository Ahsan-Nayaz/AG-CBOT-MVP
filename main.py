import chainlit as cl
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel
from  utils import SYSTEM_PROMPT, END_CHAT
from typing import Optional, Dict
import asyncio
from datetime import datetime, timedelta
from operator import itemgetter
import os
from dotenv import load_dotenv
import logging

load_dotenv(dotenv_path='.venv/.env')
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGHOST"),
    port=int(os.environ.get("PGPORT")),
    database=os.environ.get("PGDATABASE"),
    user=os.environ.get("PGUSER"),
    password=os.environ.get("PGPASSWORD"),
)
EMBEDDINGS = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    openai_api_version='2023-09-01-preview'
)
# EMBEDDINGS = HuggingFaceEmbeddings()
NAMESPACE = "pgvector/cbot_corpus"
COLLECTION_NAME = 'CBOT-CORPUS'
REDIS_URL = os.getenv('REDIS_URL')
vectorstore = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=EMBEDDINGS,
)
retriever = vectorstore.as_retriever()
prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
# prompt = hub.pull("rlm/rag-prompt")
MESSAGE_TIMEOUT = 60

# This is the event to notify when a new message is received
new_message_event = asyncio.Event()

@cl.header_auth_callback
def header_auth_callback(headers: Dict) -> Optional[cl.User]:
  if headers:
    return cl.User(identifier="user", metadata={"role": "user", "provider": "header"})
  else:
    return None

async def send_message_after_timeout():
    while True:
        # Wait for the timeout duration
        try:
            await asyncio.wait_for(new_message_event.wait(), MESSAGE_TIMEOUT)
        except asyncio.TimeoutError:
            # If timeout occurs, no new message was received; send your message
            await send_your_message_function()
            background_task = cl.user_session.get('background_task')
            background_task.cancel()
        else:
            # If a new message was received, reset the event
            new_message_event.clear()
async def send_your_message_function():
    # Your logic to send a message from your end
    rating_actions = [
        cl.Action(
            name="rating", value=str(i), label=str(i), description=str(i)
        )
        for i in range(1, 6)
    ]
    await cl.Message(content=END_CHAT, actions=rating_actions).send()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@cl.on_chat_start
async def chat_start():
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4-1106",
        openai_api_version="2023-09-01-preview",
    )
    chain_answer = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "history": itemgetter("history"),
            }
            | prompt
            | llm
            # | StrOutputParser()
    )
    runnable_initial = RunnableWithMessageHistory(
        chain_answer,
        lambda session_id: RedisChatMessageHistory(session_id, url=REDIS_URL),
        input_messages_key="question",
        history_messages_key="history",
    )
    runnable = RunnableParallel(
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    ).assign(answer=runnable_initial)
    cl.user_session.set('runnable_history', runnable_initial)
    cl.user_session.set('runnable', runnable)



@cl.step(name='Fetching relevant docs')
async def fetch_docs(message):
    return await retriever.aget_relevant_documents(message.content)


@cl.on_message
async def on_message(message: cl.Message):
    background_task = cl.user_session.get('background_task')
    if not background_task or background_task.cancelled():
        background_task = asyncio.create_task(send_message_after_timeout())
        cl.user_session.set('background_task', background_task)
    new_message_event.set()
    current_message_time = datetime.utcnow()
    runnable = cl.user_session.get('runnable')
    user_id = cl.user_session.get("id")
    transcript = cl.Action(name="Transcript", value="transcript", description="Transcript")
    rating_actions = [
        cl.Action(
            name="rating", value=str(i), label=str(i), description=str(i)
        )
        for i in range(1, 6)
    ]
    # Set up rating and transcript actions in the user session
    actions = [transcript]
    cl.user_session.set('rating_actions', rating_actions)
    cl.user_session.set('transcript', transcript)
    # await asyncio.create_task(idle_check(user_id, current_message_time))

    msg: cl.Message = cl.Message(content="",actions=actions)
    # relevant_docs = await fetch_docs(message)
    async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler()],
                                  configurable={"session_id": user_id})
    ):
        # print(chunk)
        # sources = []
        new_message_event.set()
        if 'answer' in chunk.keys():
            await msg.stream_token(chunk['answer'].content)
        if 'context' in chunk.keys():
            sources = [source.metadata['source'] for source in chunk['context']]
            # print(sources)
    msg.content += "\n\nSources:"
    for i, link in enumerate(sources, start=1):
        msg.content += f" [{i}]({link})"
    msg.content += "\n"
    await msg.send()
    background_task = cl.user_session.get('background_task')
    if not background_task or background_task.cancelled():
        background_task = asyncio.create_task(send_message_after_timeout())
        cl.user_session.set('background_task', background_task)
    new_message_event.set()



@cl.action_callback("Transcript")
async def on_action_transcript(action: cl.Action):
    """
    Handles the "Transcript" action asynchronously.
    Retrieves the user's runnable and user_id from the user session,
    then gets and sends the session history as a message.
    If an error occurs, logs the error message.
    Removes the action after handling.
    """
    # Retrieve user's runnable and user_id from the user session
    runnable = cl.user_session.get("runnable_history")
    user_id = cl.user_session.get("id")

    if runnable and user_id:
        try:
            if session_history := runnable.get_session_history(user_id):
                await cl.Message(content=session_history).send()
        except Exception as e:
            # Log error message if an error occurs
            logging.error(f"Error occurred while getting session history: {e}")
    else:
        # Log a warning if runnable or user_id is None
        logging.warning("runnable or user_id is None")

    try:
        # Remove the action after handling
        await action.remove()
    except Exception as e:
        # Log error message if an error occurs
        logging.error(f"Error occurred while removing action: {e}")


@cl.action_callback("rating")
async def rating(action: cl.Action):
    """
    Handle the rating action by setting the rating and collecting feedback.
    """
    rating_actions = cl.user_session.get('rating_actions')
    transcript: cl.Action = cl.user_session.get('transcript')

    # Remove all previous rating actions
    for rating_action in rating_actions:
        await rating_action.remove()

    # Remove the transcript if it exists
    if transcript:
        await transcript.remove()

    # Set the rating value in the user session
    cl.user_session.set('rating', action.value)

    # Ask the user for feedback and store it in the user session
    feedback = await cl.AskUserMessage(
        content="Please enter your feedback...",
        timeout=600,
        disable_feedback=True
    ).send()
    transcript = cl.Action(
        name="Transcript",
        value="transcript",
        description="Transcript"
    )
    if feedback:
        cl.user_session.set('feedback', feedback['output'])


        # Send a thank you message and provide the option to view the transcript
        await cl.Message(
            content="Thank you for your feedback!",
            actions=[transcript]
        ).send()
    else:
        await cl.Message(
            content="No feedback provided",
            actions=[transcript]
        ).send()