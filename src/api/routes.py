# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

import asyncio
import json
import os
import logging
from typing import AsyncGenerator, Optional, Dict

import fastapi
from fastapi import Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import (
    Agent,
    MessageDeltaChunk,
    ThreadMessage,
    ThreadRun,
    AsyncAgentEventHandler,
)

logger = logging.getLogger("azureaiapp")
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

directory = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=directory)

router = fastapi.APIRouter()


def get_ai_client(request: Request) -> AIProjectClient:
    """
    Retrieve the AIProjectClient from the FastAPI app state.

    :param request: The incoming HTTP request.
    :type request: Request
    :return: The AIProjectClient from the app state.
    :rtype: AIProjectClient
    """
    return request.app.state.ai_client


def get_agent(request: Request) -> Agent:
    """
    Retrieve the Agent from the FastAPI app state.

    :param request: The incoming HTTP request.
    :type request: Request
    :return: The Agent from the app state.
    :rtype: Agent
    """
    return request.app.state.agent


def serialize_sse_event(data: Dict) -> str:
    """
    Convert a dictionary to a Server-Sent Event (SSE) string.

    :param data: The data to be formatted into SSE.
    :type data: Dict
    :return: An SSE-formatted string.
    :rtype: str
    """
    return f"data: {json.dumps(data)}\n\n"


class MyEventHandler(AsyncAgentEventHandler[str]):
    """
    Custom event handler to receive streamed events from an AI agent.
    Each overridden method returns an SSE-formatted string (or None to skip).
    """

    def __init__(self, ai_client: AIProjectClient) -> None:
        """
        Initialize the MyEventHandler with an AIProjectClient.

        :param ai_client: The AIProjectClient used for fetching file details.
        :type ai_client: AIProjectClient
        """
        super().__init__()
        self.ai_client = ai_client

    async def on_message_delta(self, delta: MessageDeltaChunk) -> Optional[str]:
        """
        Called as partial message content is generated by the agent.

        :param delta: The chunk of text for this partial message.
        :type delta: MessageDeltaChunk
        :return: SSE-formatted data or None to skip publishing.
        :rtype: Optional[str]
        """
        stream_data = {'content': delta.text, 'type': "message"}
        return serialize_sse_event(stream_data)

    async def on_thread_message(self, message: ThreadMessage) -> Optional[str]:
        """
        Called when a new message is posted to the thread.

        :param message: The thread message object from the agent.
        :type message: ThreadMessage
        :return: SSE-formatted data if the message is complete, otherwise None.
        :rtype: Optional[str]
        """
        try:
            logger.info(f"Received thread message, ID: {message.id}, status: {message.status}")
            if message.status != "completed":
                return None

            annotations = []
            for annotation in (a.as_dict() for a in message.file_citation_annotations):
                file_id = annotation["file_citation"]["file_id"]
                logger.info(f"Fetching file by ID for annotation {file_id}")
                openai_file = await self.ai_client.agents.get_file(file_id)
                annotation["file_name"] = openai_file.filename
                annotations.append(annotation)

            stream_data = {
                'content': message.text_messages[0].text.value,
                'annotations': annotations,
                'type': "completed_message"
            }
            return serialize_sse_event(stream_data)
        except Exception as e:
            logger.error("Error in on_thread_message handler", exc_info=True)
            return None

    async def on_thread_run(self, run: ThreadRun) -> Optional[str]:
        """
        Called for thread run events, such as agent actions or steps.

        :param run: The ThreadRun details from the agent.
        :type run: ThreadRun
        :return: SSE-formatted run result, or None to skip.
        :rtype: Optional[str]
        """
        logger.info("Received on_thread_run event")
        run_info = f"ThreadRun status: {run.status}, thread ID: {run.thread_id}"
        if run.status == "failed":
            run_info += f", error: {run.last_error}"
        return serialize_sse_event({'content': run_info, 'type': 'thread_run'})

    async def on_error(self, data: str) -> Optional[str]:
        """
        Called if an error occurs during the streaming process.

        :param data: The error message or context.
        :type data: str
        :return: SSE-formatted data or None.
        :rtype: Optional[str]
        """
        logger.error(f"on_error event: {data}")
        return serialize_sse_event({'type': "stream_end"})

    async def on_done(self) -> Optional[str]:
        """
        Called after all events have been processed.

        :return: SSE-formatted final indicator or None.
        :rtype: Optional[str]
        """
        logger.info("on_done event received")
        return serialize_sse_event({'type': "stream_end"})


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """
    Render the main index page.

    :param request: The incoming HTTP request.
    :type request: Request
    :return: The rendered `index.html` page.
    :rtype: HTMLResponse
    """
    return templates.TemplateResponse("index.html", {"request": request})


async def get_result(thread_id: str, agent_id: str, ai_client: AIProjectClient) -> AsyncGenerator[str, None]:
    """
    Create a generator that yields SSE events for an agent's thread.

    :param thread_id: The unique ID of the conversation thread.
    :type thread_id: str
    :param agent_id: The associated agent ID.
    :type agent_id: str
    :param ai_client: The AIProjectClient instance to fetch stream events.
    :type ai_client: AIProjectClient
    :yield: SSE-formatted string events.
    :rtype: AsyncGenerator[str, None]
    """
    logger.info(f"get_result invoked for thread_id={thread_id}, agent_id={agent_id}")
    try:
        async with await ai_client.agents.create_stream(
            thread_id=thread_id,
            assistant_id=agent_id,
            event_handler=MyEventHandler(ai_client)
        ) as stream:
            logger.info("Successfully created stream; processing events")
            async for event in stream:
                _, _, event_return_val = event
                if event_return_val:
                    logger.debug(f"Yielding SSE event: {event_return_val.strip()}")
                    yield event_return_val
                else:
                    logger.debug("Received event but nothing to yield")
    except Exception as e:
        logger.exception(f"Exception in get_result: {e}")
        yield serialize_sse_event({'type': "error", 'message': str(e)})


@router.post("/chat")
async def chat(
    request: Request,
    ai_client: AIProjectClient = Depends(get_ai_client),
    agent: Agent = Depends(get_agent),
) -> StreamingResponse:
    """
    Handle user chats by creating (or reusing) a thread, sending a user message,
    and returning a streaming SSE response.

    :param request: The incoming HTTP request with JSON body {"message": "..."}.
    :type request: Request
    :param ai_client: Dependency-injected AIProjectClient from app state.
    :type ai_client: AIProjectClient
    :param agent: Dependency-injected Agent from app state.
    :type agent: Agent
    :return: A streaming SSE response of the conversation's events.
    :rtype: StreamingResponse
    """
    thread_id = request.cookies.get('thread_id')
    agent_id = request.cookies.get('agent_id')

    try:
        if thread_id and agent_id == agent.id:
            logger.info(f"Retrieving existing thread with ID {thread_id}")
            thread = await ai_client.agents.get_thread(thread_id)
        else:
            logger.info("Creating new thread")
            thread = await ai_client.agents.create_thread()
    except Exception as e:
        logger.error(f"Error handling thread creation or retrieval: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Thread error: {e}")

    thread_id = thread.id
    agent_id = agent.id

    try:
        user_data = await request.json()
        user_message = user_data.get('message', '')
    except Exception as e:
        logger.error(f"Invalid JSON in request: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    try:
        message = await ai_client.agents.create_message(
            thread_id=thread_id,
            role="user",
            content=user_message
        )
        logger.info(f"Created user message, ID: {message.id}")
    except Exception as e:
        logger.error(f"Error creating user message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Message creation error: {e}")

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
    }
    logger.info(f"Starting SSE stream for thread ID {thread_id}")
    response = StreamingResponse(get_result(thread_id, agent_id, ai_client), headers=headers)

    # Persist the thread and agent IDs via cookies
    response.set_cookie("thread_id", thread_id)
    response.set_cookie("agent_id", agent_id)
    return response


@router.get("/fetch-document")
async def fetch_document(request: Request) -> fastapi.Response:
    """
    Fetch the contents of a previously uploaded file by its name.

    :param request: The incoming HTTP request (expects ?file_name=...).
    :type request: Request
    :return: A plain-text response with file contents or a JSON error if not found.
    :rtype: fastapi.Response
    """
    file_name = request.query_params.get('file_name')
    if not file_name:
        raise HTTPException(status_code=400, detail="file_name is required")

    files_env = os.environ.get("UPLOADED_FILE_MAP", "{}")
    try:
        files = json.loads(files_env)
    except json.JSONDecodeError:
        files = {}
        logger.warning("Failed to parse UPLOADED_FILE_MAP from environment variable.", exc_info=True)

    logger.info(f"File requested: {file_name}. Available keys: {list(files.keys())}")
    if file_name not in files:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        file_path = files[file_name]["path"]
        data = await asyncio.to_thread(read_file, file_path)
        return PlainTextResponse(data)
    except Exception as e:
        logger.error(f"Error fetching document '{file_name}': {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


def read_file(path: str) -> str:
    """
    Synchronously read the file content from a given path.

    :param path: The file system path to read from.
    :type path: str
    :return: The file contents.
    :rtype: str
    """
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()
