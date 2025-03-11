# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

import multiprocessing
import os
import json
import asyncio
import logging
from typing import Dict

from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import FilePurpose, FileSearchTool, AsyncToolSet, Agent, VectorStore
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from logging_config import configure_logging

load_dotenv()

logger = configure_logging(os.getenv("APP_LOG_FILE", ""))

FILES_NAMES = ["product_info_1.md", "product_info_2.md"]


async def _upload_files_and_build_vector_store(ai_client: AIProjectClient) -> VectorStore:
    """
    Upload each file in FILES_NAMES to the AIProjectClient, create a vector store,
    and set the uploaded info in an environment variable.

    :param ai_client: The AIProjectClient used for file uploads and vector store creation.
    :type ai_client: AIProjectClient
    :return: The newly created VectorStore containing these files.
    :rtype: VectorStore
    """
    logger.info("Uploading files for agent resources")

    files: Dict[str, Dict[str, str]] = {}
    for file_name in FILES_NAMES:
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'files', file_name))
        uploaded_file = await ai_client.agents.upload_file_and_poll(file_path=file_path, purpose=FilePurpose.AGENTS)
        files[file_name] = {"id": uploaded_file.id, "path": file_path}

    os.environ["UPLOADED_FILE_MAP"] = json.dumps(files)
    logger.info(f"Set env UPLOADED_FILE_MAP = {os.environ['UPLOADED_FILE_MAP']}")

    vector_store = await ai_client.agents.create_vector_store_and_poll(
        file_ids=[info["id"] for info in files.values()],
        name="sample_store"
    )
    logger.info("Successfully created file store and vector store")

    return vector_store


async def create_agent(ai_client: AIProjectClient) -> Agent:
    """
    Create and return a new Agent, uploading files and creating a vector store along the way.

    :param ai_client: The AIProjectClient used for creating the agent.
    :type ai_client: AIProjectClient
    :return: The newly created Agent.
    :rtype: Agent
    """
    logger.info("Creating new agent with resources")
    vector_store = await _upload_files_and_build_vector_store(ai_client)

    file_search_tool = FileSearchTool(vector_store_ids=[vector_store.id])
    toolset = AsyncToolSet()
    toolset.add(file_search_tool)

    agent = await ai_client.agents.create_agent(
        model=os.environ["AZURE_AI_AGENT_DEPLOYMENT_NAME"],
        name=os.environ["AZURE_AI_AGENT_NAME"],
        instructions="You are helpful assistant",
        toolset=toolset
    )

    logger.info(f"Created agent: {agent.name} (ID: {agent.id})")
    return agent


async def update_agent(agent: Agent, ai_client: AIProjectClient) -> Agent:
    """
    Update an existing Agent with newly uploaded files and a new vector store.

    :param agent: The existing Agent to update.
    :type agent: Agent
    :param ai_client: The AIProjectClient used to perform the update.
    :type ai_client: AIProjectClient
    :return: The updated Agent.
    :rtype: Agent
    """
    logger.info("Updating agent with resources")
    vector_store = await _upload_files_and_build_vector_store(ai_client)

    file_search_tool = FileSearchTool(vector_store_ids=[vector_store.id])
    toolset = AsyncToolSet()
    toolset.add(file_search_tool)

    updated_agent = await ai_client.agents.update_agent(
        agent_id=agent.id,
        model=os.environ["AZURE_AI_AGENT_DEPLOYMENT_NAME"],
        name=os.environ["AZURE_AI_AGENT_NAME"],
        instructions="You are helpful assistant",
        toolset=toolset
    )

    logger.info(f"Updated agent: {updated_agent.name} (ID: {updated_agent.id})")
    return updated_agent


async def initialize_resources() -> None:
    """
    Attempt to retrieve, update, or create an Azure AI Agent resource, and store its ID
    in an environment variable. Raises a RuntimeError on failure.
    """
    try:
        ai_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(exclude_shared_token_cache_credential=True),
            conn_str=os.environ["AZURE_AIPROJECT_CONNECTION_STRING"],
        )

        existing_agent_id = os.environ.get("AZURE_AI_AGENT_ID")
        if existing_agent_id:
            try:
                agent = await ai_client.agents.get_agent(existing_agent_id)
                logger.info(f"Found agent by ID: {agent.id}")
                await update_agent(agent, ai_client)
                return
            except Exception as e:
                logger.warning(
                    f"Could not retrieve agent by AZURE_AI_AGENT_ID = {existing_agent_id}, error: {e}",
                    exc_info=True
                )

        agent_list = await ai_client.agents.list_agents()
        if agent_list.data:
            for agent_object in agent_list.data:
                if agent_object.name == os.environ["AZURE_AI_AGENT_NAME"]:
                    logger.info(f"Found existing agent named '{agent_object.name}', ID: {agent_object.id}")
                    os.environ["AZURE_AI_AGENT_ID"] = agent_object.id
                    await update_agent(agent_object, ai_client)
                    return

        # If no agent found, create a new one
        agent = await create_agent(ai_client)
        os.environ["AZURE_AI_AGENT_ID"] = agent.id

    except Exception as e:
        logger.error("Error creating or updating agent", exc_info=True)
        raise RuntimeError(f"Failed to create or update the agent: {e}")


def on_starting(server) -> None:
    """
    Gunicorn 'on_starting' hook. Runs once before workers launch. Ensures resources are ready.

    :param server: The Gunicorn server instance.
    """
    asyncio.run(initialize_resources())


# Gunicorn configuration
max_requests = 1000
max_requests_jitter = 50
log_file = "-"
bind = "0.0.0.0:50505"

if not os.getenv("RUNNING_IN_PRODUCTION"):
    reload = True

preload_app = True
num_cpus = multiprocessing.cpu_count()
workers = (num_cpus * 2) + 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
