# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
import contextlib
import os
import sys
from typing import AsyncIterator

from azure.ai.projects.aio import AIProjectClient
from azure.identity import DefaultAzureCredential
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from logging_config import configure_logging

logger = configure_logging(os.getenv("APP_LOG_FILE", ""))

enable_trace_string = os.getenv("ENABLE_AZURE_MONITOR_TRACING", "")
enable_trace = str(enable_trace_string).lower() == "true"

if enable_trace:
    logger.info("Tracing is enabled.")
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
    except ModuleNotFoundError:
        logger.error("Telemetry libraries not installed. Please install azure-monitor-opentelemetry.")
        sys.exit(1)
else:
    logger.info("Tracing is not enabled.")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Perform startup (loading environment, retrieving agent) and teardown (closing AIProjectClient).

    :param app: The FastAPI application instance.
    :type app: FastAPI
    :yield: Nothing, but ensures resources are cleaned up afterward.
    :rtype: AsyncIterator[None]
    """
    agent = None
    try:
        if not os.getenv("RUNNING_IN_PRODUCTION"):
            logger.info("Loading .env file")
            load_dotenv(override=True)

        ai_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(exclude_shared_token_cache_credential=True),
            conn_str=os.environ["AZURE_AIPROJECT_CONNECTION_STRING"],
        )
        logger.info("Created AIProjectClient")

        if enable_trace:
            try:
                application_insights_conn_str = await ai_client.telemetry.get_connection_string()
                if not application_insights_conn_str:
                    logger.warning("App Insights not enabled. Enable it in your AI Foundry project page.")
                    sys.exit(1)
                else:
                    configure_azure_monitor(connection_string=application_insights_conn_str)
            except Exception as e:
                logger.error("Failed to retrieve App Insights connection string.", exc_info=True)
                sys.exit(1)

        agent_id = os.environ.get("AZURE_AI_AGENT_ID")
        if agent_id:
            try:
                logger.info(f"Fetching agent by ID: {agent_id}")
                agent = await ai_client.agents.get_agent(agent_id)
                logger.info(f"Agent found: Name={agent.name}, ID={agent.id}")
            except Exception as e:
                logger.error(f"Error fetching agent: {e}", exc_info=True)

        if not agent:
            # Fallback to searching by name
            agent_name = os.environ["AZURE_AI_AGENT_NAME"]
            agent_list = await ai_client.agents.list_agents()
            if agent_list.data:
                for a_obj in agent_list.data:
                    if a_obj.name == agent_name:
                        agent = a_obj
                        logger.info(f"Found agent by name '{agent_name}', ID={a_obj.id}")
                        break

        if not agent:
            raise RuntimeError("No agent found. Ensure qunicorn.conf.py created one or set AZURE_AI_AGENT_ID.")

        app.state.ai_client = ai_client
        app.state.agent = agent

        yield

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise RuntimeError(f"Error during startup: {e}")

    finally:
        try:
            await ai_client.close()
            logger.info("Closed AIProjectClient")
        except Exception as close_error:
            logger.error("Error closing AIProjectClient", exc_info=close_error)


def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.

    :return: Configured FastAPI application instance.
    :rtype: FastAPI
    """
    directory = os.path.join(os.path.dirname(__file__), "static")
    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=directory), name="static")

    from . import routes  # Import routes from this package
    app.include_router(routes.router)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Global exception handler returning a 500 for unhandled errors.

        :param request: The HTTP request that raised the exception.
        :type request: Request
        :param exc: The unhandled exception.
        :type exc: Exception
        :return: 500 JSONResponse with generic error detail.
        :rtype: JSONResponse
        """
        logger.error("Unhandled exception occurred", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

    return app