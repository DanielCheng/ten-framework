import asyncio
import json
from ten_ai_base.struct import MLLMRequestTranscript, MLLMResponseTranscript
from ten_runtime import AsyncTenEnv, Cmd, CmdResult, Data, StatusCode
from ten_ai_base.types import LLMToolMetadata
from .events import *

class Agent:
    def __init__(self, ten_env: AsyncTenEnv):
        self.ten_env: AsyncTenEnv = ten_env
        self.stopped = False
        self.event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

    async def on_cmd(self, cmd: Cmd):
        cmd_name = cmd.get_name()
        try:
            if cmd_name == "on_user_joined":
                event = UserJoinedEvent()
            elif cmd_name == "on_user_left":
                event = UserLeftEvent()
            elif cmd_name == "tool_register":
                tool_json, err = cmd.get_property_to_json("tool")
                if err:
                    raise RuntimeError(f"Invalid tool metadata: {err}")
                tool = LLMToolMetadata.model_validate_json(tool_json)
                event = ToolRegisterEvent(tool=tool, source=cmd.get_source().extension_name)
            else:
                self.ten_env.log_warn(f"Unhandled cmd: {cmd_name}")
                return

            await self.event_queue.put(event)
            await self.ten_env.return_result(CmdResult.create(StatusCode.OK, cmd))

        except Exception as e:
            self.ten_env.log_error(f"on_cmd error: {e}")
            await self.ten_env.return_result(CmdResult.create(StatusCode.ERROR, cmd))

    async def on_data(self, data: Data):
        data_name = data.get_name()
        self.ten_env.log_info(f"on_data: {data_name}")
        try:
            if data_name == "mllm_request_transcript":
                transcript_json, _ = data.get_property_to_json(None)
                transcript = MLLMRequestTranscript.model_validate_json(transcript_json)
                event = MLLMRequestTranscriptEvent(
                    delta=transcript.delta,
                    content=transcript.content,
                    metadata=transcript.metadata,
                    final=transcript.final,
                )
                await self.event_queue.put(event)
            elif data_name == "mllm_response_transcript":
                response_json, _ = data.get_property_to_json(None)
                response = MLLMResponseTranscript.model_validate_json(response_json)
                event = MLLMResponseTranscriptEvent(
                    delta=response.delta,
                    content=response.content,
                    metadata=response.metadata,
                    is_final=response.final,
                )
                await self.event_queue.put(event)
            else:
                self.ten_env.log_warn(f"Unhandled data: {data_name}")

        except Exception as e:
            self.ten_env.log_error(f"on_data error: {e}")

    async def get_event(self) -> AgentEvent:
        return await self.event_queue.get()


    async def register_llm_tool(self, tool: LLMToolMetadata, source: str):
        """
        Register tools with the LLM.
        This method sends a command to register the provided tools.
        """
        pass
        # await self.llm_exec.register_tool(tool, source)

    async def queue_llm_input(self, text: str):
        """
        Queue a new message to the LLM context.
        This method sends the text input to the LLM for processing.
        """
        pass
        # await self.llm_exec.queue_input(text)

    async def flush_llm(self):
        """
        Flush the LLM input queue.
        This will ensure that all queued inputs are processed.
        """
        pass
        # await self.llm_exec.flush()

    async def stop(self):
        """
        Stop the agent processing.
        This will stop the event queue and any ongoing tasks.
        """
        self.stopped = True
        # await self.llm_exec.stop()
        await self.event_queue.put(None)