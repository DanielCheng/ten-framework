#
#
# Agora Real Time Engagement
# Created by Wei Hu in 2024-08.
# Copyright (c) 2024 Agora IO. All rights reserved.
#
#
import asyncio
import base64
import json
from enum import Enum
import traceback
import time
from typing import Iterable, Literal

from ten_ai_base.mllm import AsyncMLLMBaseExtension
from ten_ai_base.struct import MLLMRequestTranscript, MLLMResponseTranscript
from ten_runtime import (
    AudioFrame,
    AsyncTenEnv,
    Cmd,
    StatusCode,
    Data,
)
from ten_runtime.audio_frame import AudioFrameDataFmt
from ten_ai_base.const import CMD_PROPERTY_RESULT, CMD_TOOL_CALL
from dataclasses import dataclass
from ten_ai_base.config import BaseConfig
from ten_ai_base.types import (
    LLMToolMetadata,
    LLMToolResult,
    LLMChatCompletionContentPartParam,
)
from .realtime.connection import RealtimeApiConnection
from .realtime.struct import (
    ItemCreate,
    ItemInputAudioTranscriptionDelta,
    SessionCreated,
    ItemCreated,
    SessionUpdated,
    UserMessageItemParam,
    ItemInputAudioTranscriptionCompleted,
    ItemInputAudioTranscriptionFailed,
    ResponseCreated,
    ResponseDone,
    ResponseAudioTranscriptDelta,
    ResponseTextDelta,
    ResponseAudioTranscriptDone,
    ResponseTextDone,
    ResponseOutputItemDone,
    ResponseOutputItemAdded,
    ResponseAudioDelta,
    ResponseAudioDone,
    InputAudioBufferSpeechStarted,
    InputAudioBufferSpeechStopped,
    ResponseFunctionCallArgumentsDone,
    ErrorMessage,
    ItemTruncate,
    SessionUpdate,
    SessionUpdateParams,
    InputAudioTranscription,
    ContentType,
    FunctionCallOutputItemParam,
    ResponseCreate,
    ServerVADUpdateParams,
    SemanticVADUpdateParams,
)

CMD_IN_FLUSH = "flush"
CMD_IN_ON_USER_JOINED = "on_user_joined"
CMD_IN_ON_USER_LEFT = "on_user_left"
CMD_OUT_FLUSH = "flush"


class Role(str, Enum):
    User = "user"
    Assistant = "assistant"


@dataclass
class OpenAIRealtimeConfig(BaseConfig):
    base_uri: str = "wss://api.openai.com"
    api_key: str = ""
    path: str = "/v1/realtime"
    model: str = "gpt-4o-realtime-preview"
    language: str = "en-US"
    prompt: str = ""
    temperature: float = 0.5
    max_tokens: int = 1024
    voice: str = "alloy"
    server_vad: bool = True
    audio_out: bool = True
    input_transcript: bool = True
    sample_rate: int = 24000
    vad_type: Literal["server_vad", "semantic_vad"] = "server_vad"
    vad_eagerness: Literal["low", "medium", "high", "auto"] = "auto"
    vad_threshold: float = 0.5
    vad_prefix_padding_ms: int = 300
    vad_silence_duration_ms: int = 500
    vendor: str = ""
    stream_id: int = 0
    dump: bool = False
    greeting: str = ""
    max_history: int = 20
    enable_storage: bool = False

    def build_ctx(self) -> dict:
        return {
            "language": self.language,
            "model": self.model,
        }


class OpenAIRealtime2Extension(AsyncMLLMBaseExtension):

    def __init__(self, name: str):
        super().__init__(name)
        self.ten_env: AsyncTenEnv = None
        self.conn = None
        self.openai_session = None
        self.openai_session_id = None

        self.config: OpenAIRealtimeConfig = None
        self.stopped: bool = False
        self.connected: bool = False

        self.request_transcript: str = ""
        self.response_transcript: str = ""
        self.available_tools: list[LLMToolMetadata] = []

    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        await super().on_init(ten_env)
        ten_env.log_debug("on_init")
        self.ten_env = ten_env

        self.loop = asyncio.get_event_loop()

        self.config = await OpenAIRealtimeConfig.create_async(ten_env=ten_env)
        ten_env.log_info(f"config: {self.config}")

        if not self.config.api_key:
            ten_env.log_error("api_key is required")
            raise ValueError("api_key is required")

        self.conn = RealtimeApiConnection(
            ten_env=ten_env,
            base_uri=self.config.base_uri,
            path=self.config.path,
            api_key=self.config.api_key,
            model=self.config.model,
            vendor=self.config.vendor,
        )

    async def start_connection(self) -> None:
        try:
            await self.conn.connect()
            item_id = ""  # For truncate
            response_id = ""
            content_index = 0
            flushed = set()
            session_start_ms = int(
                time.time() * 1000
            )  # Use proper timestamp in milliseconds

            self.ten_env.log_info("Client loop started")
            async for message in self.conn.listen():
                try:
                    # self.ten_env.log_info(f"Received message: {message.type}")
                    match message:
                        case SessionCreated():
                            self.ten_env.log_info(
                                f"Session is created: {message.session}"
                            )
                            self.openai_session_id = message.session.id
                            self.openai_session = message.session
                            await self._update_session()
                        case SessionUpdated():
                            self.ten_env.log_info(
                                f"Session is updated: {message.session}"
                            )
                            self.connected = True
                        case ItemInputAudioTranscriptionDelta():
                            self.ten_env.log_info(
                                f"On request transcript delta {message.item_id} {message.content_index}"
                            )
                            self.request_transcript += message.transcript
                            await self.send_mllm_request_transcript(MLLMRequestTranscript(
                                content=self.request_transcript,
                                delta=message.transcript,
                                final=False,
                                metadata={
                                    "session_id": self.session_id if self.session_id else "-1",
                                }
                            ))
                        case ItemInputAudioTranscriptionCompleted():
                            self.ten_env.log_info(
                                f"On request transcript {message.transcript}"
                            )
                            await self.send_mllm_request_transcript(MLLMRequestTranscript(
                                content=self.request_transcript,
                                delta=message.transcript,
                                final=True,
                                metadata={
                                    "session_id": self.session_id if self.session_id else "-1",
                                }
                            ))
                        case ItemInputAudioTranscriptionFailed():
                            self.ten_env.log_warn(
                                f"On request transcript failed {message.item_id} {message.error}"
                            )
                        case ItemCreated():
                            self.ten_env.log_info(
                                f"On item created {message.item}"
                            )
                        case ResponseCreated():
                            response_id = message.response.id
                            self.ten_env.log_info(
                                f"On response created {response_id}"
                            )
                        case ResponseDone():
                            msg_resp_id = message.response.id
                            status = message.response.status
                            if msg_resp_id == response_id:
                                response_id = ""
                            self.ten_env.log_info(
                                f"On response done {msg_resp_id} {status} {message.response.usage}"
                            )
                            if message.response.usage:
                                pass
                                # await self._update_usage(message.response.usage)
                        case ResponseAudioTranscriptDelta():
                            self.ten_env.log_info(
                                f"On response transcript delta {message.response_id} {message.output_index} {message.content_index} {message.delta}"
                            )
                            if message.response_id in flushed:
                                self.ten_env.log_warn(
                                    f"On flushed transcript delta {message.response_id} {message.output_index} {message.content_index} {message.delta}"
                                )
                                continue

                            self.response_transcript += message.delta
                            await self.send_mllm_response_text(MLLMResponseTranscript(
                                content=self.response_transcript,
                                delta=message.delta,
                                final=False,
                                metadata={
                                    "session_id": self.session_id if self.session_id else "-1",
                                }
                            ))
                        case ResponseTextDelta():
                            self.ten_env.log_info(
                                f"On response text delta {message.response_id} {message.output_index} {message.content_index} {message.delta}"
                            )
                            if message.response_id in flushed:
                                self.ten_env.log_warn(
                                    f"On flushed text delta {message.response_id} {message.output_index} {message.content_index} {message.delta}"
                                )
                                continue
                            if item_id != message.item_id:
                                item_id = message.item_id

                            self.response_transcript += message.delta
                            await self.send_mllm_response_text(MLLMResponseTranscript(
                                content=self.response_transcript,
                                delta=message.delta,
                                final=False,
                                metadata={
                                    "session_id": self.session_id if self.session_id else "-1",
                                }
                            ))
                        case ResponseAudioTranscriptDone():
                            self.ten_env.log_info(
                                f"On response transcript done {message.output_index} {message.content_index} {message.transcript}"
                            )
                            if message.response_id in flushed:
                                self.ten_env.log_warn(
                                    f"On flushed transcript done {message.response_id}"
                                )
                                continue
                            await self.send_mllm_response_text(MLLMResponseTranscript(
                                content=self.response_transcript,
                                delta=None,
                                final=True,
                                metadata={
                                    "session_id": self.session_id if self.session_id else "-1",
                                }
                            ))
                            self.response_transcript = ""
                        case ResponseTextDone():
                            self.ten_env.log_info(
                                f"On response text done {message.output_index} {message.content_index} {message.text}"
                            )
                            if message.response_id in flushed:
                                self.ten_env.log_warn(
                                    f"On flushed text done {message.response_id}"
                                )
                                continue
                            await self.send_mllm_response_text(MLLMResponseTranscript(
                                content=self.response_transcript,
                                delta=None,
                                final=True,
                                metadata={
                                    "session_id": self.session_id if self.session_id else "-1",
                                }
                            ))
                            self.response_transcript = ""
                        case ResponseOutputItemDone():
                            self.ten_env.log_info(
                                f"Output item done {message.item}"
                            )
                        case ResponseOutputItemAdded():
                            self.ten_env.log_info(
                                f"Output item added {message.output_index} {message.item}"
                            )
                        case ResponseAudioDelta():
                            if message.response_id in flushed:
                                self.ten_env.log_warn(
                                    f"On flushed audio delta {message.response_id} {message.item_id} {message.content_index}"
                                )
                                continue
                            if item_id != message.item_id:
                                item_id = message.item_id
                            content_index = message.content_index
                            audio_data = base64.b64decode(message.delta)
                            await self.send_mllm_response_audio_data(audio_data)
                        case ResponseAudioDone():
                            pass
                        case InputAudioBufferSpeechStarted():
                            self.ten_env.log_info(
                                f"On server listening, in response {response_id}, last item {item_id}"
                            )
                            # Calculate proper truncation time - elapsed milliseconds since session start
                            current_ms = int(time.time() * 1000)
                            end_ms = current_ms - session_start_ms
                            if (
                                item_id and end_ms > 0
                            ):  # Only truncate if we have a valid positive timestamp
                                truncate = ItemTruncate(
                                    item_id=item_id,
                                    content_index=content_index,
                                    audio_end_ms=end_ms,
                                )
                                await self.conn.send_request(truncate)
                            if self.config.server_vad:
                                await self._flush()
                            if response_id and self.response_transcript:
                                transcript = self.response_transcript + "[interrupted]"
                                await self.send_mllm_response_text(MLLMResponseTranscript(
                                    content=transcript,
                                    delta=None,
                                    final=True,
                                    metadata={
                                        "session_id": self.session_id if self.session_id else "-1",
                                    }
                                ))
                                self.response_transcript = ""
                                # memory leak, change to lru later
                                flushed.add(response_id)
                            item_id = ""
                        case InputAudioBufferSpeechStopped():
                            # Only for server vad
                            self.input_end = time.time()
                            # Update session start to properly track relative timing
                            session_start_ms = (
                                int(time.time() * 1000) - message.audio_end_ms
                            )
                            self.ten_env.log_info(
                                f"On server stop listening, audio_end_ms: {message.audio_end_ms}, session_start_ms updated to: {session_start_ms}"
                            )
                        case ResponseFunctionCallArgumentsDone():
                            tool_call_id = message.call_id
                            name = message.name
                            arguments = message.arguments
                            self.ten_env.log_info(f"need to call func {name}")
                            self.loop.create_task(
                                self._handle_tool_call(
                                    tool_call_id, name, arguments
                                )
                            )
                        case ErrorMessage():
                            self.ten_env.log_error(
                                f"Error message received: {message.error}"
                            )
                        case _:
                            self.ten_env.log_debug(
                                f"Not handled message {message}"
                            )
                except Exception as e:
                    traceback.print_exc()
                    self.ten_env.log_error(
                        f"Error processing message: {message} {e}"
                    )

            self.ten_env.log_info("Client loop finished")
        except Exception as e:
            traceback.print_exc()
            self.ten_env.log_error(f"Failed to handle loop {e}")

        # clear so that new session can be triggered
        self.connected = False
        self.remote_stream_id = 0

        if not self.stopped:
            await self.conn.close()
            await asyncio.sleep(0.5)
            self.ten_env.log_info("Reconnect")

            self.conn = RealtimeApiConnection(
                ten_env=self.ten_env,
                base_uri=self.config.base_uri,
                path=self.config.path,
                api_key=self.config.api_key,
                model=self.config.model,
                vendor=self.config.vendor,
            )

            self.loop.create_task(self._loop())
        # except Exception as e:
        #     traceback.print_exc()
        #     self.ten_env.log_error(f"Failed to init client {e}")

    async def stop_connection(self) -> None:
        await self.conn.close()
        self.stopped = True

    def is_connected(self) -> bool:
        return self.connected

    async def send_audio(self, frame: AudioFrame, session_id: str | None) -> bool:
        self.session_id = session_id
        await self.conn.send_audio_data(frame.get_buf())
        return True

    async def on_data(self, ten_env: AsyncTenEnv, data: Data) -> None:
        pass

    async def _update_session(self) -> None:
        tools = []

        def tool_dict(tool: LLMToolMetadata):
            t = {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            }

            for param in tool.parameters:
                t["parameters"]["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    t["parameters"]["required"].append(param.name)

            return t

        if self.available_tools:
            tools = [tool_dict(t) for t in self.available_tools]
        prompt = self.config.prompt

        self.ten_env.log_info(f"update session {prompt} {tools}")
        if self.config.vad_type == "server_vad":
            vad_params = ServerVADUpdateParams(
                threshold=self.config.vad_threshold,
                prefix_padding_ms=self.config.vad_prefix_padding_ms,
                silence_duration_ms=self.config.vad_silence_duration_ms,
            )
        else:  # semantic vad
            vad_params = SemanticVADUpdateParams(
                eagerness=self.config.vad_eagerness,
            )
        su = SessionUpdate(
            session=SessionUpdateParams(
                instructions=prompt,
                model=self.config.model,
                tool_choice="auto" if self.available_tools else "none",
                tools=tools,
                turn_detection=vad_params,
            )
        )
        if self.config.audio_out:
            su.session.voice = self.config.voice
        else:
            su.session.modalities = ["text"]

        if self.config.input_transcript:
            su.session.input_audio_transcription = InputAudioTranscription(
                model="whisper-1"
            )
        await self.conn.send_request(su)

    async def _handle_tool_call(
        self, tool_call_id: str, name: str, arguments: str
    ) -> None:
        self.ten_env.log_info(
            f"_handle_tool_call {tool_call_id} {name} {arguments}"
        )
        cmd: Cmd = Cmd.create(CMD_TOOL_CALL)
        cmd.set_property_string("name", name)
        cmd.set_property_from_json("arguments", arguments)
        [result, _] = await self.ten_env.send_cmd(cmd)

        tool_response = ItemCreate(
            item=FunctionCallOutputItemParam(
                call_id=tool_call_id,
                output='{"success":false}',
            )
        )
        if result.get_status_code() == StatusCode.OK:
            r, _ = result.get_property_to_json(CMD_PROPERTY_RESULT)
            tool_result: LLMToolResult = json.loads(r)

            result_content = tool_result["content"]
            tool_response.item.output = json.dumps(
                self._convert_to_content_parts(result_content)
            )
            self.ten_env.log_info(f"tool_result: {tool_call_id} {tool_result}")
        else:
            self.ten_env.log_error("Tool call failed")

        await self.conn.send_request(tool_response)
        await self.conn.send_request(ResponseCreate())
        self.ten_env.log_info(f"_remote_tool_call finish {name} {arguments}")

    def _greeting_text(self) -> str:
        text = "Hi, there."
        if self.config.language == "zh-CN":
            text = "你好。"
        elif self.config.language == "ja-JP":
            text = "こんにちは"
        elif self.config.language == "ko-KR":
            text = "안녕하세요"
        return text

    def _convert_tool_params_to_dict(self, tool: LLMToolMetadata):
        json_dict = {"type": "object", "properties": {}, "required": []}

        for param in tool.parameters:
            json_dict["properties"][param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.required:
                json_dict["required"].append(param.name)

        return json_dict

    def _convert_to_content_parts(
        self, content: Iterable[LLMChatCompletionContentPartParam]
    ):
        content_parts = []

        if isinstance(content, str):
            content_parts.append({"type": "text", "text": content})
        else:
            for part in content:
                # Only text content is supported currently for v2v model
                if part["type"] == "text":
                    content_parts.append(part)
        return content_parts

    async def _greeting(self) -> None:
        if self.connected and self.users_count == 1:
            text = self._greeting_text()
            if self.config.greeting:
                text = "Say '" + self.config.greeting + "' to me."
            self.ten_env.log_info(f"send greeting {text}")
            await self.conn.send_request(
                ItemCreate(
                    item=UserMessageItemParam(
                        content=[{"type": ContentType.InputText, "text": text}]
                    )
                )
            )
            await self.conn.send_request(ResponseCreate())

    async def _flush(self) -> None:
        try:
            c = Cmd.create("flush")
            await self.ten_env.send_cmd(c)
        except Exception:
            self.ten_env.log_error("Error flush")
