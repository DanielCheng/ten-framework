from pydantic import BaseModel
from typing import Literal, Optional, Union, Dict, Any
from ten_ai_base.types import LLMToolMetadata


# ==== Base Event ====

class AgentEventBase(BaseModel):
    """Base class for all agent-level events."""
    type: Literal["cmd", "data"]
    name: str


# ==== CMD Events ====

class UserJoinedEvent(AgentEventBase):
    """Event triggered when a user joins the session."""
    type: Literal["cmd"] = "cmd"
    name: Literal["on_user_joined"] = "on_user_joined"

class UserLeftEvent(AgentEventBase):
    """Event triggered when a user leaves the session."""
    type: Literal["cmd"] = "cmd"
    name: Literal["on_user_left"] = "on_user_left"

class ToolRegisterEvent(AgentEventBase):
    """Event triggered when a tool is registered by the user."""
    type: Literal["cmd"] = "cmd"
    name: Literal["tool_register"] = "tool_register"
    tool: LLMToolMetadata
    source: str


# ==== DATA Events ====

class MLLMRequestTranscriptEvent(AgentEventBase):
    """Event triggered when MLLM request transcript is received (partial or final)."""
    type: Literal["data"] = "data"
    name: Literal["mllm_request_transcript"] = "mllm_request_transcript"
    content: Optional[str] = None
    delta: Optional[str] = None
    final: bool
    metadata: Dict[str, Any]


class MLLMResponseTranscriptEvent(AgentEventBase):
    """Event triggered when LLM returns a streaming response."""
    type: Literal["llm"] = "llm"
    name: Literal["llm_response"] = "llm_response"
    delta: str
    content: str
    is_final: bool

# ==== Unified Event Union ====

AgentEvent = Union[
    UserJoinedEvent,
    UserLeftEvent,
    ToolRegisterEvent,
    MLLMRequestTranscriptEvent,
    MLLMResponseTranscriptEvent
]
