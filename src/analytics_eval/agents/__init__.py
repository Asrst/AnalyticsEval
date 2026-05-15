"""Agent interface for BYOA (Bring Your Own Agent) evaluation.

Available agents:
- AgentInterface: Abstract interface — implement this to evaluate your agent
- LLMSQLAgent: Concrete agent that generates SQL via LLM API
"""

from analytics_eval.agents.base import AgentInterface, AgentResponse
from analytics_eval.agents.llm_sql_agent import LLMSQLAgent

__all__ = ["AgentInterface", "AgentResponse", "LLMSQLAgent"]
