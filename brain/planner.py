import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import MODELS_WITH_EXTRA_BODY
from utils.config_manager import get_config_manager
from .computer_use import ComputerUseAdapter

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Task:
    id: str
    title: str
    original_query: str
    server_id: Optional[str] = None
    steps: List[str] = field(default_factory=list)
    status: str = "queued"  # queued | running | done | failed
    meta: Dict[str, Any] = field(default_factory=dict)


class TaskPlanner:
    """
    Planner module: preloads server capabilities, judges executability, decomposes task into executable queries.
    """
    def __init__(self, computer_use: Optional[ComputerUseAdapter] = None):
        self.task_pool: Dict[str, Task] = {}
        self.computer_use = computer_use or ComputerUseAdapter()
        self._config_manager = get_config_manager()
    
    def _get_llm(self):
        """动态获取LLM实例以支持配置热重载"""
        core_config = self._config_manager.get_core_config()
        return ChatOpenAI(model=core_config['SUMMARY_MODEL'], base_url=core_config['OPENROUTER_URL'], api_key=core_config['OPENROUTER_API_KEY'], temperature=0, extra_body={"enable_thinking": False} if core_config['SUMMARY_MODEL'] in MODELS_WITH_EXTRA_BODY else None)

    async def refresh_capabilities(self, force_refresh: bool = True) -> Dict[str, Dict[str, Any]]:
        return {}

    async def assess_and_plan(self, task_id: str, query: str, register: bool = True) -> Task:
        cu_decision = None
        cu = self.computer_use.is_available()

        if cu.get('ready'):
            cu_system = (
                "You are deciding whether a GUI computer-use agent that can control mouse/keyboard, open/close"
                " apps, browse the web, and interact with typical Windows UI can accomplish the task."
                " Output strict JSON:"
                " {use_computer: bool, reason: string}"
            )
            cu_user = f"Task: {query}"
            llm = self._get_llm()
            resp2 = await llm.ainvoke([
                {"role": "system", "content": cu_system},
                {"role": "user", "content": cu_user},
            ])
            text2 = resp2.content.strip()
            import json, uuid
            try:
                if text2.startswith("```"):
                    text2 = text2.replace("```json", "").replace("```", "").strip()
                cu_decision = json.loads(text2)
            except Exception:
                cu_decision = {"use_computer": False, "reason": "LLM parse error"}
        else:
            cu_decision = {"use_computer": False, "reason": "ComputerUse not ready"}

        status = "queued" if cu_decision and cu_decision.get('use_computer') else "failed"

        t = Task(
            id=task_id or str(uuid.uuid4()),
            title=query[:50],
            original_query=query,
            status=status,
            meta={
                "computer_use_decision": cu_decision
            },
        )
        if register:
            self.task_pool[t.id] = t
        return t
