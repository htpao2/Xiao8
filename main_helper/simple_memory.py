import json
import os
import logging
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import messages_from_dict, messages_to_dict, SystemMessage
from config import MODELS_WITH_EXTRA_BODY
from utils.config_manager import get_config_manager
from config.prompts_sys import recent_history_manager_prompt, further_summarize_prompt

# Setup logger
logger = logging.getLogger("SimpleMemory")

class SimpleRecentHistoryManager:
    def __init__(self, max_history_length=10):
        self._config_manager = get_config_manager()
        # Get character data to find where to store logs
        _, _, _, _, self.name_mapping, _, _, _, _, self.recent_log = self._config_manager.get_character_data()
        self.max_history_length = max_history_length
        self.user_histories = {}

        # Load existing histories
        for ln in self.recent_log:
            if os.path.exists(self.recent_log[ln]):
                try:
                    with open(self.recent_log[ln], encoding='utf-8') as f:
                        self.user_histories[ln] = messages_from_dict(json.load(f))
                except Exception as e:
                    logger.error(f"Failed to load history for {ln}: {e}")
                    self.user_histories[ln] = []
            else:
                self.user_histories[ln] = []

    def _get_llm(self):
        """Get LLM instance for summarization"""
        core_config = self._config_manager.get_core_config()
        api_key = core_config.get('OPENROUTER_API_KEY')
        model = core_config.get('SUMMARY_MODEL', 'gpt-4o-mini') # Default fallback

        # Safe check for extra body models
        extra_body = None
        if model in MODELS_WITH_EXTRA_BODY:
            extra_body = {"enable_thinking": False}

        return ChatOpenAI(
            model=model,
            base_url=core_config.get('OPENROUTER_URL'),
            api_key=api_key,
            temperature=0.3,
            extra_body=extra_body
        )

    async def update_history(self, new_messages, lanlan_name):
        # Reload from disk in case other processes modified it
        if os.path.exists(self.recent_log.get(lanlan_name, "")):
            try:
                with open(self.recent_log[lanlan_name], encoding='utf-8') as f:
                    self.user_histories[lanlan_name] = messages_from_dict(json.load(f))
            except Exception:
                pass # Keep memory version if file load fails

        if lanlan_name not in self.user_histories:
            self.user_histories[lanlan_name] = []

        try:
            self.user_histories[lanlan_name].extend(new_messages)

            # Save immediately
            self._save_history(lanlan_name)

            # Compress if too long
            if len(self.user_histories[lanlan_name]) > self.max_history_length:
                logger.info(f"Compressing history for {lanlan_name}...")
                to_compress = self.user_histories[lanlan_name][:-self.max_history_length+1]
                summary_msg, _ = await self.compress_history(to_compress, lanlan_name)

                # Keep summary + recent messages
                self.user_histories[lanlan_name] = [summary_msg] + self.user_histories[lanlan_name][-self.max_history_length+1:]
                self._save_history(lanlan_name)

        except Exception as e:
            logger.error(f"Error when updating history: {e}")

    def _save_history(self, lanlan_name):
        if lanlan_name in self.recent_log and self.recent_log[lanlan_name]:
             with open(self.recent_log[lanlan_name], "w", encoding='utf-8') as f:
                json.dump(messages_to_dict(self.user_histories[lanlan_name]), f, indent=2, ensure_ascii=False)

    async def compress_history(self, messages, lanlan_name):
        # Prepare text for summarization
        name_mapping = self.name_mapping.copy()
        name_mapping['ai'] = lanlan_name

        lines = []
        for msg in messages:
            role = name_mapping.get(getattr(msg, 'type', ''), getattr(msg, 'type', ''))
            content = getattr(msg, 'content', '')

            # Handle content list (e.g. from tool calls or multi-modal)
            if not isinstance(content, str):
                try:
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            parts.append(item.get('text', f"|{item.get('type', '')}|"))
                        else:
                            parts.append(str(item))
                    content = "\n".join(parts)
                except Exception:
                    content = str(content)

            lines.append(f"{role} | {content}")

        messages_text = "\n".join(lines)
        prompt = recent_history_manager_prompt % messages_text

        # Retry logic
        retries = 0
        max_retries = 2
        while retries < max_retries:
            try:
                llm = self._get_llm()
                response = await llm.ainvoke(prompt)
                response_content = response.content

                if isinstance(response_content, list):
                    response_content = str(response_content)

                # Clean markdown json
                if response_content.startswith("```"):
                    response_content = response_content.replace('```json','').replace('```', '')

                summary_json = json.loads(response_content)
                summary = summary_json.get('对话摘要', '')

                if len(summary) > 500:
                    summary = await self.further_compress(summary)

                if summary:
                    return SystemMessage(content=f"先前对话的备忘录: {summary}"), summary
                else:
                    retries += 1
            except Exception as e:
                logger.warning(f"Summarization failed (attempt {retries+1}): {e}")
                retries += 1
                await asyncio.sleep(1)

        return SystemMessage(content="先前对话的备忘录: (摘要生成失败)"), ""

    async def further_compress(self, initial_summary):
        try:
            llm = self._get_llm()
            response = await llm.ainvoke(further_summarize_prompt % initial_summary)
            content = response.content
            if content.startswith("```"):
                content = content.replace('```json', '').replace('```', '')

            data = json.loads(content)
            return data.get('对话摘要', initial_summary[:500])
        except Exception:
            return initial_summary[:500]

    def get_recent_history(self, lanlan_name):
        # Ensure up-to-date
        if os.path.exists(self.recent_log.get(lanlan_name, "")):
             with open(self.recent_log[lanlan_name], encoding='utf-8') as f:
                self.user_histories[lanlan_name] = messages_from_dict(json.load(f))
        return self.user_histories.get(lanlan_name, [])
