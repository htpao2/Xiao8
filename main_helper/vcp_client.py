import asyncio
import json
import logging
import aiohttp
import time
from typing import Optional, Callable, Dict, Any, Awaitable, List

logger = logging.getLogger(__name__)

class VCPClient:
    """
    Client for interacting with the VCPToolBox API.
    Acts as a proxy layer for text chat and handles simple audio/video bridging if needed.
    """
    def __init__(
        self,
        base_url: str,
        api_key: str = None,
        model: str = None,
        on_text_delta: Optional[Callable[[str, bool], Awaitable[None]]] = None,
        on_audio_delta: Optional[Callable[[bytes], Awaitable[None]]] = None,
        on_response_done: Optional[Callable[[], Awaitable[None]]] = None,
        on_connection_error: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model

        # Callbacks
        self.on_text_delta = on_text_delta
        self.on_audio_delta = on_audio_delta
        self.on_response_done = on_response_done
        self.on_connection_error = on_connection_error

        self._is_responding = False
        self._cancel_event = asyncio.Event()

    async def chat_stream(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None):
        """
        Send a chat request to VCP and stream the response.
        """
        self._is_responding = True
        self._cancel_event.clear()

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }
        # Only add tools if provided and not empty
        if tools:
            payload["tools"] = tools

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"VCP API Error {response.status}: {error_text}")
                        if self.on_connection_error:
                            await self.on_connection_error(f"VCP Error {response.status}: {error_text}")
                        return

                    is_first_chunk = True
                    async for line in response.content:
                        if self._cancel_event.is_set():
                            logger.info("Response cancelled by user.")
                            break

                        line = line.decode('utf-8').strip()
                        if not line or line == 'data: [DONE]':
                            continue

                        if line.startswith('data: '):
                            json_str = line[6:]
                            try:
                                chunk = json.loads(json_str)
                                if not chunk.get('choices'):
                                    continue

                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')

                                if content:
                                    if self.on_text_delta:
                                        await self.on_text_delta(content, is_first_chunk)
                                    is_first_chunk = False

                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON chunk: {json_str}")
                                continue

        except Exception as e:
            logger.error(f"Error in chat_stream: {e}")
            if self.on_connection_error:
                await self.on_connection_error(str(e))
        finally:
            self._is_responding = False
            if self.on_response_done:
                await self.on_response_done()

    async def cancel_response(self):
        """Cancel the current streaming response."""
        self._cancel_event.set()

    async def bridge_audio_input(self, text_input: str):
        """
        Bridge method: Accept text (converted from audio locally) and send to VCP.
        In the future, this could send audio bytes directly if VCP supports it.
        For now, we assume the caller (SessionManager) handles STT.
        """
        pass # This is just a placeholder, logic is in chat_stream
