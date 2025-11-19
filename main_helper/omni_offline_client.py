import asyncio
import json
import httpx
import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import base64

logger = logging.getLogger(__name__)

class OmniOfflineClient:
    def __init__(self, base_url, api_key, model, vision_model, on_text_delta, on_input_transcript, on_output_transcript, on_connection_error, on_response_done):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.vision_model = vision_model
        self.on_text_delta = on_text_delta
        self.on_input_transcript = on_input_transcript
        self.on_output_transcript = on_output_transcript
        self.on_connection_error = on_connection_error
        self.on_response_done = on_response_done
        self._conversation_history = []
        self._is_responding = False
        self.current_request_task = None
        self.image_cache = []

    async def connect(self, initial_prompt, **kwargs):
        self._conversation_history.append(SystemMessage(content=initial_prompt))

    async def close(self):
        if self.current_request_task:
            self.current_request_task.cancel()
        self._conversation_history = []

    async def stream_image(self, image_b64):
        self.image_cache.append(image_b64)

    async def stream_text(self, text):
        if self._is_responding:
            logger.warning("Already responding, ignoring new text stream.")
            return

        self._is_responding = True
        
        try:
            user_message_content = [{"type": "text", "text": text}]
            if self.image_cache:
                for img_b64 in self.image_cache:
                    user_message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
                self.model = self.vision_model
            
            self._conversation_history.append(HumanMessage(content=user_message_content))
            self.image_cache = []

            messages = [msg.dict() for msg in self._conversation_history]
            
            # 兼容旧版langchain
            for msg in messages:
                if msg.get('type') == 'human':
                    msg['role'] = 'user'
                elif msg.get('type') == 'ai':
                    msg['role'] = 'assistant'
                
                # 删除 "type" 键（如果存在），因为它不是 OpenAI API 的一部分
                msg.pop('type', None)


            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
            }

            full_response = ""
            is_first_chunk = True

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", self.base_url, headers={"Authorization": f"Bearer {self.api_key}"}, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        await self.on_connection_error(f"VCP connection failed: {response.status_code} {error_text.decode()}")
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            line = line[6:]
                            if line.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(line)
                                if chunk['choices'][0]['delta']['content']:
                                    content = chunk['choices'][0]['delta']['content']
                                    full_response += content
                                    await self.on_text_delta(content, is_first_chunk)
                                    is_first_chunk = False
                            except json.JSONDecodeError:
                                logger.error(f"Failed to decode JSON chunk: {line}")
                                continue
            
            self._conversation_history.append(AIMessage(content=full_response))
            await self.on_response_done()

        except asyncio.CancelledError:
            logger.info("Text streaming task was cancelled.")
        except Exception as e:
            logger.error(f"Error streaming text: {e}")
            await self.on_connection_error(str(e))
        finally:
            self._is_responding = False

    async def handle_messages(self):
        # This client doesn't need a separate message handler loop
        pass
