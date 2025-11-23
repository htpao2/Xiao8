import asyncio
import json
import struct  # For packing audio data
import threading
import re
import requests
import logging
import time
from datetime import datetime
from websockets import exceptions as web_exceptions
from fastapi import WebSocket, WebSocketDisconnect
from utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, \
    is_only_punctuation, split_paragraph
from utils.audio import make_wav_header
from main_helper.omni_realtime_client import OmniRealtimeClient
from main_helper.omni_offline_client import OmniOfflineClient
from main_helper.vcp_client import VCPClient
from main_helper.tts_helper import get_tts_worker
import inflect
import base64
from io import BytesIO
from PIL import Image
from utils.config_manager import get_config_manager
from multiprocessing import Process, Queue as MPQueue
from uuid import uuid4
import numpy as np
from librosa import resample
import httpx 
from main_helper.simple_memory import SimpleRecentHistoryManager

# Setup logger for this module
logger = logging.getLogger(__name__)

# --- 一个带有定期上下文压缩+在线热切换的语音会话管理器 ---
class LLMSessionManager:
    def __init__(self, sync_message_queue, lanlan_name, lanlan_prompt):
        self.websocket = None
        self.sync_message_queue = sync_message_queue
        self.session = None
        self.last_time = None
        self.is_active = False
        self.active_session_is_idle = False
        self.current_expression = None
        self.tts_request_queue = MPQueue() # TTS request (多进程队列)
        self.tts_response_queue = MPQueue() # TTS response (多进程队列)
        self.tts_process = None  # TTS子进程
        self.lock = asyncio.Lock()  # 使用异步锁替代同步锁
        self.websocket_lock = None  # websocket操作的共享锁，由main_server设置
        self.current_speech_id = None
        self.inflect_parser = inflect.engine()
        self.emoji_pattern = re.compile(r'[^\w\u4e00-\u9fff\s>][^\w\u4e00-\u9fff\s]{2,}[^\w\u4e00-\u9fff\s<]', flags=re.UNICODE)
        self.emoji_pattern2 = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
        self.emotion_pattern = re.compile('<(.*?)>')

        self.lanlan_prompt = lanlan_prompt
        self.lanlan_name = lanlan_name
        # 获取角色相关配置
        self._config_manager = get_config_manager()

        (
            self.master_name,
            self.her_name,
            self.master_basic_config,
            self.lanlan_basic_config,
            self.name_mapping,
            self.lanlan_prompt_map,
            _, _, _, _
        ) = self._config_manager.get_character_data()

        # Simple memory manager
        self.simple_memory = SimpleRecentHistoryManager()

        # 获取API相关配置（动态读取以支持热重载）
        core_config = self._config_manager.get_core_config()
        self.vcp_base_url = core_config.get('VCP_BASE_URL', 'http://localhost:6005')
        self.model = core_config['CORE_MODEL']  # For realtime voice
        self.text_model = core_config['CORRECTION_MODEL']  # For text-only mode
        self.vision_model = core_config['VISION_MODEL']  # For vision tasks
        self.core_url = core_config['CORE_URL']
        self.core_api_key = core_config['CORE_API_KEY']
        self.core_api_type = core_config['CORE_API_TYPE']
        self.openrouter_url = core_config['OPENROUTER_URL']
        self.openrouter_api_key = core_config['OPENROUTER_API_KEY']
        self.audio_api_key = core_config['AUDIO_API_KEY']
        self.voice_id = self.lanlan_basic_config[self.lanlan_name].get('voice_id', '')
        # 注意：use_tts 会在 start_session 中根据 input_mode 重新设置
        self.use_tts = False
        self.generation_config = {}  # Qwen暂时不用
        self.message_cache_for_new_session = []
        self.is_preparing_new_session = False
        self.summary_triggered_time = None
        self.initial_cache_snapshot_len = 0
        self.pending_session_warmed_up_event = None
        self.pending_session_final_prime_complete_event = None
        self.session_start_time = None
        self.pending_connector = None
        self.pending_session = None
        self.is_hot_swap_imminent = False
        self.tts_handler_task = None
        # 热切换相关变量
        self.background_preparation_task = None
        self.final_swap_task = None
        self.receive_task = None
        self.message_handler_task = None
        # 任务完成后的额外回复队列（将在下一次切换时统一汇报）
        self.pending_extra_replies = []
        # 由前端控制的Agent相关开关
        self.agent_flags = {
            'agent_enabled': False,
            'computer_use_enabled': False,
            'mcp_enabled': False,
        }
        
        # 模式标志: 'audio' 或 'text'
        self.input_mode = 'audio'
        
        # 初始化时创建audio模式的session（默认）
        self.session = None
        
        # 防止无限重试的保护机制
        self.session_start_failure_count = 0
        self.session_start_last_failure_time = None
        self.session_start_cooldown_seconds = 3.0  # 冷却时间：3秒
        self.session_start_max_failures = 3  # 最大连续失败次数
        
        # 防止并发启动的标志
        self.is_starting_session = False
        
        # TTS缓存机制：确保不丢包
        self.tts_ready = False  # TTS是否完全就绪
        self.tts_pending_chunks = []  # 待处理的TTS文本chunk: [(speech_id, text), ...]
        self.tts_cache_lock = asyncio.Lock()  # 保护缓存的锁
        
        # 输入数据缓存机制：确保session初始化期间的输入不丢失
        self.session_ready = False  # Session是否完全就绪
        self.pending_input_data = []  # 待处理的输入数据: [message_dict, ...]
        self.input_cache_lock = asyncio.Lock()  # 保护输入缓存的锁

    async def handle_new_message(self):
        """处理新模型输出：清空TTS队列并通知前端"""
        if self.use_tts and self.tts_process and self.tts_process.is_alive():
            # 清空响应队列中待发送的音频数据
            while not self.tts_response_queue.empty():
                try:
                    self.tts_response_queue.get_nowait()
                except:
                    break
            # 发送终止信号以清空TTS请求队列并停止当前合成
            try:
                self.tts_request_queue.put((None, None))
            except Exception as e:
                logger.warning(f"⚠️ 发送TTS中断信号失败: {e}")
        
        # 清空待处理的TTS缓存
        async with self.tts_cache_lock:
            self.tts_pending_chunks.clear()
        
        await self.send_user_activity()

    async def handle_text_data(self, text: str, is_first_chunk: bool = False):
        """文本回调：处理文本显示和TTS（用于文本模式）"""
        # 如果是新消息的第一个chunk，清空TTS队列和缓存以打断之前的语音
        if is_first_chunk and self.use_tts:
            async with self.tts_cache_lock:
                self.tts_pending_chunks.clear()
            
            if self.tts_process and self.tts_process.is_alive():
                # 清空响应队列中待发送的音频数据
                while not self.tts_response_queue.empty():
                    try:
                        self.tts_response_queue.get_nowait()
                    except:
                        break
        
        # 文本模式下，无论是否使用TTS，都要发送文本到前端显示
        await self.send_lanlan_response(text, is_first_chunk)
        
        # 如果配置了TTS，将文本发送到TTS队列或缓存
        if self.use_tts:
            async with self.tts_cache_lock:
                # 检查TTS是否就绪
                if self.tts_ready and self.tts_process and self.tts_process.is_alive():
                    # TTS已就绪，直接发送
                    try:
                        self.tts_request_queue.put((self.current_speech_id, text))
                    except Exception as e:
                        logger.warning(f"⚠️ 发送TTS请求失败: {e}")
                else:
                    # TTS未就绪，先缓存
                    self.tts_pending_chunks.append((self.current_speech_id, text))
                    if len(self.tts_pending_chunks) == 1:
                        logger.info(f"TTS未就绪，开始缓存文本chunk...")

    async def handle_response_complete(self):
        """完成回调"""
        if self.use_tts and self.tts_process and self.tts_process.is_alive():
            print("Response complete")
            try:
                self.tts_request_queue.put((None, None))
            except Exception as e:
                logger.warning(f"⚠️ 发送TTS结束信号失败: {e}")
        self.sync_message_queue.put({'type': 'system', 'data': 'turn end'})
        
        # 直接向前端发送turn end消息
        try:
            if self.websocket and hasattr(self.websocket, 'client_state') and self.websocket.client_state == self.websocket.client_state.CONNECTED:
                await self.websocket.send_json({'type': 'system', 'data': 'turn end'})
        except Exception as e:
            logger.error(f"💥 WS Send Turn End Error: {e}")

        # Update simple memory
        if self.message_cache_for_new_session:
            await self.simple_memory.update_history(self.message_cache_for_new_session, self.lanlan_name)
            self.message_cache_for_new_session = []

    async def handle_audio_data(self, audio_data: bytes):
        """音频回调：推送音频到WebSocket前端 (仅用于 Realtime API)"""
        if not self.use_tts:
            if self.websocket and hasattr(self.websocket, 'client_state') and self.websocket.client_state == self.websocket.client_state.CONNECTED:
                # 这里假设audio_data为PCM16字节流，直接推送
                audio = np.frombuffer(audio_data, dtype=np.int16)
                audio = (resample(audio.astype(np.float32) / 32768.0, orig_sr=24000, target_sr=48000)*32767.).clip(-32768, 32767).astype(np.int16)

                await self.send_speech(audio.tobytes())
            else:
                pass  # websocket未连接时忽略

    async def handle_input_transcript(self, transcript: str):
        """输入转录回调"""
        # 推送到同步消息队列
        self.sync_message_queue.put({"type": "user", "data": {"input_type": "transcript", "data": transcript.strip()}})
        
        # 发送前端显示
        if self.websocket and hasattr(self.websocket, 'client_state') and self.websocket.client_state == self.websocket.client_state.CONNECTED:
            try:
                message = {
                    "type": "user_transcript",
                    "text": transcript.strip()
                }
                await self.websocket.send_json(message)
            except Exception as e:
                logger.error(f"⚠️ 发送用户转录到前端失败: {e}")
        
        # 缓存到session cache
        if not hasattr(self, 'message_cache_for_new_session'):
            self.message_cache_for_new_session = []

        self.message_cache_for_new_session.append({"role": self.master_name, "text": transcript.strip()})

        # 可选：推送用户活动
        async with self.lock:
            self.current_speech_id = str(uuid4())

    async def handle_output_transcript(self, text: str, is_first_chunk: bool = False):
        """输出转录回调：处理文本显示和TTS"""
        # 无论是否使用TTS，都要发送文本到前端显示
        await self.send_lanlan_response(text, is_first_chunk)
        
        # 如果配置了TTS，将文本发送到TTS队列或缓存
        if self.use_tts:
            async with self.tts_cache_lock:
                # 检查TTS是否就绪
                if self.tts_ready and self.tts_process and self.tts_process.is_alive():
                    # TTS已就绪，直接发送
                    try:
                        self.tts_request_queue.put((self.current_speech_id, text))
                    except Exception as e:
                        logger.warning(f"⚠️ 发送TTS请求失败: {e}")
                else:
                    # TTS未就绪，先缓存
                    self.tts_pending_chunks.append((self.current_speech_id, text))
                    if len(self.tts_pending_chunks) == 1:
                        logger.info(f"TTS未就绪，开始缓存文本chunk...")

    async def send_lanlan_response(self, text: str, is_first_chunk: bool = False):
        """输出转录回调：可用于前端显示/缓存/同步。"""
        try:
            if self.websocket and hasattr(self.websocket, 'client_state') and self.websocket.client_state == self.websocket.client_state.CONNECTED:
                text = self.emotion_pattern.sub('', text)
                message = {
                    "type": "gemini_response",
                    "text": text,
                    "isNewMessage": is_first_chunk
                }
                await self.websocket.send_json(message)
                self.sync_message_queue.put({"type": "json", "data": message})

                if not hasattr(self, 'message_cache_for_new_session'):
                    self.message_cache_for_new_session = []

                if len(self.message_cache_for_new_session) > 0 and self.message_cache_for_new_session[-1]['role'] == self.lanlan_name:
                     self.message_cache_for_new_session[-1]['text'] += text
                else:
                    self.message_cache_for_new_session.append({"role": self.lanlan_name, "text": text})

        except WebSocketDisconnect:
            logger.info("Frontend disconnected.")
        except Exception as e:
            logger.error(f"💥 WS Send Lanlan Response Error: {e}")
        
    async def handle_silence_timeout(self):
        """处理语音输入静默超时"""
        try:
            logger.warning(f"[{self.lanlan_name}] 检测到长时间无语音输入，自动关闭session")
            if self.websocket and hasattr(self.websocket, 'client_state') and self.websocket.client_state == self.websocket.client_state.CONNECTED:
                await self.websocket.send_json({
                    "type": "auto_close_mic",
                    "message": f"{self.lanlan_name}检测到长时间无语音输入，已自动关闭麦克风"
                })
            await self.end_session(by_server=True)
        except Exception as e:
            logger.error(f"处理静默超时时出错: {e}")
    
    async def handle_connection_error(self, message=None):
        if message:
            await self.send_status(message)
        logger.info("💥 Session closed by API Server.")
        await self.disconnected_by_server()

    def _init_renew_status(self):
        self.session_start_time = None  # 记录当前 session 开始时间
        self.pending_session = None  # Managed by connector's __aexit__
        self.is_hot_swap_imminent = False

    async def _flush_tts_pending_chunks(self):
        """将缓存的TTS文本chunk发送到TTS队列"""
        async with self.tts_cache_lock:
            if not self.tts_pending_chunks:
                return
            
            chunk_count = len(self.tts_pending_chunks)
            logger.info(f"TTS就绪，开始处理缓存的 {chunk_count} 个文本chunk...")
            
            if self.tts_process and self.tts_process.is_alive():
                for speech_id, text in self.tts_pending_chunks:
                    try:
                        self.tts_request_queue.put((speech_id, text))
                    except Exception as e:
                        logger.error(f"💥 发送缓存的TTS请求失败: {e}")
                        break
            self.tts_pending_chunks.clear()
    
    async def _flush_pending_input_data(self):
        """将缓存的输入数据发送到session"""
        async with self.input_cache_lock:
            if not self.pending_input_data:
                return
            
            if self.session and self.is_active:
                for message in self.pending_input_data:
                    try:
                        await self._process_stream_data_internal(message)
                    except Exception as e:
                        logger.error(f"💥 发送缓存的输入数据失败: {e}")
                        break
            self.pending_input_data.clear()
    
    def normalize_text(self, text):
        text = text.strip()
        text = text.replace("\n", "")
        if contains_chinese(text):
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "。")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r'[，、]+$', '。', text)
        else:
            text = remove_bracket(text)
            text = spell_out_number(text, self.inflect_parser)
        text = self.emoji_pattern2.sub('', text)
        text = self.emoji_pattern.sub('', text)
        if is_only_punctuation(text) and text not in ['<', '>']:
            return ""
        return text

    async def start_session(self, websocket: WebSocket, new=False, input_mode='audio'):
        # 检查是否正在启动中
        if self.is_starting_session:
            logger.warning(f"⚠️ Session正在启动中，忽略重复请求")
            return
        
        # 标记正在启动
        self.is_starting_session = True
        
        logger.info(f"启动新session: input_mode={input_mode}, new={new}")
        self.websocket = websocket
        self.input_mode = input_mode
        
        # 重新读取核心配置以支持热重载
        core_config = self._config_manager.get_core_config()
        self.vcp_base_url = core_config.get('VCP_BASE_URL', 'http://localhost:6005')
        self.model = core_config['CORE_MODEL']
        self.audio_api_key = core_config['AUDIO_API_KEY']
        
        # 重置TTS缓存状态
        async with self.tts_cache_lock:
            self.tts_ready = False
            self.tts_pending_chunks.clear()
        
        # 重置输入缓存状态
        async with self.input_cache_lock:
            self.session_ready = False
        
        # 文本模式或者自定义语音模式都需要TTS
        # 但VCP原生可能返回Audio，这里我们假设VCP返回文本，使用本地TTS
        self.use_tts = True
        
        async with self.lock:
            if self.is_active:
                logger.warning(f"检测到活跃的旧session，正在清理...")
        
        # 如果检测到旧 session，先清理
        if self.is_active:
            await self.end_session(by_server=True)
            await asyncio.sleep(0.5)
            logger.info("旧session清理完成")
        
        # 定义 TTS 启动协程
        async def start_tts_if_needed():
            if not self.use_tts:
                return True
            
            if self.tts_process is None or not self.tts_process.is_alive():
                has_custom_voice = bool(self.voice_id)
                tts_worker = get_tts_worker(
                    core_api_type=self.core_api_type,
                    has_custom_voice=has_custom_voice
                )
                
                self.tts_request_queue = MPQueue()
                self.tts_response_queue = MPQueue()
                self.tts_process = Process(
                    target=tts_worker,
                    args=(self.tts_request_queue, self.tts_response_queue, self.audio_api_key if has_custom_voice else self.core_api_key, self.voice_id)
                )
                self.tts_process.daemon = True
                self.tts_process.start()
                
                # Wait for TTS ready
                tts_ready = False
                start_time = time.time()
                timeout = 8.0
                
                while time.time() - start_time < timeout:
                    try:
                        if not self.tts_response_queue.empty():
                            msg = self.tts_response_queue.get_nowait()
                            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "__ready__":
                                tts_ready = msg[1]
                                if tts_ready:
                                    logger.info(f"✅ TTS进程已就绪")
                                break
                            else:
                                self.tts_response_queue.put(msg)
                                break
                    except:
                        pass
                    await asyncio.sleep(0.05)
                
                if not tts_ready:
                    logger.warning(f"⚠️ TTS进程就绪信号超时，继续执行...")
            
            if self.tts_handler_task and not self.tts_handler_task.done():
                self.tts_handler_task.cancel()
            
            self.tts_handler_task = asyncio.create_task(self.tts_response_handler())
            
            async with self.tts_cache_lock:
                self.tts_ready = True
            
            await self._flush_tts_pending_chunks()
            return True

        # 定义 LLM Session 启动协程
        async def start_llm_session():
            """异步创建并连接 VCP Client"""
            # 获取初始 prompt (如果需要发送给VCP)
            # VCP可能自己管理System Prompt，这里我们主要连接客户端
            
            logger.info(f"🤖 开始创建 VCP Client Session (input_mode={input_mode})")
            
            self.session = VCPClient(
                base_url=self.vcp_base_url,
                api_key=self.core_api_key, # Assuming same key structure or None
                model=self.model, # Or a specific VCP model name
                on_text_delta=self.handle_text_data,
                on_audio_delta=self.handle_audio_data, # Only if VCP supports returning audio directly
                on_response_done=self.handle_response_complete,
                on_connection_error=self.handle_connection_error
            )

            # VCPClient doesn't have a persistent WebSocket connect method like OmniRealtimeClient
            # It acts more like HTTP client for now. But we mark it ready.
            logger.info(f"✅ VCP Client Ready")
            return True
        
        # 重置状态
        if new:
            self.message_cache_for_new_session = []
            async with self.input_cache_lock:
                self.pending_input_data.clear()

        try:
            logger.info(f"🚀 并行启动 TTS 和 VCP Client...")
            tts_result, llm_result = await asyncio.gather(
                start_tts_if_needed(),
                start_llm_session(),
                return_exceptions=True
            )
            
            if isinstance(tts_result, Exception):
                logger.error(f"TTS 启动失败: {tts_result}")
            if isinstance(llm_result, Exception):
                raise llm_result
            
            if self.session:
                async with self.lock:
                    self.is_active = True
                    
                self.session_start_time = datetime.now()
                self.session_start_failure_count = 0
                
                await self.send_session_started(input_mode)
                
                async with self.input_cache_lock:
                    self.session_ready = True
                
                await self._flush_pending_input_data()
            else:
                raise Exception("Session not initialized")
        
        except Exception as e:
            self.session_start_failure_count += 1
            self.session_start_last_failure_time = datetime.now()
            error_message = f"Error starting session: {e}"
            logger.error(f"💥 {error_message}")
            await self.send_status(f"{error_message}")
            await self.cleanup()
        
        finally:
            self.is_starting_session = False

    async def send_user_activity(self):
        try:
            if self.websocket and hasattr(self.websocket, 'client_state') and self.websocket.client_state == self.websocket.client_state.CONNECTED:
                message = {
                    "type": "user_activity"
                }
                await self.websocket.send_json(message)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"💥 WS Send User Activity Error: {e}")

    # 供主服务调用，更新Agent模式相关开关
    def update_agent_flags(self, flags: dict):
        try:
            for k in ['agent_enabled', 'computer_use_enabled', 'mcp_enabled']:
                if k in flags and isinstance(flags[k], bool):
                    self.agent_flags[k] = flags[k]
        except Exception:
            pass

    async def disconnected_by_server(self):
        await self.send_status(f"{self.lanlan_name}失联了，即将重启！")
        self.sync_message_queue.put({'type': 'system', 'data': 'API server disconnected'})
        await self.cleanup()

    async def stream_data(self, message: dict):  # 向Core API发送数据
        input_type = message.get("input_type")
        
        # 检查session是否就绪
        async with self.input_cache_lock:
            if not self.session_ready:
                self.pending_input_data.append(message)
                return
        
        await self._process_stream_data_internal(message)
    
    async def _process_stream_data_internal(self, message: dict):
        """内部方法：实际处理stream_data的逻辑"""
        data = message.get("data")
        input_type = message.get("input_type")
        
        if self.is_starting_session:
            return
        
        if not self.session or not self.is_active:
             # Auto reconnect logic simplified
            mode = 'text' if input_type == 'text' else 'audio'
            await self.start_session(self.websocket, new=False, input_mode=mode)
            if not self.session or not self.is_active:
                return
        
        try:
            if input_type == 'text':
                if isinstance(data, str):
                    async with self.lock:
                        self.current_speech_id = str(uuid4())
                    await self.send_user_activity()
                    
                    # Prepare messages for VCP
                    history = self.simple_memory.get_recent_history(self.lanlan_name)
                    # Convert langchain messages to dict format expected by VCP/OpenAI
                    messages = []
                    # System Prompt
                    messages.append({"role": "system", "content": self.lanlan_prompt})
                    
                    for msg in history:
                        role = msg.type if hasattr(msg, 'type') else 'user'
                        if role == 'human': role = 'user'
                        if role == 'ai': role = 'assistant'
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        messages.append({"role": role, "content": content})
                        
                    messages.append({"role": "user", "content": data})

                    # Send to VCP
                    asyncio.create_task(self.session.chat_stream(messages))

                    # Update local history with user input immediately
                    await self.simple_memory.update_history([{"role": "user", "content": data}], self.lanlan_name)
                return

            if input_type == 'audio':
                if isinstance(data, list):
                    # Convert audio to text using VCP Speech-to-Text API (via bridge or direct call if supported)
                    # For now, we will rely on a placeholder for future STT integration or VCP's multimodal support
                    # If VCP supports audio input directly in chat/completions (like GPT-4o-audio), we could send it.
                    # Assuming we need text for now, we log a warning.
                    logger.warning("Audio input received. VCP text-only client is active. Please enable STT or use text mode.")
                    # In a real implementation, we would:
                    # 1. Convert PCM `data` to WAV/WebM
                    # 2. Send to VCP's /v1/audio/transcriptions (if compatible) or local Whisper
                    # 3. Get text -> send to chat_stream(text)
                pass

        except Exception as e:
            error_message = f"Stream: Error sending data to session: {e}"
            logger.error(f"💥 {error_message}")
            await self.send_status(error_message)

    async def end_session(self, by_server=False):  # 与Core API断开连接
        self._init_renew_status()

        async with self.lock:
            if not self.is_active:
                return

        logger.info("End Session: Starting cleanup...")
        self.sync_message_queue.put({'type': 'system', 'data': 'session end'})
        async with self.lock:
            self.is_active = False

        if self.session:
            try:
                await self.session.cancel_response()
                self.session = None
            except Exception:
                pass

        if self.tts_handler_task and not self.tts_handler_task.done():
            self.tts_handler_task.cancel()
            self.tts_handler_task = None
            
        if self.tts_process and self.tts_process.is_alive():
            try:
                self.tts_request_queue.put((None, None))
                self.tts_process.terminate()
                self.tts_process.join(timeout=2.0)
                if self.tts_process.is_alive():
                    self.tts_process.kill()
            except Exception as e:
                logger.error(f"💥 关闭TTS进程时出错: {e}")
            finally:
                self.tts_process = None
                
        # Clean queues
        try:
            while not self.tts_request_queue.empty():
                self.tts_request_queue.get_nowait()
        except:
            pass
        
        async with self.tts_cache_lock:
            self.tts_ready = False
            self.tts_pending_chunks.clear()
        
        async with self.input_cache_lock:
            self.session_ready = False
            self.pending_input_data.clear()

        self.last_time = None
        if not by_server:
            await self.send_status(f"{self.lanlan_name}已离开。")
            logger.info("End Session: Resources cleaned up.")

    async def cleanup(self):
        await self.end_session(by_server=True)
        if self.websocket_lock:
            async with self.websocket_lock:
                self.websocket = None
        else:
            self.websocket = None

    async def send_status(self, message: str): # 向前端发送status message
        try:
            if self.websocket and hasattr(self.websocket, 'client_state') and self.websocket.client_state == self.websocket.client_state.CONNECTED:
                data = json.dumps({"type": "status", "message": message})
                await self.websocket.send_text(data)
                self.sync_message_queue.put({'type': 'json', 'data': {"type": "status", "message": message}})
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"💥 WS Send Status Error: {e}")
    
    async def send_session_started(self, input_mode: str): # 通知前端session已启动
        try:
            if self.websocket and hasattr(self.websocket, 'client_state') and self.websocket.client_state == self.websocket.client_state.CONNECTED:
                data = json.dumps({"type": "session_started", "input_mode": input_mode})
                await self.websocket.send_text(data)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"💥 WS Send Session Started Error: {e}")

    async def send_speech(self, tts_audio):
        try:
            if self.websocket and hasattr(self.websocket, 'client_state') and self.websocket.client_state == self.websocket.client_state.CONNECTED:
                await self.websocket.send_bytes(tts_audio)
                self.sync_message_queue.put({"type": "binary", "data": tts_audio})
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"💥 WS Send Response Error: {e}")

    async def tts_response_handler(self):
        while True:
            while not self.tts_response_queue.empty():
                data = self.tts_response_queue.get_nowait()
                if isinstance(data, tuple) and len(data) == 2 and data[0] == "__ready__":
                    continue
                await self.send_speech(data)
            await asyncio.sleep(0.01)
