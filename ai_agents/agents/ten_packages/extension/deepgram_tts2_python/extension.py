#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import traceback
import os
import time

from ten_ai_base.struct import TTSTextInput
from ten_ai_base.message import TTSAudioEndReason
from ten_ai_base.tts2 import AsyncTTS2BaseExtension
from .deepgram_tts import DeepgramTTS, DeepgramTTSConfig
from ten_runtime import (
    AsyncTenEnv,
)


class DeepgramTTSExtension(AsyncTTS2BaseExtension):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.config = None
        self.client = None

        # Flush handling state
        self.flush_requested = False
        
        # Circuit breaker for connection resilience
        self.circuit_breaker = {
            'failure_count': 0,
            'last_failure_time': 0,
            'state': 'CLOSED',  # CLOSED, OPEN, HALF_OPEN
            'failure_threshold': 5,
            'recovery_timeout': 30  # seconds
        }

    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        """Initialize the extension with lazy connection"""
        try:
            await super().on_init(ten_env)
            ten_env.log_debug("DeepgramTTS on_init - INITIALIZING")

            # Use TEN framework method to read config as per PR feedback
            config_json, _ = await ten_env.get_property_to_json("")
            self.config = await DeepgramTTSConfig.create_async(ten_env=ten_env)
            ten_env.log_info(f"DEBUG: Received config - api_key: [{self.config.api_key}], type: {type(self.config.api_key)}")



            if not self.config.api_key:
                # Send fatal error using TTS2 base class method
                await self._send_initialization_error("Deepgram API key is required")
                return

            ten_env.log_info(f"Initializing Deepgram TTS with model: {self.config.model}, voice: {self.config.voice}")
            
            # Create client but don't initialize connection yet (lazy connection)
            self.client = DeepgramTTS(self.config)
            self.client.ten_env = ten_env  # Set ten_env for logging
            
            ten_env.log_info("DeepgramTTS extension initialized successfully (lazy connection)")

        except Exception as e:
            ten_env.log_error(f"Failed to initialize Deepgram TTS: {str(e)}")
            ten_env.log_error(f"Traceback: {traceback.format_exc()}")
            # Send fatal error for any other initialization failures
            await self._send_initialization_error(f"Failed to initialize Deepgram TTS: {str(e)}")

    async def on_start(self, ten_env: AsyncTenEnv) -> None:
        """Start the extension - only call parent on_start"""
        try:
            ten_env.log_info("DeepgramTTS extension on_start called")
            await super().on_start(ten_env)
            ten_env.log_info("DeepgramTTS extension started successfully")

        except Exception as e:
            ten_env.log_error(f"Failed to start Deepgram TTS: {str(e)}")
            ten_env.log_error(f"Traceback: {traceback.format_exc()}")

    async def _send_initialization_error(self, message: str):
        """Send initialization error using TTS2 base class method"""
        try:
            # Fixed ModuleError structure as per PR feedback
            from ten_ai_base.message import ModuleError, ModuleErrorCode, ModuleErrorVendorInfo, ModuleType

            error_info = ModuleErrorVendorInfo(
                vendor="deepgram",
                code="INITIALIZATION_ERROR",
                message=message
            )

            error = ModuleError(
                message=message,
                module_name=ModuleType.TTS,
                code=ModuleErrorCode.FATAL_ERROR,
                vendor_info=error_info
            )

            # Use TTS2 base class error sending method
            await self.send_tts_error(None, error)  # No request_id for initialization errors
            self.ten_env.log_error(f"Sent initialization error: {message}")

        except Exception as e:
            self.ten_env.log_error(f"Failed to send initialization error: {str(e)}")

    async def on_stop(self, ten_env: AsyncTenEnv) -> None:
        await super().on_stop(ten_env)
        ten_env.log_debug("DeepgramTTS on_stop")

        # Cleanup client connection
        if self.client:
            await self.client.cleanup()

    def vendor(self) -> str:
        return "deepgram"

    async def request_tts(self, t: TTSTextInput) -> None:
        """Handle TTS request with proper streaming architecture"""
        try:
            # Get text_input_end flag - use direct access instead of getattr
            text_input_end = hasattr(t, 'text_input_end') and t.text_input_end
            
            self.ten_env.log_info(f"Received TTS request: {t.request_id} - {t.text[:50]}... (text_input_end: {text_input_end})")

            # Check circuit breaker before processing
            if not self._should_allow_request():
                self.ten_env.log_error("Circuit breaker OPEN - rejecting TTS request")
                await self._send_circuit_breaker_error(t.request_id)
                return

            # Handle empty text
            if t.text.strip() == "":
                self.ten_env.log_info("Received empty text for TTS request")
                if text_input_end:
                    # Send empty audio response
                    await self._handle_empty_request(t.request_id)
                return

            # Add text chunk to streaming request
            await self.client.add_text_chunk(t.request_id, t.text, text_input_end)
            
            # If this is the end of the request, start streaming audio
            if text_input_end:
                await self._start_streaming_audio(t.request_id)

        except Exception as e:
            self.ten_env.log_error(f"TTS request processing failed: {str(e)}")
            import traceback
            self.ten_env.log_error(f"Traceback: {traceback.format_exc()}")
            await self._send_processing_error(t.request_id, str(e))

    async def _start_streaming_audio(self, request_id: str) -> None:
        """Start streaming audio for a complete request"""
        try:
            self.ten_env.log_info(f"Starting audio streaming for request: {request_id}")
            
            # Send TTS audio start event (once per request)
            await self.send_tts_audio_start(request_id)
            
            # Get streaming request to track TTFB
            streaming_request = self.client.active_requests.get(request_id)
            
            chunk_count = 0
            total_audio_bytes = 0
            
            # Stream audio chunks
            async for audio_chunk in self.client.get_streaming_audio(request_id):
                if audio_chunk and len(audio_chunk) > 0:
                    chunk_count += 1
                    total_audio_bytes += len(audio_chunk)
                    
                    # Mark audio started and send TTFB metrics on first chunk
                    if streaming_request and not streaming_request.audio_started:
                        streaming_request.mark_audio_started()
                        ttfb_ms = streaming_request.get_ttfb_ms()
                        await self.send_tts_ttfb_metrics(request_id, ttfb_ms, -1)
                        self.ten_env.log_info(f"Sent TTFB metrics: {ttfb_ms}ms for request {request_id}")
                    
                    # Dump audio if enabled
                    if self.config.dump_enabled:
                        await self._dump_audio_if_enabled(audio_chunk, request_id)
                    
                    # Send audio data
                    await self.send_tts_audio_data(audio_chunk)
            
            # Send TTS audio end event (once per request)
            duration_ms = self._calculate_audio_duration_ms(total_audio_bytes)
            processing_time_ms = int((time.time() - streaming_request.start_time) * 1000) if streaming_request else 0
            
            reason = TTSAudioEndReason.INTERRUPTED if self.flush_requested else TTSAudioEndReason.REQUEST_END
            
            await self.send_tts_audio_end(
                request_id,
                processing_time_ms,
                duration_ms,
                -1,
                reason
            )
            
            self.ten_env.log_info(f"Completed audio streaming for request {request_id}: {chunk_count} chunks, {total_audio_bytes} bytes")
            
            # Record successful operation
            self._record_success()
            
        except Exception as e:
            self.ten_env.log_error(f"Error streaming audio for request {request_id}: {str(e)}")
            await self._send_processing_error(request_id, str(e))
            self._record_failure()

    async def _handle_empty_request(self, request_id: str) -> None:
        """Handle empty text request"""
        # Send minimal audio events for empty request
        await self.send_tts_audio_start(request_id)
        await self.send_tts_audio_end(request_id, 0, 0, -1, TTSAudioEndReason.REQUEST_END)

    async def _send_circuit_breaker_error(self, request_id: str) -> None:
        """Send circuit breaker error"""
        from ten_ai_base.message import ModuleError, ModuleErrorCode, ModuleErrorVendorInfo, ModuleType
        
        error_info = ModuleErrorVendorInfo(
            vendor="deepgram",
            code="SERVICE_UNAVAILABLE",
            message="TTS service temporarily unavailable due to circuit breaker"
        )
        
        error = ModuleError(
            message="TTS service temporarily unavailable",
            module_name=ModuleType.TTS,
            code=ModuleErrorCode.NON_FATAL_ERROR,
            vendor_info=error_info
        )
        await self.send_tts_error(request_id, error)

    async def _send_processing_error(self, request_id: str, error_message: str) -> None:
        """Send processing error"""
        from ten_ai_base.message import ModuleError, ModuleErrorCode, ModuleErrorVendorInfo, ModuleType
        
        error_info = ModuleErrorVendorInfo(
            vendor="deepgram",
            code="PROCESSING_ERROR",
            message=f"Failed to process TTS request: {error_message}"
        )
        
        error = ModuleError(
            message=f"Failed to process TTS request: {error_message}",
            module_name=ModuleType.TTS,
            code=ModuleErrorCode.NON_FATAL_ERROR,
            vendor_info=error_info
        )
        await self.send_tts_error(request_id, error)

    async def _process_tts_request(self, t: TTSTextInput) -> None:
        """Process TTS request using persistent WebSocket connection"""
        self.ten_env.log_info(f"DEBUG: Starting TTS request processing for request_id: {t.request_id}, text: {t.text[:50]}...")
        try:
            if not self.client:
                self.ten_env.log_error("Deepgram client not initialized")
                return

            self.ten_env.log_info(f"Processing TTS request: {t.request_id} - {t.text[:50]}...")

            self.ten_env.log_info(f"KEYPOINT: TTS request processing started - {t.request_id}")

            # Record start time for TTFB metrics
            start_time = time.time()
            first_chunk_received = False

            # Send TTS audio start event
            await self.send_tts_audio_start(t.request_id)

            # Send TTS text result
            from ten_ai_base.struct import TTSTextResult
            text_result = TTSTextResult(
                request_id=t.request_id,
                text=t.text,
                start_ms=0,
                duration_ms=0,  # Will be updated when we know the total duration
                words=None,  # Deepgram doesn't provide word-level timing in this mode
                text_result_end=True,
                metadata={}
            )
            await self.send_tts_text_result(text_result)

            # Stream audio data from Deepgram using persistent connection
            total_audio_duration_ms = 0
            chunk_count = 0

            async for audio_chunk in self.client.get(self.ten_env, t.text):
                chunk_count += 1

                # Send TTFB metrics for first chunk
                if not first_chunk_received:
                    ttfb_ms = int((time.time() - start_time) * 1000)
                    await self.send_tts_ttfb_metrics(t.request_id, ttfb_ms)
                    first_chunk_received = True

                    # KEYPOINT logging as per TTS standard
                    # Format: TTS [ttfb:100ms] [text:你好你好] [audio_chunk_bytes:1024] [audio_chunk_duration:200ms] [voice_type:xxx]
                    chunk_duration_ms = len(audio_chunk) / (self.config.sample_rate * 2) * 1000
                    self.ten_env.log_info(f"KEYPOINT: TTS [ttfb:{ttfb_ms}ms] [text:{t.text[:20]}...] [audio_chunk_bytes:{len(audio_chunk)}] [audio_chunk_duration:{chunk_duration_ms:.0f}ms] [voice_type:{self.config.voice}]")

                # Send audio data using TTS2 interface
                await self.send_tts_audio_data(audio_chunk)

                # Dump audio data if enabled
                await self._dump_audio_if_enabled(audio_chunk, t.request_id)

                # Estimate audio duration (rough calculation)
                # For 24kHz, 16-bit, mono: bytes / (24000 * 2) * 1000 = ms
                chunk_duration_ms = len(audio_chunk) / (self.config.sample_rate * 2) * 1000
                total_audio_duration_ms += chunk_duration_ms

            # Send TTS audio end event with correct reason
            request_duration_ms = int((time.time() - start_time) * 1000)

            print(f"DEBUG: flush_requested = {self.flush_requested}")
            # Fixed: moved debug after reason assignment
            # Reason 2 = flush requested, Reason 1 = normal completion
            reason = TTSAudioEndReason.INTERRUPTED if self.flush_requested else TTSAudioEndReason.REQUEST_END
            print(f"DEBUG: reason enum = {reason}, reason.value = {reason.value}")

            print(f"DEBUG: About to call send_tts_audio_end with reason={reason}, reason.value={reason.value}")
            await self.send_tts_audio_end(
                t.request_id,
                request_duration_ms,
                int(total_audio_duration_ms),
                -1,
                reason
            )
            
            # Mark request as completed
            self.completed_request_ids.add(t.request_id)
            self.ten_env.log_info(f"Request {t.request_id} added to completed set")

            # If flush was requested, now call base class to send tts_flush_end
            if self.flush_requested and self.pending_flush_data:
                self.ten_env.log_info("KEYPOINT: Sending tts_flush_end after tts_audio_end")
                await super().on_data(self.pending_flush_ten_env, self.pending_flush_data)
                self.pending_flush_data = None
                self.pending_flush_ten_env = None

            # Note: Do NOT send drain command here - it should only be sent in response to flush command
            # The TTS protocol expects tts_audio_end first, then tts_flush_end only after receiving flush command

            # Record successful operation for circuit breaker
            self._record_success()

            self.ten_env.log_info(f"TTS request completed: {t.request_id} - {chunk_count} chunks, {total_audio_duration_ms:.0f}ms audio")

        except Exception as e:
            self.ten_env.log_error(f"TTS request processing failed: {str(e)}")
            self.ten_env.log_error(f"Traceback: {traceback.format_exc()}")

            # Record failure for circuit breaker
            self._record_failure()

            # Send error notification with fixed ModuleError structure
            from ten_ai_base.message import ModuleError, ModuleErrorCode, ModuleErrorVendorInfo, ModuleType

            error_info = ModuleErrorVendorInfo(
                vendor="deepgram",
                code="PROCESSING_ERROR",
                message=f"Failed to process TTS request: {str(e)}"
            )

            error = ModuleError(
                message=f"Failed to process TTS request: {str(e)}",
                module_name=ModuleType.TTS,
                code=ModuleErrorCode.NON_FATAL_ERROR,
                vendor_info=error_info
            )
            await self.send_tts_error(t.request_id, error)

    async def _dump_audio_if_enabled(self, audio_data: bytes, request_id: str = "default"):
        """Dump audio data if dump is enabled"""
        try:
            # Check if dump is enabled in config
            self.ten_env.log_info(f"DEBUG: Checking dump config - hasattr(dump_enabled): {hasattr(self.config, 'dump_enabled')}, dump_enabled: {getattr(self.config, 'dump_enabled', 'NOT_SET')}")
            if hasattr(self.config, 'dump_enabled') and self.config.dump_enabled:
                # Fixed dump path default as per PR feedback
                dump_path = getattr(self.config, 'dump_path', '') + f"/deepgram_tts2_python_out_{request_id}.pcm"

                # Ensure directory exists
                import os
                os.makedirs(os.path.dirname(dump_path), exist_ok=True)

                # Append audio data to dump file
                with open(dump_path, 'ab') as f:
                    f.write(audio_data)

                self.ten_env.log_debug(f"Dumped {len(audio_data)} bytes to {dump_path}")
        except Exception as e:
            self.ten_env.log_error(f"Failed to dump audio data: {str(e)}")

    def _should_allow_request(self) -> bool:
        """
        Circuit breaker pattern - prevents requests during system failures.

        This implements a basic circuit breaker to protect against cascading failures
        when the Deepgram API is experiencing issues. The circuit breaker has three states:
        - CLOSED: Normal operation, requests are allowed
        - OPEN: Too many failures detected, requests are rejected
        - HALF_OPEN: Testing if service has recovered

        The circuit breaker tracks failure counts and automatically transitions between
        states based on failure thresholds and recovery timeouts.

        Returns:
            bool: True if request should be allowed, False if circuit breaker is open
        """
        import time

        current_time = time.time()

        if self.circuit_breaker['state'] == 'CLOSED':
            return True
        elif self.circuit_breaker['state'] == 'OPEN':
            # Check if recovery timeout has passed
            if current_time - self.circuit_breaker['last_failure_time'] > self.circuit_breaker['recovery_timeout']:
                self.circuit_breaker['state'] = 'HALF_OPEN'
                self.ten_env.log_info("KEYPOINT: Circuit breaker transitioning to HALF_OPEN state")
                return True
            return False
        elif self.circuit_breaker['state'] == 'HALF_OPEN':
            return True

        return False

    def _record_success(self):
        """Record successful operation"""
        if self.circuit_breaker['state'] == 'HALF_OPEN':
            self.circuit_breaker['state'] = 'CLOSED'
            self.circuit_breaker['failure_count'] = 0
            self.ten_env.log_info("KEYPOINT: Circuit breaker reset to CLOSED state")

    def _record_failure(self):
        """Record failed operation"""
        import time

        self.circuit_breaker['failure_count'] += 1
        self.circuit_breaker['last_failure_time'] = time.time()

        if self.circuit_breaker['failure_count'] >= self.circuit_breaker['failure_threshold']:
            self.circuit_breaker['state'] = 'OPEN'
            self.ten_env.log_error(f"KEYPOINT: Circuit breaker OPEN - too many failures ({self.circuit_breaker['failure_count']})")

    def synthesize_audio_sample_rate(self) -> int:
        """Return the audio sample rate"""
        return self.config.sample_rate if self.config else 24000

    def synthesize_audio_channels(self) -> int:
        """Return the number of audio channels (mono)"""
        return 1

    def synthesize_audio_sample_width(self) -> int:
        """Return the sample width in bytes (16-bit PCM)"""
        return 2

    async def on_data(self, ten_env: AsyncTenEnv, data):
        """Override on_data to handle flush properly"""
        try:
            self.ten_env.log_info(f"KEYPOINT: on_data called with data_name: {data.get_name()}")
            data_name = data.get_name()

            if data_name == "tts_flush":
                # Set flush flag and store flush data for later processing
                self.flush_requested = True
                self.pending_flush_data = data
                self.pending_flush_ten_env = ten_env
                self.ten_env.log_info("KEYPOINT: Flush requested - will send tts_flush_end after tts_audio_end")
                # Do NOT call base class yet - we will call it after tts_audio_end
                return
            else:
                # Let parent class handle other data
                await super().on_data(ten_env, data)

        except Exception as e:
            self.ten_env.log_error(f"Error handling data: {str(e)}")

    def _calculate_audio_duration_ms(self, total_audio_bytes: int) -> int:
        """Calculate audio duration in milliseconds based on bytes"""
        if total_audio_bytes == 0:
            return 0
        
        # For linear16 encoding: 2 bytes per sample
        bytes_per_sample = 2
        samples = total_audio_bytes // bytes_per_sample
        duration_ms = int((samples / self.config.sample_rate) * 1000)
        
        return max(duration_ms, 100)  # Minimum 100ms


