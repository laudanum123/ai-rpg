"""
AI Service module that handles interactions with OpenAI's API.
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from flask import current_app

from app.services.game_state_service import GameStateService

logger = logging.getLogger(__name__)

class AIService:
    """
    Service that handles all interactions with the OpenAI API, including
    preparing prompts, sending requests, and processing responses.
    """

    def __init__(self, openai_client, system_prompt, game_master_debug_logs=None, debug_enabled=False):
        """
        Initialize the AI service.
        
        Args:
            openai_client: The OpenAI client instance
            system_prompt: The system prompt to use for AI interactions
            game_master_debug_logs: Reference to GameMaster's debug logs collection
            debug_enabled: Whether debug mode is enabled, controlled by GameMaster
        """
        self.openai_client = openai_client
        self.system_prompt = system_prompt
        self.api_debug_logs = game_master_debug_logs  # Store reference to GameMaster's logs
        self.debug_enabled = debug_enabled  # Initialize with value from GameMaster
        
        print(f"AIService initialized with debug_enabled={self.debug_enabled}")

    def get_ai_response(
        self,
        messages: Union[List[Dict], str],
        session_id: str = None,
        recent_history_turns: int = 5,
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 2000,
        memory_graph = None,
        game_state_service = None
    ) -> str:
        """Get a response from the AI model with contextual memory.

        Args:
            messages: List of conversation messages
            session_id: Current game session ID for memory retrieval
            recent_history_turns: Number of recent conversation turns to include (default: 5)
            model_name: The OpenAI model to use
            max_tokens: Maximum tokens for the response
            memory_graph: Optional memory graph to use (if not provided, will be fetched from session)
            game_state_service: Optional game state service (if not provided, a new one will be created)

        Returns:
            AI model's response text
        """  # noqa: E501
        # Log request debug info if needed
        if self.debug_enabled:
            self._log_request_debug_info(messages, session_id, model_name)

        # Convert string message to proper format if needed
        processed_messages = self._normalize_messages(messages)

        # Prepare all messages with context and history
        final_messages = self._prepare_messages_with_context(
            processed_messages, session_id, recent_history_turns, memory_graph, game_state_service
        )

        # Initialize debug entry if debug is enabled
        debug_entry = self._create_debug_entry(messages, model_name, max_tokens) if self.debug_enabled else None

        try:
            # Call the OpenAI API and process the response
            response_content = self._call_openai_api(final_messages, model_name, max_tokens)

            # Store response data if debug is enabled
            if self.debug_enabled and debug_entry is not None:
                # Format response to match the structure expected by the debug.html template
                debug_entry["response"] = {
                    "choices": [
                        {
                            "message": {
                                "content": response_content,
                                "role": "assistant"
                            }
                        }
                    ]
                }
                debug_entry["session_id"] = session_id
                debug_entry["request"]["messages"] = final_messages

                # Add to debug logs
                if self.api_debug_logs is not None:
                    print("Adding successful API call to debug logs")
                    self.api_debug_logs.append(debug_entry)

            return response_content

        except Exception as e:
            return self._handle_ai_response_error(e, messages, debug_entry)

    def _log_request_debug_info(self, messages, session_id, model_name):
        """Log debug information about the request."""
        print("DEBUG - get_ai_response called with:")
        print(f"  - debug_enabled: {self.debug_enabled}")
        print(f"  - api_debug_logs count: {len(self.api_debug_logs)}")
        print(f"  - messages type: {type(messages)}")
        if isinstance(messages, list) and len(messages) > 0:
            print(f"  - first message type: {type(messages[0])}")
            print(f"  - first message keys: {messages[0].keys() if isinstance(messages[0], dict) else 'N/A'}")
        print(f"  - session_id: {session_id}")
        print(f"  - model_name: {model_name}")

    def _normalize_messages(self, messages):
        """Convert string messages to the proper format."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return messages

    def _prepare_messages_with_context(self, messages, session_id, recent_history_turns, memory_graph=None, game_state_service=None):
        """Prepare messages with memory context and conversation history."""
        # Extract the current situation from the latest user message
        current_situation = self._extract_current_situation(messages)

        # Get memory context if we have a session ID
        memory_context = self._get_memory_context(session_id, current_situation, memory_graph)

        # Create the system message with memory context
        final_messages = [
            {
                "role": "system",
                "content": self.system_prompt + (f"\n\nRelevant game history:\n{memory_context}" if memory_context else ""),
            }
        ]

        # Get the user's current query
        current_user_query = self._get_current_user_query(messages)

        # Add conversation history if available
        if session_id and recent_history_turns > 0 and current_user_query:
            if game_state_service is None:
                game_state_service = GameStateService()

            final_messages.extend(
                self._get_conversation_history(session_id, recent_history_turns, game_state_service)
            )

        # Add the current query as the final user message
        if current_user_query and (not final_messages or final_messages[-1]["role"] != "user"):
            final_messages.append(current_user_query)

        return final_messages

    def _extract_current_situation(self, messages):
        """Extract the current situation from the latest user message."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _get_memory_context(self, session_id, current_situation, memory_graph=None):
        """Get memory context for the current situation."""
        if not session_id or not current_situation:
            return ""

        # If memory_graph was provided, use it directly
        if memory_graph is not None:
            context = memory_graph.get_relevant_context(
                current_situation, node_limit=10, max_tokens=10000
            )
            return context

        # Otherwise, we would need to get it from a session, but we don't have access
        # to get_session_memory_graph in this service, so we'll return empty
        logger.warning("No memory_graph provided and no way to get it, returning empty context")
        return ""

    def _get_current_user_query(self, messages):
        """Get the current user query from messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg
        return None

    def _get_conversation_history(self, session_id, recent_history_turns, game_state_service):
        """Get relevant conversation history messages."""
        history = game_state_service.get_session_history(session_id)

        if not history or len(history) == 0:
            return []

        # Filter to just player/GM exchanges
        filtered_history = []
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Only include player statements and GM responses
            if role in ["player", "gm"] and content:
                # Further filter out inventory updates or status messages
                if not (
                    "inventory" in content.lower()
                    or "stats:" in content.lower()
                    or "health:" in content.lower()
                    or "updated" in content.lower()
                ):
                    filtered_history.append(msg)

        # Take only recent turns based on parameter
        recent_turns = filtered_history[-min(recent_history_turns * 2, len(filtered_history)):]

        # Convert to OpenAI message format
        openai_messages = []
        for msg in recent_turns:
            role = msg.get("role", "")
            # Convert internal roles to OpenAI format
            openai_role = (
                "user" if role == "player"
                else "assistant" if role == "gm"
                else "system"
            )
            openai_messages.append({"role": openai_role, "content": msg.get("content", "")})

        return openai_messages

    def _create_debug_entry(self, messages, model_name, max_tokens):
        """Create a debug entry for logging API interactions."""
        # Extract the user's prompt appropriately based on message format
        prompt_content = ""
        
        # Handle string messages
        if isinstance(messages, str):
            prompt_content = messages
            # Create a properly formatted message for the debug UI
            formatted_messages = [{"role": "user", "content": messages}]
        # Handle list of message dictionaries
        elif isinstance(messages, list) and len(messages) > 0:
            # Find the user message(s)
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if user_messages:
                # Use the most recent user message for prompt display
                prompt_content = user_messages[-1].get("content", "")
            else:
                # Fallback if no user message found
                prompt_content = str(messages)
            # Use the original message list for formatted display
            formatted_messages = messages
        else:
            # Fallback for any other case
            prompt_content = str(messages)
            formatted_messages = [{"role": "user", "content": prompt_content}]

        # Create a debug entry that matches the expected format in debug.html template
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "prompt": prompt_content,  # This is for backward compatibility
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "response": None,
            "request": {
                "model": model_name,
                "messages": formatted_messages,  # This ensures proper message display in the debug UI
                "temperature": 0.7,
                "max_tokens": max_tokens
            },
            "error": None,
        }

        print(f"Created debug entry: {entry['timestamp']}")
        return entry

    def _call_openai_api(self, final_messages, model_name, max_tokens):
        """Call the OpenAI API and process the response."""
        print(f"About to call OpenAI API, debug_enabled={self.debug_enabled}")
        print(f"Client object: {self.openai_client}")

        response = self.openai_client.chat.completions.create(
            model=model_name,
            messages=final_messages,
            temperature=0.7,
            max_tokens=max_tokens,
        )

        # Handle the response safely in case it's not the expected object type
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content
            else:
                # Handle the case where message structure is unexpected
                return str(choice)
        else:
            # If response doesn't have expected structure, convert to string
            return str(response)

    def _handle_ai_response_error(self, error, messages, debug_entry):
        """Handle errors from the API call."""

        if self.debug_enabled:
            # Print for debugging
            print(f"Debug enabled: {self.debug_enabled}, Debug entry: {debug_entry}")

            # If debug_entry wasn't initialized, create it now
            if debug_entry is None:
                debug_entry = self._create_debug_entry(messages, "unknown", 0)

            debug_entry["error"] = str(error)

            # Add to debug logs
            if self.api_debug_logs is not None:
                self.api_debug_logs.append(debug_entry)

        # Call original error handler with caller info
        return self._handle_original_ai_response_error(error, caller_info={
            "function": "get_ai_response",
            "args": {
                "messages": messages,
            }
        }, debug_entry=debug_entry)

    def _handle_original_ai_response_error(
        self,
        e: Exception,
        caller_info: Dict = None,
        debug_entry: Optional[Dict] = None
    ) -> str:
        """Handle errors from AI responses."""
        import inspect
        import traceback

        # Get detailed error information
        error_type = type(e).__name__
        error_message = str(e)
        error_traceback = traceback.format_exc()

        # Get the frame where the error occurred
        frames = inspect.trace()
        caller_frame = frames[0] if frames else None
        caller_info_str = ""
        if caller_frame:
            frame_info = caller_frame[0]
            caller_file = frame_info.f_code.co_filename
            caller_line = frame_info.f_lineno
            caller_function = frame_info.f_code.co_name
            caller_locals = {k: repr(v) for k, v in frame_info.f_locals.items()
                         if k not in ['self', 'e'] and not k.startswith('__')}
            caller_info_str = f"Error occurred in {caller_file}:{caller_line}, function {caller_function}\n"
            caller_info_str += f"Local variables: {json.dumps(caller_locals, indent=2)}"

        # Log detailed error information
        print(f"ERROR DETAILS\nType: {error_type}\nMessage: {error_message}")
        print(f"Traceback:\n{error_traceback}")
        print(f"Caller Info:\n{caller_info_str}")

        # Store error info in debug entry if provided
        if debug_entry is not None:
            debug_entry["error"] = {
                "type": error_type,
                "message": error_message,
                "traceback": error_traceback,
                "caller_info": caller_info_str
            }

            # Add to debug logs
            if self.api_debug_logs is not None:
                self.api_debug_logs.append(debug_entry)

        # Return a user-friendly error message
        return f"I apologize, but I encountered an error: {str(e)}"

    def get_structured_ai_response(
        self,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        debug_entry: Optional[Dict] = None,
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Tuple[Dict, str]:
        """Get structured AI response using OpenAI function calling.
        
        Args:
            messages: The messages to send to the OpenAI API
            schema: The JSON schema for the function call
            debug_entry: Optional debug entry for logging
            model_name: The model to use (default: gpt-4o-mini)
            max_tokens: Maximum tokens for the response (default: 1000)
            temperature: Temperature for sampling (default: 0.7)
            
        Returns:
            Tuple of (function_args, text_response)
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                functions=[{"name": "output_result", "parameters": schema}],
                function_call={"name": "output_result"},
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Extract the function call from the response
            response_message = response.choices[0].message
            function_call = response_message.function_call
            function_args = json.loads(function_call.arguments)
            text_response = function_args.get("message", "")

            # If debugging, store the response
            if debug_entry is not None:
                debug_entry["response"] = {
                    "choices": [
                        {
                            "message": {
                                "content": text_response
                            }
                        }
                    ]
                }
                self.api_debug_logs.append(debug_entry)

            return function_args, text_response
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {str(e)}")

            if debug_entry is not None:
                debug_entry["error"] = str(e)
                self.api_debug_logs.append(debug_entry)

            # Return empty function args and an error message instead of raising
            return {}, f"Error: {str(e)}"

    def process_function_call_response(self, response_message):
        """Process a response containing a function call.

        Args:
            response_message: The AI response message to process
        
        Returns:
            Tuple containing the function name and arguments
        """
        try:
            # Debug logging
            print(f"Processing function call response: {response_message}")
            print(f"Response message type: {type(response_message)}")

            if hasattr(response_message, 'function_call'):
                function_call = response_message.function_call
                function_name = function_call.name
                function_args = json.loads(function_call.arguments)

                return function_name, function_args
            elif isinstance(response_message, dict) and 'function_call' in response_message:
                function_call = response_message['function_call']
                function_name = function_call.get('name')
                # Try to parse arguments as JSON if they're in string format
                function_args_str = function_call.get('arguments', '{}')
                try:
                    function_args = json.loads(function_args_str)
                except json.JSONDecodeError:
                    function_args = {"raw_args": function_args_str}

                return function_name, function_args
            else:
                print(f"No function call found in response: {response_message}")
                return None, {}

        except Exception as e:
            # Log the error but don't crash
            print(f"Error processing function call: {e}")
            print(f"Response that caused error: {response_message}")
            return None, {}
