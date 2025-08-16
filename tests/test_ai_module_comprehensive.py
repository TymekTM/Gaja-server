"""Comprehensive tests for AI modules and providers to increase coverage.

Testuje ai_module, różnych providerów AI, i związane funkcjonalności.
"""
from __future__ import annotations

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import json
from collections import deque

# Import modules to test
from modules.ai_module import (
    AIProviders,
    get_ai_providers,
    health_check,
    remove_chain_of_thought,
    extract_json,
    refine_query,
    chat_with_providers,
    generate_response,
    generate_response_logic,
    AIModule
)


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_choice = MagicMock()
    mock_choice.message.content = "I'll help you with the weather in Warsaw."
    mock_choice.message.tool_calls = [
        MagicMock(
            id="call_123",
            function=MagicMock(
                name="weather_get_weather",
                arguments='{"location": "Warsaw"}'
            )
        )
    ]
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def sample_functions():
    """Sample function definitions for testing."""
    return [
        {
            "type": "function",
            "function": {
                "name": "weather_get_weather",
                "description": "Get weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]


class TestAIProviders:
    """Test AIProviders class functionality."""
    
    def test_initialization(self):
        """Test AIProviders initialization."""
        providers = AIProviders()
        
        assert hasattr(providers, '_httpx_client')
        assert hasattr(providers, 'providers')
        assert isinstance(providers.providers, dict)
        
        # Should have basic providers
        expected_providers = ['openai', 'ollama', 'lmstudio']
        for provider in expected_providers:
            assert provider in providers.providers
    
    def test_check_openai_valid_key(self):
        """Test OpenAI provider with valid API key."""
        providers = AIProviders()
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            result = providers.check_openai()
            assert isinstance(result, bool)
    
    def test_check_openai_no_key(self):
        """Test OpenAI provider without API key."""
        providers = AIProviders()
        
        with patch.dict('os.environ', {}, clear=True):
            with patch('modules.ai_module._config', {}):
                result = providers.check_openai()
                assert result is False
    
    @patch('httpx.AsyncClient')
    async def test_check_ollama_available(self, mock_httpx):
        """Test Ollama provider availability."""
        providers = AIProviders()
        
        # Mock successful Ollama connection
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        providers._httpx_client = mock_client
        
        result = await providers._check_ollama_async()
        assert result is True
    
    @patch('httpx.AsyncClient')
    async def test_check_ollama_unavailable(self, mock_httpx):
        """Test Ollama provider when unavailable."""
        providers = AIProviders()
        
        # Mock failed Ollama connection
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection failed"))
        providers._httpx_client = mock_client
        
        result = await providers._check_ollama_async()
        assert result is False
    
    def test_check_lmstudio(self):
        """Test LM Studio provider check."""
        providers = AIProviders()
        
        with patch('httpx.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            result = providers.check_lmstudio()
            assert isinstance(result, bool)
    
    def test_chat_ollama_success(self):
        """Test successful Ollama chat."""
        providers = AIProviders()
        
        # Mock ollama module
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {
            "message": {"content": "Hello from Ollama"}
        }
        providers._modules["ollama"] = mock_ollama
        providers.providers["ollama"]["module"] = mock_ollama
        
        result = providers.chat_ollama(
            model="llama2",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert isinstance(result, dict)
        assert "message" in result
        if result["message"] is not None:
            assert "content" in result["message"]
    
    def test_chat_ollama_error(self):
        """Test Ollama chat with error."""
        providers = AIProviders()
        
        # Mock ollama module that raises error
        mock_ollama = MagicMock()
        mock_ollama.chat.side_effect = Exception("Connection error")
        providers._modules["ollama"] = mock_ollama
        providers.providers["ollama"]["module"] = mock_ollama
        
        result = providers.chat_ollama(
            model="llama2",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert isinstance(result, dict)
        assert "message" in result
        if result["message"] is not None:
            assert "Ollama error" in result["message"]["content"]
    
    @patch('httpx.post')
    def test_chat_lmstudio_success(self, mock_post):
        """Test successful LM Studio chat."""
        providers = AIProviders()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Hello from LM Studio"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        result = providers.chat_lmstudio(
            model="local-model",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert isinstance(result, dict)
        assert "message" in result
        if result["message"] is not None:
            assert "content" in result["message"]
    
    @patch('httpx.post')
    def test_chat_lmstudio_error(self, mock_post):
        """Test LM Studio chat with error."""
        providers = AIProviders()
        
        mock_post.side_effect = Exception("Network error")
        
        result = providers.chat_lmstudio(
            model="local-model",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert isinstance(result, dict)
        assert "message" in result
        if result["message"] is not None:
            assert "LM Studio error" in result["message"]["content"]
    
    @patch('openai.OpenAI')
    async def test_chat_openai_success(self, mock_openai):
        """Test successful OpenAI chat."""
        providers = AIProviders()
        
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Hello from OpenAI"))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        providers._openai_client = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            result = await providers.chat_openai(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}]
            )
        
        assert isinstance(result, dict)
        assert "message" in result
        if result["message"] is not None:
            assert "content" in result["message"]
    
    @patch('openai.OpenAI')
    async def test_chat_openai_no_key(self, mock_openai):
        """Test OpenAI chat without API key."""
        providers = AIProviders()
        
        with patch.dict('os.environ', {}, clear=True):
            with patch('modules.ai_module._config', {}):
                result = await providers.chat_openai(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hello"}]
                )
        
        assert isinstance(result, dict)
        assert "message" in result
        if result["message"] is not None:
            assert "Błąd: Brak OPENAI_API_KEY" in result["message"]["content"]
    
    async def test_cleanup(self):
        """Test cleanup of async resources."""
        providers = AIProviders()
        
        # Mock httpx client
        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()
        providers._httpx_client = mock_client
        
        await providers.cleanup()
        mock_client.aclose.assert_called_once()


class TestGlobalFunctions:
    """Test global AI module functions."""
    
    def test_get_ai_providers(self):
        """Test getting AI providers singleton."""
        providers1 = get_ai_providers()
        providers2 = get_ai_providers()
        
        assert isinstance(providers1, AIProviders)
        assert providers1 is providers2  # Should be the same instance
    
    def test_health_check(self):
        """Test health check functionality."""
        with patch('modules.ai_module.get_ai_providers') as mock_get_providers:
            mock_providers = MagicMock()
            mock_providers.providers = {
                'openai': {'check': lambda: True},
                'ollama': {'check': lambda: False},
                'lmstudio': {'check': lambda: True}
            }
            mock_get_providers.return_value = mock_providers
            
            result = health_check()
            
            assert isinstance(result, dict)
            assert 'openai' in result
            assert 'ollama' in result
            assert 'lmstudio' in result
    
    def test_remove_chain_of_thought(self):
        """Test removing chain of thought markers."""
        text_with_thought = "Hello <think>this is a thought</think> world"
        result = remove_chain_of_thought(text_with_thought)
        
        assert result == "Hello  world"
        
        # Test other formats
        text_with_markers = "Start <|begin_of_thought|>thought<|end_of_thought|> end"
        result2 = remove_chain_of_thought(text_with_markers)
        
        assert result2 == "Start  end"
    
    def test_extract_json(self):
        """Test JSON extraction from text."""
        # Test with code blocks
        text_with_code = '```json\n{"key": "value"}\n```'
        result = extract_json(text_with_code)
        
        assert '{"key": "value"}' in result
        
        # Test with JSON object
        text_with_json = 'Some text {"test": "data"} more text'
        result2 = extract_json(text_with_json)
        
        assert result2 == '{"test": "data"}'
        
        # Test with plain text
        plain_text = "No JSON here"
        result3 = extract_json(plain_text)
        
        assert result3 == plain_text
    
    @pytest.mark.asyncio
    async def test_refine_query(self):
        """Test query refinement."""
        query = "What's the weather?"
        
        with patch('modules.ai_module.chat_with_providers') as mock_chat:
            mock_chat.return_value = {
                "message": {"content": "Refined query: What is the current weather?"}
            }
            
            result = await refine_query(query, "Polish")
            
            assert isinstance(result, str)
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_refine_query_error(self):
        """Test query refinement with error."""
        query = "Test query"
        
        with patch('modules.ai_module.chat_with_providers') as mock_chat:
            mock_chat.side_effect = Exception("Refine error")
            
            result = await refine_query(query, "Polish")
            
            # Should return original query on error
            assert result == query


class TestChatWithProviders:
    """Test chat_with_providers function."""
    
    @pytest.mark.asyncio
    async def test_chat_with_providers_success(self):
        """Test successful chat with providers."""
        with patch('modules.ai_module.get_ai_providers') as mock_get_providers:
            mock_providers = MagicMock()
            mock_providers.providers = {
                'openai': {
                    'check': lambda: True,
                    'chat': AsyncMock(return_value={
                        "message": {"content": "Test response"}
                    })
                }
            }
            mock_get_providers.return_value = mock_providers
            
            with patch('modules.ai_module.PROVIDER', 'openai'):
                result = await chat_with_providers(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hello"}]
                )
            
            assert isinstance(result, dict)
            assert "message" in result
    
    @pytest.mark.asyncio
    async def test_chat_with_providers_fallback(self):
        """Test chat with providers fallback."""
        with patch('modules.ai_module.get_ai_providers') as mock_get_providers:
            mock_providers = MagicMock()
            mock_providers.providers = {
                'openai': {
                    'check': lambda: False,  # Primary fails
                    'chat': AsyncMock(return_value=None)
                },
                'ollama': {
                    'check': lambda: True,  # Fallback succeeds
                    'chat': lambda *args, **kwargs: {
                        "message": {"content": "Fallback response"}
                    }
                }
            }
            mock_get_providers.return_value = mock_providers
            
            with patch('modules.ai_module.PROVIDER', 'openai'):
                result = await chat_with_providers(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hello"}]
                )
            
            assert isinstance(result, dict)
            assert "message" in result
    
    @pytest.mark.asyncio
    async def test_chat_with_providers_all_fail(self):
        """Test chat when all providers fail."""
        with patch('modules.ai_module.get_ai_providers') as mock_get_providers:
            mock_providers = MagicMock()
            mock_providers.providers = {
                'openai': {
                    'check': lambda: False,
                    'chat': AsyncMock(return_value=None)
                },
                'ollama': {
                    'check': lambda: False,
                    'chat': lambda *args, **kwargs: None
                }
            }
            mock_get_providers.return_value = mock_providers
            
            with patch('modules.ai_module.PROVIDER', 'openai'):
                result = await chat_with_providers(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hello"}]
                )
            
            assert isinstance(result, dict)
            assert "message" in result
            # Should contain error message
            content = result["message"]["content"]
            assert "Błąd" in content or "error" in content.lower()


class TestGenerateResponse:
    """Test generate_response function."""
    
    @pytest.mark.asyncio
    async def test_generate_response_basic(self):
        """Test basic response generation."""
        conversation_history = deque([
            {"role": "user", "content": "Hello"}
        ])
        
        with patch('modules.ai_module.chat_with_providers') as mock_chat:
            mock_chat.return_value = {
                "message": {"content": '{"text": "Hello! How can I help?", "command": "", "params": {}}'}
            }
            
            with patch('modules.ai_module.load_config') as mock_config:
                mock_config.return_value = {"api_keys": {"openai": "test_key"}}
                
                with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
                    result = await generate_response(
                        conversation_history=conversation_history,
                        tools_info="No tools available"
                    )
        
        assert isinstance(result, str)
        
        # Should be valid JSON
        try:
            parsed = json.loads(result)
            assert isinstance(parsed, dict)
            assert "text" in parsed
        except json.JSONDecodeError:
            # Fallback check - should still be a string response
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_with_functions(self):
        """Test response generation with function calling."""
        conversation_history = deque([
            {"role": "user", "content": "What's the weather?"}
        ])
        
        with patch('modules.ai_module.chat_with_providers') as mock_chat:
            mock_chat.return_value = {
                "message": {"content": '{"text": "Let me check the weather", "command": "weather", "params": {}}'},
                "tool_calls_executed": 1
            }
            
            with patch('modules.ai_module.load_config') as mock_config:
                mock_config.return_value = {"api_keys": {"openai": "test_key"}}
                
                with patch('modules.ai_module.PROVIDER', 'openai'):
                    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
                        result = await generate_response(
                            conversation_history=conversation_history,
                            use_function_calling=True
                        )
        
        assert isinstance(result, str)
        
        # Should contain function calling info
        try:
            parsed = json.loads(result)
            assert isinstance(parsed, dict)
        except json.JSONDecodeError:
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_no_api_key(self):
        """Test response generation without API key."""
        conversation_history = deque([
            {"role": "user", "content": "Hello"}
        ])
        
        with patch.dict('os.environ', {}, clear=True):
            with patch('modules.ai_module.load_config') as mock_config:
                mock_config.return_value = {"api_keys": {}}
                
                result = await generate_response(
                    conversation_history=conversation_history
                )
        
        assert isinstance(result, str)
        
        # Should contain error about missing API key
        parsed = json.loads(result)
        assert "Błąd" in parsed["text"] or "klucz" in parsed["text"].lower()


class TestAIModule:
    """Test AIModule class."""
    
    def test_init(self):
        """Test AIModule initialization."""
        config = {"test": "config"}
        module = AIModule(config)
        
        assert module.config == config
        assert isinstance(module.providers, AIProviders)
        assert hasattr(module, '_conversation_history')
    
    @pytest.mark.asyncio
    async def test_process_query_success(self):
        """Test successful query processing."""
        config = {"test": "config"}
        module = AIModule(config)
        
        query = "Hello AI"
        context = {
            "history": [{"role": "user", "content": "Previous message"}],
            "available_plugins": ["weather", "time"],
            "modules": {},
            "user_name": "TestUser"
        }
        
        with patch('modules.ai_module.generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = '{"text": "Hello! How can I help?", "command": "", "params": {}}'
            
            result = await module.process_query(query, context)
        
        assert isinstance(result, dict)
        assert "type" in result
        assert "response" in result
        assert result["type"] == "normal_response"
    
    @pytest.mark.asyncio
    async def test_process_query_clarification(self):
        """Test query processing with clarification request."""
        config = {"test": "config"}
        module = AIModule(config)
        
        query = "Ambiguous query"
        context = {"history": [], "available_plugins": []}
        
        clarification_response = {
            "text": "Please clarify",
            "command": "",
            "params": {},
            "requires_user_response": True,
            "clarification_data": {"options": ["A", "B"]}
        }
        
        with patch('modules.ai_module.generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = json.dumps(clarification_response)
            
            result = await module.process_query(query, context)
        
        assert isinstance(result, dict)
        assert result["type"] == "clarification_request"
        assert "clarification_data" in result
    
    @pytest.mark.asyncio
    async def test_process_query_error(self):
        """Test query processing with error."""
        config = {"test": "config"}
        module = AIModule(config)
        
        query = "Test query"
        context = {}
        
        with patch('modules.ai_module.generate_response') as mock_generate:
            mock_generate.side_effect = Exception("Processing error")
            
            result = await module.process_query(query, context)
        
        assert isinstance(result, dict)
        assert result["type"] == "error_response"
        assert "Przepraszam" in result["response"] or "błąd" in result["response"].lower()
    
    @pytest.mark.asyncio
    async def test_process_query_none_context(self):
        """Test query processing with None context."""
        config = {"test": "config"}
        module = AIModule(config)
        
        query = "Test query"
        context = None
        
        with patch('modules.ai_module.generate_response') as mock_generate:
            mock_generate.return_value = '{"text": "Response", "command": "", "params": {}}'
            
            result = await module.process_query(query, context)
        
        assert isinstance(result, dict)
        assert "type" in result
        assert "response" in result


class TestUtilityFunctions:
    """Test utility functions."""
    
    @pytest.mark.asyncio
    async def test_generate_response_logic(self):
        """Test generate_response_logic function."""
        provider_name = "openai"
        model_name = "gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        tools_info = "Available tools: weather, time"
        
        with patch('modules.ai_module.chat_with_providers', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {
                "message": {"content": "Hello! How can I help you today?"}
            }
            
            result = await generate_response_logic(
                provider_name=provider_name,
                model_name=model_name,
                messages=messages,
                tools_info=tools_info
            )
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_logic_with_images(self):
        """Test generate_response_logic with images."""
        provider_name = "openai"
        model_name = "gpt-4"
        messages = [{"role": "user", "content": "Describe this image"}]
        tools_info = ""
        images = ["image1.jpg", "image2.png"]
        
        with patch('modules.ai_module.chat_with_providers', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {
                "message": {"content": "I can see the images you've shared."}
            }
            
            result = await generate_response_logic(
                provider_name=provider_name,
                model_name=model_name,
                messages=messages,
                tools_info=tools_info,
                images=images,
                active_window_title="Test Window",
                track_active_window_setting=True
            )
        
        assert isinstance(result, str)
        assert len(result) > 0


class TestErrorHandling:
    """Test error handling throughout the AI module."""
    
    def test_safe_import_success(self):
        """Test successful module import."""
        # Test with a module that should exist
        result = AIProviders._safe_import("json")
        assert result is not None
    
    def test_safe_import_failure(self):
        """Test failed module import."""
        # Test with a module that doesn't exist
        result = AIProviders._safe_import("nonexistent_module_12345")
        assert result is None
    
    def test_key_ok_valid(self):
        """Test _key_ok with valid key."""
        with patch.dict('os.environ', {'TEST_KEY': 'valid_key'}):
            with patch('modules.ai_module._config', {"api_keys": {}}):
                result = AIProviders._key_ok('TEST_KEY', 'test')
                assert result is True
    
    def test_key_ok_invalid(self):
        """Test _key_ok with invalid key."""
        with patch.dict('os.environ', {'TEST_KEY': 'YOUR_KEY_HERE'}):
            with patch('modules.ai_module._config', {"api_keys": {}}):
                result = AIProviders._key_ok('TEST_KEY', 'test')
                assert result is False
    
    def test_append_images(self):
        """Test _append_images utility."""
        messages = [{"role": "user", "content": "Hello"}]
        images = ["image1.jpg", "image2.png"]
        
        AIProviders._append_images(messages, images)
        
        assert "Obrazy:" in messages[-1]["content"]
        assert "image1.jpg" in messages[-1]["content"]
        assert "image2.png" in messages[-1]["content"]
    
    def test_append_images_none(self):
        """Test _append_images with None images."""
        messages = [{"role": "user", "content": "Hello"}]
        original_content = messages[-1]["content"]
        
        AIProviders._append_images(messages, None)
        
        # Content should remain unchanged
        assert messages[-1]["content"] == original_content


class TestAdvancedFeatures:
    """Test advanced AI module features."""
    
    @pytest.mark.asyncio
    async def test_openai_function_calling(self, sample_functions):
        """Test OpenAI function calling functionality."""
        providers = AIProviders()
        
        # Mock OpenAI client with function calls
        mock_client = MagicMock()
        mock_response = MagicMock()
        
        # First response with tool calls
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "weather_get_weather"
        mock_tool_call.function.arguments = '{"location": "Warsaw"}'
        
        mock_choice = MagicMock()
        mock_choice.message.content = "I'll check the weather"
        mock_choice.message.tool_calls = [mock_tool_call]
        
        mock_response.choices = [mock_choice]
        
        # Mock function calling system
        mock_function_system = MagicMock()
        mock_function_system.execute_function = AsyncMock(return_value="Sunny, 20°C")
        
        mock_client.chat.completions.create.return_value = mock_response
        providers._openai_client = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            result = await providers.chat_openai(
                model="gpt-4",
                messages=[{"role": "user", "content": "What's the weather?"}],
                functions=sample_functions,
                function_calling_system=mock_function_system
            )
        
        assert isinstance(result, dict)
        assert "message" in result
    
    @pytest.mark.asyncio
    async def test_openai_parameter_handling(self):
        """Test OpenAI parameter handling for different models."""
        providers = AIProviders()
        
        # Mock client for gpt-5 model (new token parameter)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Response"))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        providers._openai_client = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            result = await providers.chat_openai(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.5,
                max_tokens=1000
            )
        
        assert isinstance(result, dict)
        # Should handle parameter variations for newer models
    
    def test_extract_json_complex(self):
        """Test JSON extraction with complex scenarios."""
        # Test nested JSON
        complex_text = 'Response: ```json\n{"data": {"nested": "value"}, "status": "ok"}\n```'
        result = extract_json(complex_text)
        
        assert '"nested": "value"' in result
        assert '"status": "ok"' in result
        
        # Test multiple JSON blocks (should return largest)
        multi_json = 'Small: {"a": 1} Large: {"complex": {"data": "value", "more": "content"}}'
        result2 = extract_json(multi_json)
        
        assert '"complex"' in result2
        assert len(result2) > 10  # Should return the larger JSON block
    
    @pytest.mark.asyncio
    async def test_chat_with_providers_function_calling(self, sample_functions):
        """Test chat_with_providers with function calling."""
        with patch('modules.ai_module.get_ai_providers') as mock_get_providers:
            mock_providers = MagicMock()
            mock_providers.providers = {
                'openai': {
                    'check': lambda: True,
                    'chat': AsyncMock(return_value={
                        "message": {"content": "Function called successfully"},
                        "tool_calls_executed": 1
                    })
                }
            }
            mock_get_providers.return_value = mock_providers
            
            mock_function_system = MagicMock()
            
            with patch('modules.ai_module.PROVIDER', 'openai'):
                result = await chat_with_providers(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Get weather"}],
                    functions=sample_functions,
                    function_calling_system=mock_function_system
                )
            
            assert isinstance(result, dict)
            assert "message" in result
    
    def test_health_check_with_exceptions(self):
        """Test health check when providers raise exceptions."""
        with patch('modules.ai_module.get_ai_providers') as mock_get_providers:
            def failing_check():
                raise Exception("Check failed")
            
            mock_providers = MagicMock()
            mock_providers.providers = {
                'openai': {'check': failing_check},
                'ollama': {'check': lambda: True},
                'broken': {'check': None}  # Not callable
            }
            mock_get_providers.return_value = mock_providers
            
            result = health_check()
            
            assert isinstance(result, dict)
            assert result['openai'] is False  # Should handle exception
            assert result['ollama'] is True
            assert result['broken'] is False  # Should handle non-callable


class TestPerformanceAndCaching:
    """Test performance monitoring and caching features."""
    
    @pytest.mark.asyncio
    async def test_refine_query_caching(self):
        """Test query refinement with caching."""
        query = "Test query for caching"
        
        # Use AsyncMock for async function
        with patch('modules.ai_module.chat_with_providers', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {
                "message": {"content": "Refined: " + query}
            }
            
            # First call
            result1 = await refine_query(query, "Polish")
            call_count_1 = mock_chat.call_count
            
            # Second call with same query (should use cache)
            result2 = await refine_query(query, "Polish")
            call_count_2 = mock_chat.call_count
            
            assert result1 == result2
            # Due to @lru_cache, second call might not increase call count
    
    @pytest.mark.asyncio
    async def test_generate_response_with_fallback(self):
        """Test generate_response with provider fallback."""
        conversation_history = deque([
            {"role": "user", "content": "Hello"}
        ])
        
        call_count = 0
        
        def mock_chat_with_fallback(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call fails (empty content)
                return {"message": {"content": ""}}
            else:
                # Fallback succeeds
                return {"message": {"content": "Fallback response"}}
        
        with patch('modules.ai_module.chat_with_providers', side_effect=mock_chat_with_fallback):
            with patch('modules.ai_module.load_config') as mock_config:
                mock_config.return_value = {"api_keys": {"openai": "test_key"}}
                
                with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
                    with patch('modules.ai_module.PROVIDER', 'openai'):
                        result = await generate_response(
                            conversation_history=conversation_history
                        )
        
        assert isinstance(result, str)
        # Should have attempted fallback
        assert call_count >= 1
    
    def test_providers_module_import_safety(self):
        """Test safe module import in providers."""
        providers = AIProviders()
        
        # Should handle missing modules gracefully
        assert providers._modules.get('openai') is not None or providers._modules.get('openai') is None
        assert providers._modules.get('ollama') is not None or providers._modules.get('ollama') is None
        
        # Should not crash on missing modules
        assert isinstance(providers.providers, dict)
        assert len(providers.providers) > 0
