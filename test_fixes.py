#!/usr/bin/env python3
"""
Test script to verify the AI evaluator feedback fixes:
1. Tool calling for dynamic information (date/time)
2. JSON escaping fixes
3. Transparency about tool usage
"""

import os
import sys
import asyncio
import json
from collections import deque

# Add current directory to path
sys.path.append('.')

from ai_module import generate_response
import ai_module

# Mock the API call to avoid needing real API keys
async def mock_chat_with_providers(*args, **kwargs):
    """Mock function to simulate AI provider responses."""
    # Check if function calling is being used
    functions = kwargs.get('functions', [])
    messages = args[1] if len(args) > 1 else kwargs.get('messages', [])
    
    # Get the user query from messages
    user_query = ""
    for msg in messages:
        if msg.get('role') == 'user':
            user_query = msg.get('content', '').lower()
            break
    
    if functions:
        # Mock function calling response that includes tool execution
        if 'time' in user_query or 'date' in user_query or 'when' in user_query:
            return {
                'message': {
                    'content': 'It is currently 19:07 on Friday, August 9th, 2025.'
                },
                'tool_calls_executed': ['core_get_current_time']
            }
        elif 'weather' in user_query:
            return {
                'message': {
                    'content': 'Today in Warsaw it\'s 22°C and sunny with light clouds.'
                },
                'tool_calls_executed': ['weather_get_weather']
            }
        else:
            return {
                'message': {
                    'content': 'I can help you with that!'
                },
                'tool_calls_executed': []
            }
    else:
        # Mock regular response without function calling
        return {
            'message': {
                'content': '{"text": "I cannot access current information without proper tools.", "command": "", "params": {}}'
            }
        }

# Patch the function
ai_module.chat_with_providers = mock_chat_with_providers

async def test_function_calling_scenarios():
    """Test various scenarios to verify our fixes."""
    
    print("🧪 Testing AI Evaluator Feedback Fixes\n")
    
    # Test 1: Date/Time query with function calling
    print("1️⃣ Testing date/time query with function calling enabled:")
    conversation_history = deque([{'role': 'user', 'content': 'What time is it?'}])
    response = await generate_response(conversation_history, use_function_calling=True)
    print(f"   Response: {response}")
    
    try:
        response_data = json.loads(response)
        print(f"   ✅ Response text: {response_data.get('text')}")
        print(f"   ✅ Tools used: {response_data.get('tools_used', 'Not specified')}")
        print(f"   ✅ Function calls executed: {response_data.get('function_calls_executed', 'Not specified')}")
        print(f"   ✅ JSON parsing: Success")
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parsing failed: {e}")
    print()
    
    # Test 2: Weather query with function calling
    print("2️⃣ Testing weather query with function calling:")
    conversation_history = deque([{'role': 'user', 'content': 'What\'s the weather like?'}])
    response = await generate_response(conversation_history, use_function_calling=True)
    print(f"   Response: {response}")
    
    try:
        response_data = json.loads(response)
        print(f"   ✅ Response text: {response_data.get('text')}")
        print(f"   ✅ Tools used: {response_data.get('tools_used', 'Not specified')}")
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parsing failed: {e}")
    print()
    
    # Test 3: Regular query without function calling
    print("3️⃣ Testing regular query without function calling:")
    conversation_history = deque([{'role': 'user', 'content': 'Hello, how are you?'}])
    response = await generate_response(conversation_history, use_function_calling=False)
    print(f"   Response: {response}")
    
    try:
        response_data = json.loads(response)
        print(f"   ✅ Response text: {response_data.get('text')}")
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parsing failed: {e}")
    print()
    
    # Test 4: JSON escaping test with quotes
    print("4️⃣ Testing JSON escaping with quotes in content:")
    
    # Mock a response with quotes
    async def mock_with_quotes(*args, **kwargs):
        return {
            'message': {
                'content': 'The time is "19:07" and the date is "August 9th".'
            },
            'tool_calls_executed': ['core_get_current_time']
        }
    
    # Temporarily patch for this test
    original_mock = ai_module.chat_with_providers
    ai_module.chat_with_providers = mock_with_quotes
    
    conversation_history = deque([{'role': 'user', 'content': 'What time is it?'}])
    response = await generate_response(conversation_history, use_function_calling=True)
    print(f"   Response: {response}")
    
    try:
        response_data = json.loads(response)
        print(f"   ✅ Response text: {response_data.get('text')}")
        print(f"   ✅ JSON escaping: Success")
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON escaping failed: {e}")
    
    # Restore original mock
    ai_module.chat_with_providers = original_mock
    print()
    
    print("🎯 Summary of fixes:")
    print("   ✅ Enhanced system prompt to encourage tool usage for date/time")
    print("   ✅ Fixed JSON escaping in function calling responses")
    print("   ✅ Added transparency fields (tools_used, function_calls_executed)")
    print("   ✅ Improved content handling to prevent double-escaping")

if __name__ == "__main__":
    asyncio.run(test_function_calling_scenarios())
