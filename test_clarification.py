#!/usr/bin/env python3
"""Test clarification request functionality."""

import asyncio
import json
import websockets


async def test_clarification():
    """Test weather query without location to trigger clarification."""
    uri = "ws://localhost:8001/ws/client1"  # Fixed: added user_id parameter
    
    async with websockets.connect(uri) as websocket:
        # Wait for handshake response first
        handshake_response = await websocket.recv()
        print(f"Handshake: {json.loads(handshake_response)['type']}")
        
        # Send weather query with location
        test_message = {
            "type": "query",  # Fixed: use correct message type
            "data": {
                "query": "sprawdź pogodę",  # Weather request WITHOUT location - should trigger clarification
                "context": {}
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        await websocket.send(json.dumps(test_message))
        print(f"Sent: {test_message}")
        
        # Wait for response
        response = await websocket.recv()
        response_data = json.loads(response)
        
        print(f"Received: {json.dumps(response_data, indent=2)}")
        
        # Check if it's a clarification request
        if response_data.get("type") == "clarification_request":
            print("✅ Clarification request received successfully!")
            print(f"Question: {response_data.get('data', {}).get('question', 'No question found')}")
        else:
            print("❌ Expected clarification request but got different response type")
            print(f"Response type: {response_data.get('type')}")
            if response_data.get("type") == "ai_response":
                print(f"AI Response: {response_data.get('data', {}).get('text', 'No text')}")


if __name__ == "__main__":
    asyncio.run(test_clarification())
