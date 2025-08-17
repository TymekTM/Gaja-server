#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the server directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config_manager import DatabaseManager

async def test_conversation_history():
    """Test saving and retrieving conversation history"""
    db_manager = DatabaseManager("server_data.db")
    
    print("Testing conversation history functionality...")
    
    # Test user
    user_id = "client1"
    
    # Save a test interaction
    test_query = "Jaka jest pogoda?"
    test_response = "Dla jakiego miasta chcesz pogodę?"
    
    print(f"Saving interaction: '{test_query}' -> '{test_response}'")
    await db_manager.save_interaction(user_id, test_query, test_response)
    
    # Retrieve history
    print("Retrieving history...")
    history = await db_manager.get_user_history(user_id, limit=10)
    
    print(f"Retrieved {len(history)} messages:")
    for i, msg in enumerate(history):
        print(f"  {i+1}. [{msg['role']}] {msg['content']}")
    
    # Save another interaction
    test_query2 = "W Warszawie"
    test_response2 = "Aktualna pogoda w Warszawie: słonecznie, 22°C"
    
    print(f"\nSaving second interaction: '{test_query2}' -> '{test_response2}'")
    await db_manager.save_interaction(user_id, test_query2, test_response2)
    
    # Retrieve updated history
    print("Retrieving updated history...")
    history = await db_manager.get_user_history(user_id, limit=10)
    
    print(f"Retrieved {len(history)} messages:")
    for i, msg in enumerate(history):
        print(f"  {i+1}. [{msg['role']}] {msg['content']}")

if __name__ == "__main__":
    asyncio.run(test_conversation_history())