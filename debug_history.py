#!/usr/bin/env python3

import sqlite3
import json

def check_messages_table():
    db_path = 'server_data.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check if messages table exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
    table_exists = cursor.fetchone()
    print(f'Messages table exists: {table_exists is not None}')

    if table_exists:
        # Check table structure
        cursor = conn.execute('PRAGMA table_info(messages)')
        columns = cursor.fetchall()
        print('\nTable structure:')
        for col in columns:
            print(f'  {col["name"]}: {col["type"]}')
        
        # Check if there are any messages
        cursor = conn.execute('SELECT COUNT(*) as count FROM messages')
        count = cursor.fetchone()['count']
        print(f'\nTotal messages in DB: {count}')
        
        # Show recent messages
        cursor = conn.execute('SELECT user_id, role, content, created_at FROM messages ORDER BY created_at DESC LIMIT 10')
        messages = cursor.fetchall()
        print('\nRecent messages:')
        for msg in messages:
            print(f'  User {msg["user_id"]}: [{msg["role"]}] {msg["content"][:50]}... at {msg["created_at"]}')
    else:
        print("Messages table does not exist!")

    conn.close()

if __name__ == "__main__":
    check_messages_table()