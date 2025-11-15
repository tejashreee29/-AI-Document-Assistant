"""
SQLite database management for chat sessions and messages.
"""
import sqlite3
import os
from typing import List, Dict, Optional
from datetime import datetime
from utils import generate_session_id, get_timestamp


class ChatDatabase:
    """Manages SQLite database for chat history."""
    
    def __init__(self, db_path: str = "chat_history.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Create messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id 
            ON messages(session_id)
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self) -> str:
        """
        Create a new chat session.
        
        Returns:
            Session ID
        """
        session_id = generate_session_id()
        timestamp = get_timestamp()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (session_id, created_at, updated_at)
            VALUES (?, ?, ?)
        """, (session_id, timestamp, timestamp))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def save_message(self, session_id: str, role: str, content: str):
        """
        Save a message to the database.
        
        Args:
            session_id: Session ID
            role: 'user' or 'assistant'
            content: Message content
        """
        timestamp = get_timestamp()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update session's updated_at timestamp
        cursor.execute("""
            UPDATE sessions 
            SET updated_at = ? 
            WHERE session_id = ?
        """, (timestamp, session_id))
        
        # Insert message
        cursor.execute("""
            INSERT INTO messages (session_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
        """, (session_id, role, content, timestamp))
        
        conn.commit()
        conn.close()
    
    def get_session_messages(self, session_id: str) -> List[Dict]:
        """
        Get all messages for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of message dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT role, content, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        messages = [
            {
                "role": row[0],
                "content": row[1],
                "timestamp": row[2]
            }
            for row in rows
        ]
        
        return messages
    
    def get_all_sessions(self) -> List[Dict]:
        """
        Get all chat sessions.
        
        Returns:
            List of session dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        sessions = [
            {
                "session_id": row[0],
                "created_at": row[1],
                "updated_at": row[2]
            }
            for row in rows
        ]
        
        return sessions
    
    def delete_session(self, session_id: str):
        """
        Delete a session and all its messages.
        
        Args:
            session_id: Session ID to delete
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete messages first (foreign key constraint)
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        
        # Delete session
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()

