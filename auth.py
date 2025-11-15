"""
User authentication system for the AI Document Assistant.
"""
import hashlib
import sqlite3
import os
from typing import Optional, Dict
from datetime import datetime
import streamlit as st


class AuthSystem:
    """Manages user authentication and sessions."""
    
    def __init__(self, db_path: str = "users.db"):
        """
        Initialize authentication system.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create users table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TEXT NOT NULL,
                last_login TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using SHA-256.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username: str, password: str, email: Optional[str] = None) -> bool:
        """
        Register a new user.
        
        Args:
            username: Username
            password: Plain text password
            email: Optional email address
            
        Returns:
            True if successful, False if user already exists
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            password_hash = self.hash_password(password)
            created_at = datetime.now().isoformat()
            
            cursor.execute("""
                INSERT INTO users (username, password_hash, email, created_at)
                VALUES (?, ?, ?, ?)
            """, (username, password_hash, email, created_at))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """
        Authenticate a user.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            True if authentication successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        
        cursor.execute("""
            SELECT id, username FROM users
            WHERE username = ? AND password_hash = ?
        """, (username, password_hash))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Update last login
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET last_login = ? WHERE username = ?
            """, (datetime.now().isoformat(), username))
            conn.commit()
            conn.close()
            return True
        
        return False
    
    def get_user_info(self, username: str) -> Optional[Dict]:
        """
        Get user information.
        
        Args:
            username: Username
            
        Returns:
            User dictionary or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, username, email, created_at, last_login
            FROM users WHERE username = ?
        """, (username,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "id": result[0],
                "username": result[1],
                "email": result[2],
                "created_at": result[3],
                "last_login": result[4]
            }
        
        return None
    
    def user_exists(self, username: str) -> bool:
        """
        Check if a user exists.
        
        Args:
            username: Username to check
            
        Returns:
            True if user exists, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0


def check_authentication():
    """Check if user is authenticated in Streamlit session state."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    
    return st.session_state.authenticated


def login_page(auth_system: AuthSystem):
    """Display login/register page."""
    st.title("🔐 AI Document Assistant - Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary", use_container_width=True):
            if auth_system.authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    
    with tab2:
        st.subheader("Register")
        new_username = st.text_input("Username", key="register_username")
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        email = st.text_input("Email (optional)", key="register_email")
        
        if st.button("Register", type="primary", use_container_width=True):
            if not new_username or not new_password:
                st.error("❌ Username and password are required")
            elif new_password != confirm_password:
                st.error("❌ Passwords do not match")
            elif auth_system.user_exists(new_username):
                st.error("❌ Username already exists")
            else:
                if auth_system.register_user(new_username, new_password, email):
                    st.success("✅ Registration successful! Please login.")
                else:
                    st.error("❌ Registration failed. Username may already exist.")

