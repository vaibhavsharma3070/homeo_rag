#!/usr/bin/env python3
"""
Script to create a user in the database.
Usage: python create_user.py <username> <email> <password>
"""
import sys
import hashlib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.config import settings
from app.vector_store import PGVectorStore

def create_user(username: str, email: str, password: str):
    """Create a user in the database."""
    try:
        # Initialize vector store (which will set up the database)
        vector_store = PGVectorStore()
        
        # Create user with admin role
        user = vector_store.create_user(username, password, email=email, role='admin')
        
        if user:
            print(f"✅ User '{username}' created successfully!")
            print(f"   User ID: {user['id']}")
            print(f"   Email: {user.get('email', 'N/A')}")
            print(f"   Role: {user.get('role', 'N/A')}")
            return True
        else:
            print(f"❌ Failed to create user. User '{username}' or email '{email}' may already exist.")
            return False
    except Exception as e:
        print(f"❌ Error creating user: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python create_user.py <username> <email> <password>")
        sys.exit(1)
    
    username = sys.argv[1]
    email = sys.argv[2]
    password = sys.argv[3]
    
    if not username or not email or not password:
        print("❌ Username, email, and password cannot be empty")
        sys.exit(1)
    
    create_user(username, email, password)

