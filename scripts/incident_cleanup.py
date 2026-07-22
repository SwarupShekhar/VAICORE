import sqlite3
import sys
import uuid
from datetime import datetime

def cleanup_label_studio(db_path):
    print(f"Connecting to Label Studio Database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Failed to connect to DB: {e}")
        sys.exit(1)

    # 1. View current users
    cursor.execute("SELECT id, email, is_superuser FROM htx_user")
    users = cursor.fetchall()
    
    print("\n--- Current Users in Database ---")
    admin_id = None
    for u in users:
        print(f"ID: {u[0]} | Email: {u[1]} | Superuser: {u[2]}")
        if u[2] == 1:
            admin_id = u[0]
            
    if admin_id is None:
        print("\nERROR: No superadmin found! Aborting to prevent locking you out.")
        sys.exit(1)
        
    print(f"\nIdentified Superadmin ID: {admin_id}")

    # 2. Delete all non-admin users (Unauthorized accounts)
    print("\n--- Purging unauthorized accounts ---")
    cursor.execute("DELETE FROM htx_user WHERE is_superuser = 0 AND id != ?", (admin_id,))
    deleted_count = cursor.rowcount
    print(f"Deleted {deleted_count} unauthorized user(s).")
    
    # 3. Reset Organization token (Invalidates existing invite links)
    print("\n--- Resetting Organization Invite Token ---")
    new_token = str(uuid.uuid4())
    cursor.execute("UPDATE organization SET token = ?", (new_token,))
    print(f"Token reset successful. (New Token: {new_token})")

    # 4. Commit changes
    conn.commit()
    conn.close()
    
    print("\n✅ Database cleanup complete. The hacker's account has been purged.")
    print("   Please start your containers with 'docker compose up -d' now.")

if __name__ == "__main__":
    # Default path on Vultr server
    target_db = "/opt/vaidikai-data/labelstudio/label_studio.sqlite3"
    
    if len(sys.argv) > 1:
        target_db = sys.argv[1]
        
    cleanup_label_studio(target_db)
