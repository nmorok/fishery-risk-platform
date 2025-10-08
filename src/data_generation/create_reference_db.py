# create_reference_db.py
import sqlite3
import os

# Delete old database if it exists
if os.path.exists('fisheries.db'):
    os.remove('fisheries.db')
    print("🗑️  Deleted old database")

print("Reading SQL file...")
# ... rest of your code

print("Reading SQL file...")
try:
    # Try UTF-8 encoding explicitly
    with open('fisheries_database.sql', 'r', encoding='utf-8') as f:
        sql_script = f.read()
    print(f"✅ Read {len(sql_script)} characters")
except Exception as e:
    print(f"❌ Error reading file: {e}")
    exit(1)

print("Creating database...")
try:
    conn = sqlite3.connect('fisheries.db')
    cursor = conn.cursor()
    
    # Execute the script
    cursor.executescript(sql_script)
    conn.commit()
    print("✅ Database created successfully!")
    
except sqlite3.OperationalError as e:
    print(f"❌ SQL Error: {e}")
    print("\nFirst 500 characters of SQL file:")
    print(sql_script[:500])
    print("\n...this might help identify the problem")
    
finally:
    conn.close()