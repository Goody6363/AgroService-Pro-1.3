import sqlite3

DB_NAME = "ai_memory.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        gear TEXT,
        damage TEXT,
        answer TEXT
    )
    """)

    conn.commit()
    conn.close()

def save_record(user_id, gear, damage, answer):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    INSERT INTO history (user_id, gear, damage, answer)
    VALUES (?, ?, ?, ?)
    """, (user_id, gear, damage, answer))

    conn.commit()
    conn.close()

def get_history(user_id, limit=10):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    SELECT gear, damage, answer
    FROM history
    WHERE user_id=?
    ORDER BY id DESC
    LIMIT ?
    """, (user_id, limit))

    rows = c.fetchall()
    conn.close()

    return list(reversed(rows))