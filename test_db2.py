import sys, os
from engine import get_db_connection
import sqlalchemy

conn = get_db_connection()
try:
    conn.execute(sqlalchemy.text("SELECT * FROM invalid_table"))
    conn.commit()
except Exception as e:
    print("Caught:", type(e))
    conn.rollback()
    
try:
    conn.execute(sqlalchemy.text("SELECT 1"))
    print("Success after rollback!")
except Exception as e:
    print("Exception after rollback:", type(e), e)
