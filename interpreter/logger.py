import sqlite3
from sqlite3 import Error
import datetime


def log(type, message):
    try:
        conn = sqlite3.connect("logs.db")
        c = conn.cursor()
        timestamp = datetime.datetime.now()
        c.execute("INSERT INTO logs VALUES (?, ?, ?)", (timestamp, type, message))
        conn.commit()
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()
