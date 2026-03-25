import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

db_url = os.getenv("DATABASE_URL", "postgresql://analyst:localdev@localhost:5432/urban_demand")

url = db_url.replace("postgresql://", "")
user_pass, rest = url.split("@")
user, password = user_pass.split(":")
host_port, dbname = rest.split("/")
host, port = host_port.split(":")

print(f"Connecting to {host}:{port}/{dbname} ...")

conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)
conn.autocommit = True
cur = conn.cursor()

print("Refreshing ml_features materialized view (may take 1-2 min on 19M rows)...")
cur.execute("REFRESH MATERIALIZED VIEW ml_features")

cur.execute("SELECT COUNT(*) FROM ml_features")
row_count = cur.fetchone()[0]
print(f"Done. ml_features row count: {row_count:,}")

cur.close()
conn.close()