import sqlite3
import csv
import os
import argparse


def build_db(input_mapping, db_file, overwrite=False):
    if os.path.exists(db_file):
        os.remove(db_file)

    print(f"Building SQLite index from {input_mapping}...")
    
    # Connect to SQLite with performance-friendly pragmas
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('PRAGMA journal_mode=WAL;')
    c.execute('PRAGMA synchronous=OFF;')
    c.execute('PRAGMA temp_store=MEMORY;')
    c.execute('PRAGMA cache_size=-200000;') # ~200MB cache, adjust if needed
    c.execute('PRAGMA locking_mode=EXCLUSIVE;')

    # Bulk load first, then add indexes (avoids per-row index maintenance)
    conn.execute('BEGIN IMMEDIATE;')
    c.execute('CREATE TABLE entities (id INTEGER NOT NULL, uri TEXT NOT NULL)')

    # Read TSV and Insert
    # Your Java code outputs: ID \t URI
    with open(input_mapping, 'r', encoding='utf-8') as f:
        # csv.reader handles the tab separation automatically
        reader = csv.reader(f, delimiter='\t')
        
        # Skip header if present (Your Java code writes "ID\tURI")
        header = next(reader, None) 
        
        batch = []
        count = 0
        
        for row in reader:
            if len(row) < 2: continue
            
            # row[0] is ID, row[1] is URI
            batch.append((int(row[0]), row[1]))
            count += 1
            
            if len(batch) >= 100_000:
                c.executemany('INSERT INTO entities VALUES (?,?)', batch)
                batch = []
                print(f"Indexed {count:,} entities...", end='\r')
        
        if batch:
            c.executemany('INSERT INTO entities VALUES (?,?)', batch)

    # Add indexes after bulk insert for much faster load
    print("\nCreating indexes...")
    c.execute('CREATE UNIQUE INDEX idx_id ON entities(id)')
    c.execute('CREATE INDEX idx_uri ON entities(uri)')

    print(f"\nCommitting final changes...")
    conn.commit()
    conn.close()
    print("Done! 'mapping.db' is ready.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SQLite index from entity mapping TSV.")
    parser.add_argument('input_mapping', type=str,
                        help="Path to the input entity mapping TSV file.")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing database file if it exists.")
    args = parser.parse_args()
    
    db_file_path = os.path.join(os.path.dirname(args.input_mapping), 'mapping.db')
    
    build_db(args.input_mapping, db_file_path, args.overwrite)