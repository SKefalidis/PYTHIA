#!/usr/bin/env bash
set -e

DATA_DIR="/opt/graphdb/data"
RAW_DIR="$DATA_DIR/tmp_raw"

MARKER_DIR="$DATA_DIR/.preload_markers"

# A repository counts as loaded only once its marker exists, so an interrupted
# preload (which may leave a partial repository directory) is retried.
repo_loaded() {
    local repo_id="$1"

    [ -f "$MARKER_DIR/$repo_id" ]
}

# Function to process a repository
preload_repo() {
    local repo_id="$1"
    local config_file="$2"
    local data_url="$3"
    local heap_size="$4"
    
    echo "==> Setting up repository $repo_id using $config_file..."

    # Ensure that no previous data is left over from a failed preload
    rm -rf "$RAW_DIR"
    
    mkdir -p "$RAW_DIR"

    echo "==> Downloading raw data from $data_url..."
    wget -q -P "$RAW_DIR" "$data_url"
    
    echo "==> Converting .xz to .gz..."
    # pigz processes this much faster using multiple cores
    xz -T0 -dc "$RAW_DIR"/*.nt.xz | pigz -c > "$RAW_DIR"/data.nt.gz
    rm -f "$RAW_DIR"/*.nt.xz
    
    echo "==> Preloading into GraphDB..."
    # Notice we point importrdf to the DATA_DIR volume
    if [ -n "$heap_size" ]; then
        GDB_HEAP_SIZE="$heap_size" /opt/graphdb/bin/importrdf preload -f \
            -c "$config_file" "$RAW_DIR"/data.nt.gz \
            -Dgraphdb.home="$DATA_DIR" \
            -Dgraphdb.health.minimal.free.storage.warn=10
    else
        /opt/graphdb/bin/importrdf preload -f \
            -c "$config_file" "$RAW_DIR"/data.nt.gz \
            -Dgraphdb.home="$DATA_DIR" \
            -Dgraphdb.health.minimal.free.storage.warn=10
    fi
        
    echo "==> Cleaning up raw data..."
    rm -rf "$RAW_DIR"

    mkdir -p "$MARKER_DIR"
    touch "$MARKER_DIR/$repo_id"
}

# Check each repository independently so a partial volume can be completed.
if ! repo_loaded "beastiary"; then
    preload_repo "beastiary" "/tmp/bestiary-config.ttl" "https://ai.di.uoa.gr/file-server/knowledge-graphs/bestiary.nt.xz"
fi

if ! repo_loaded "wikidata-qald-10"; then
    preload_repo "wikidata-qald-10" "/tmp/wikidata-config.ttl" "https://ai.di.uoa.gr/file-server/knowledge-graphs/wikidata-qald-10.nt.xz" "60g"
fi

if ! repo_loaded "freebase"; then
    preload_repo "freebase" "/tmp/freebase-config.ttl" "https://ai.di.uoa.gr/file-server/knowledge-graphs/freebase.nt.xz"
fi

if ! repo_loaded "dbpedia2016"; then
    preload_repo "dbpedia2016" "/tmp/dbpedia201604-config.ttl" "https://ai.di.uoa.gr/file-server/knowledge-graphs/dbpedia201604.nt.xz"
fi

if ! repo_loaded "dbpedia10"; then
    preload_repo "dbpedia10" "/tmp/dbpedia201610-config.ttl" "https://ai.di.uoa.gr/file-server/knowledge-graphs/dbpedia201610.nt.xz"
fi

echo "Database build checks complete!"

echo "==> Starting GraphDB Web Engine..."
exec /opt/graphdb/bin/graphdb \
    -Dgraphdb.engine.onheap.allocation=false \
    -Dgraphdb.home="$DATA_DIR" \
    -Dgraphdb.logger.root.level=INFO