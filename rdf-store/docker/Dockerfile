FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV GRAPH_DB_VERSION=10.6.3
ENV GDB_DIST=/opt/graphdb

WORKDIR /opt

# Install required tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget openjdk-11-jdk unzip curl xz-utils pigz gzip gnupg2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Download and unpack GraphDB
RUN wget -q "https://maven.ontotext.com/repository/owlim-releases/com/ontotext/graphdb/graphdb/${GRAPH_DB_VERSION}/graphdb-${GRAPH_DB_VERSION}-dist.zip" \
    && unzip -q "graphdb-${GRAPH_DB_VERSION}-dist.zip" \
    && mv "graphdb-${GRAPH_DB_VERSION}" "${GDB_DIST}" \
    && rm "graphdb-${GRAPH_DB_VERSION}-dist.zip"

# Copy configs and the entrypoint script
COPY config/bestiary-config.ttl /tmp/bestiary-config.ttl
COPY config/wikidata-config.ttl /tmp/wikidata-config.ttl
COPY config/freebase-config.ttl /tmp/freebase-config.ttl
COPY config/dbpedia201604-config.ttl /tmp/dbpedia201604-config.ttl
COPY config/dbpedia201610-config.ttl /tmp/dbpedia201610-config.ttl

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 7200

ENTRYPOINT ["/entrypoint.sh"]