## Setup

1. Go to `rdf-store/`

2. Run `docker build --no-cache -t pythia-graphdb -f docker/Dockerfile .`
Time on our test server: 1m 36s

3. Run `docker run -d -p 65200:7200 -v pythia-graphdb-data:/opt/graphdb/data --name pythia-graphdb-container pythia-graphdb`
Time on our test server: roughly 41 hours

*Note:* This will take some time. To see progress run `docker logs -f pythia-graphdb-container`.

## Removal

1. `docker rm -f pythia-graphdb-container`

2. `docker rmi pythia-graphdb`

3. `docker volume rm pythia-graphdb-data`

## Test Server

**CPU**: Intel Xeon E5-4603 v2 @ 32x 2,2GHz
**RAM**: 128GB
**DSK**: 1.7TB HDD