# Generic argument placeholder for various targets
ARGS=

all:
	@echo "This target does nothing, you most likely want to use \
	the pre-built indices to run GRASP; follow the README to do so. \n\
	With this Makefile you can rebuild the benchmarks used with GRASP. \
	They are also available in the data/benchmark directory."

benchmarks: wikidata-benchmarks \
	freebase-benchmarks \
	dbpedia-benchmarks \
	dblp-benchmarks \
	orkg-benchmarks

wikidata-benchmarks:
	@python scripts/prepare_benchmark.py \
	--wikidata-simple-questions \
	--out-dir data/benchmark/wikidata/simplequestions \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--lc-quad2-wikidata \
	--out-dir data/benchmark/wikidata/lcquad2 \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--qald-10 \
	--out-dir data/benchmark/wikidata/qald10 \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--qald-7 data/raw/qald-7 \
	--out-dir data/benchmark/wikidata/qald7 \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--wwq data/raw/wikiwebquestions \
	--out-dir data/benchmark/wikidata/wwq \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--spinach data/raw/spinach \
	--out-dir data/benchmark/wikidata/spinach \
	$(ARGS)

freebase-benchmarks:
	@python scripts/prepare_benchmark.py \
	--wqsp \
	--out-dir data/benchmark/freebase/wqsp \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--cwq \
	--out-dir data/benchmark/freebase/cwq \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--freebase-simple-questions data/raw/simplequestions-freebase \
	--out-dir data/benchmark/freebase/simplequestions \
	$(ARGS)

dbpedia-benchmarks:
	@python scripts/prepare_benchmark.py \
	--lc-quad1-dbpedia \
	--out-dir data/benchmark/dbpedia/lcquad \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--qald-7-dbpedia data/raw/qald7 \
	--out-dir data/benchmark/dbpedia/qald7 \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--qald-9 \
	--out-dir data/benchmark/dbpedia/qald9 \
	$(ARGS)
	@python scripts/prepare_benchmark.py \
	--dbpedia-simple-questions data/raw/simplequestions-dbpedia \
	--out-dir data/benchmark/dbpedia/simplequestions \
	$(ARGS)

dblp-benchmarks:
	@python scripts/prepare_benchmark.py \
	--dblp-quad \
	--out-dir data/benchmark/dblp/dblp-quad \
	$(ARGS)

orkg-benchmarks:
	@python scripts/prepare_benchmark.py \
	--sci-qa \
	--out-dir data/benchmark/orkg-2023/sci-qa \
	$(ARGS)
