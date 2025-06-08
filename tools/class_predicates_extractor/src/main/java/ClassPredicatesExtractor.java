import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.jena.graph.Triple;
import org.apache.jena.riot.Lang;
import org.apache.jena.riot.RDFParser;
import org.apache.jena.riot.system.StreamRDF;
import org.apache.jena.sparql.core.Quad;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;

public class ClassPredicatesExtractor {
    static Map<String, Class> classes = new HashMap<>();
    static Map<String, Set<Class>> entityToClassMap = new ConcurrentHashMap<>();

    static enum Direction {
        OUTGOING,
        INCOMING
    }

    static enum Target {
        URI,
        LITERAL
    }

    static class Predicate {
        String uri;
        Direction direction;
        LongAdder countToUri;
        LongAdder countToLiteral;

        Predicate(String uri, Direction direction) {
            this.uri = uri;
            this.direction = direction;
            this.countToUri = new LongAdder();
            this.countToLiteral = new LongAdder();
        }

        void increment(Target target) {
            if (target == Target.URI)
                this.countToUri.increment();
            else
                this.countToLiteral.increment();
        }
    }

    // Key object for (predicate URI, direction) pairs
    static final class PredicateKey {
        final String uri;
        final Direction direction;
        PredicateKey(String uri, Direction direction) { this.uri = uri; this.direction = direction; }
        @Override public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof PredicateKey)) return false;
            PredicateKey other = (PredicateKey) o;
            return Objects.equals(uri, other.uri) && direction == other.direction;
        }
        @Override public int hashCode() { return Objects.hash(uri, direction); }
    }

    static class Class {
        String uri;
        // Replaced predicate storage from Set with maps to achieve O(1) updates instead of O(P) scans.
        ConcurrentHashMap<PredicateKey, Predicate> predicates;
        ConcurrentHashMap<String, Predicate> classPredicates;

        Class(String uri) {
            this.uri = uri;
            this.predicates = new ConcurrentHashMap<>();
            this.classPredicates = new ConcurrentHashMap<>();
        }

        void incrementPredicateCount(String predicateUri, Direction direction, Target target) {
            PredicateKey key = new PredicateKey(predicateUri, direction);
            predicates.computeIfAbsent(key, k -> new Predicate(predicateUri, direction))
                      .increment(target);
        }

        void incrementClassPredicateCount(String predicateUri, Target target) {
            classPredicates.computeIfAbsent(predicateUri, k -> new Predicate(predicateUri, Direction.OUTGOING))
                           .increment(target);
        }        
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: ClassPredicatesExtractor -i <inputDir> -o <outputFile> -cf <classFile> [-t numThreads]");
            System.exit(1);
        }

        // Create the command-line parser
        CommandLineParser parser = new DefaultParser();
        Options options = new Options();

        // Define options for the arguments
        options.addRequiredOption("i", "input", true, "Input directory/file");
        options.addRequiredOption("o", "output", true, "Output file");
        options.addRequiredOption("cf", "class-file", true, "Class file");
        options.addOption("t", "threads", true, "Number of threads");

        Path inputFile = null;
        Path outputFile = null;
        Path classFile = null;
        int numThreads = -1;

        try {
            // Parse the command-line arguments
            CommandLine cmd = parser.parse(options, args);

            // Get input and output paths
            inputFile = Paths.get(cmd.getOptionValue("i"));
            outputFile = Paths.get(cmd.getOptionValue("o"));
            classFile = Paths.get(cmd.getOptionValue("cf"));
            numThreads = cmd.hasOption("t") ? Integer.parseInt(cmd.getOptionValue("t")) : Runtime.getRuntime().availableProcessors();

            // Output parsed values
            System.out.println("Input File: " + inputFile);
            System.out.println("Output File: " + outputFile);
            System.out.println("Class File: " + classFile);
            System.out.println("Number of Threads: " + numThreads);
        } catch (Exception e) {
            System.err.println("Error parsing arguments: " + e.getMessage());
            new HelpFormatter().printHelp("EntityExtractor", options);
            System.exit(1);
        }

        // Load classes from class file
        List<String> classLines = Files.readAllLines(classFile);
        for (String line : classLines) {
            String classUri = line.trim();
            if (!classUri.isEmpty()) {
                classes.put(classUri, new Class(classUri));
            }
        }

        // Extract entities to class map (Preprocessing)
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CompletionService<Map<String, Set<Class>>> completionService = 
            new ExecutorCompletionService<>(executor);

        AtomicInteger fileCount = new AtomicInteger(0);
        if (Files.isDirectory(inputFile)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(inputFile)) {
                for (Path file : stream) {
                    if (Files.isRegularFile(file)) {
                        fileCount.incrementAndGet();
                        completionService.submit(() -> getEntitiesToClassMap(file));
                    }
                }
            }
        } else if (Files.isRegularFile(inputFile)) {
            // If input is a file, process it directly
            final var inputFileTemp = inputFile; // just to bypass  the need for final
            fileCount.incrementAndGet();
            completionService.submit(() -> getEntitiesToClassMap(inputFileTemp));
        } else {
            System.err.println("Input must be a directory or a file.");
            System.exit(1);
        }

        for (int i = 0; i < fileCount.get(); i++) {
            try {
                Map<String, Set<Class>> results = completionService.take().get();
                // Merge per-file results into the global map without overwriting existing classes
                for (Map.Entry<String, Set<Class>> entry : results.entrySet()) {
                    entityToClassMap
                        .computeIfAbsent(entry.getKey(), k -> ConcurrentHashMap.newKeySet())
                        .addAll(entry.getValue());
                }
            } catch (ExecutionException e) {
                System.err.println("Error processing file: " + e.getCause().getMessage());
            }
        }
        executor.shutdown();
        executor.awaitTermination(100, TimeUnit.HOURS);

        System.out.println("Found " + classes.size() + " classes");
        System.out.println("Found " + entityToClassMap.size() + " entity-class mappings");

        // Clear output file
        Files.deleteIfExists(outputFile);
        Files.createFile(outputFile);

        // Collect predicates for each class
        executor = Executors.newFixedThreadPool(numThreads);
        CompletionService<Void> predicatesCompletionService = new ExecutorCompletionService<>(executor);

        fileCount = new AtomicInteger(0);
        if (Files.isDirectory(inputFile)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(inputFile)) {
                for (Path file : stream) {
                    if (Files.isRegularFile(file)) {
                        fileCount.incrementAndGet();
                        predicatesCompletionService.submit(() -> { processPredicatesForClasses(file); return null; });
                    }
                }
            }
        } else if (Files.isRegularFile(inputFile)) {
            final var inputFileTemp = inputFile;
            fileCount.incrementAndGet();
            predicatesCompletionService.submit(() -> { processPredicatesForClasses(inputFileTemp); return null; });
        } else {
            System.err.println("Input must be a directory or a file.");
            System.exit(1);
        }

        for (int i = 0; i < fileCount.get(); i++) {
            try {
                predicatesCompletionService.take().get();
            } catch (ExecutionException e) {
                System.err.println("Error processing file for predicates: " + e.getCause().getMessage());
                e.getCause().printStackTrace();
            }
        }
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        // Write popularity counts to output file only, format:
        // <classUri>\t<DIRECT|INSTANCE>\t<OUTGOING|INCOMING>\t<predicateUri>\t<count>
        try (BufferedWriter writer = Files.newBufferedWriter(outputFile)) {
            for (var classObj : classes.values()) {
                class Row { String direction; String predicate; long count; Row(String d, String p, long c){ this.direction=d; this.predicate=p; this.count=c; } }
                List<Row> rows = new ArrayList<>();

                for (Predicate p : classObj.predicates.values()) {
                    long totalCount = p.countToUri.sum() + p.countToLiteral.sum();
                    rows.add(new Row(p.direction == Direction.OUTGOING ? "outgoing" : "incoming", p.uri, totalCount));
                }

                // Sort combined rows by descending count (like ORDER BY DESC(?count))
                rows.sort((a, b) -> Long.compare(b.count, a.count));

                for (Row r : rows) {
                    writer.write(classObj.uri + "\t" + "instance" + "\t" + r.direction + "\t" + r.predicate + "\t" + r.count + "\n");
                }

                // Class predicates
                rows.clear();
                for (Predicate p : classObj.classPredicates.values()) {
                    long totalCount = p.countToUri.sum() + p.countToLiteral.sum();
                    rows.add(new Row("outgoing", p.uri, totalCount));
                }

                // Sort class predicate rows by descending count
                rows.sort((a, b) -> Long.compare(b.count, a.count));

                for (Row r : rows) {
                    writer.write(classObj.uri + "\t" + "direct" + "\t" + r.direction + "\t" + r.predicate + "\t" + r.count + "\n");
                }
            }
        }

        System.out.println("Output written to " + outputFile);

        // Also write filtered popularity counts (outgoing excluding literals), same format
        Path filteredOutput = Paths.get(outputFile.toString() + ".no_literals");
        try (BufferedWriter writer = Files.newBufferedWriter(filteredOutput)) {
            for (var classObj : classes.values()) {
                class Row { String direction; String predicate; long count; Row(String d, String p, long c){ this.direction=d; this.predicate=p; this.count=c; } }
                List<Row> rows = new ArrayList<>();

                for (Predicate p : classObj.predicates.values()) {
                    long totalCount = p.countToUri.sum();
                    rows.add(new Row(p.direction == Direction.OUTGOING ? "outgoing" : "incoming", p.uri, totalCount));
                }

                // Sort combined rows by descending count (like ORDER BY DESC(?count))
                rows.sort((a, b) -> Long.compare(b.count, a.count));

                for (Row r : rows) {
                    writer.write(classObj.uri + "\t" + "instance" + "\t" + r.direction + "\t" + r.predicate + "\t" + r.count + "\n");
                }

                // Class predicates
                rows.clear();
                for (Predicate p : classObj.classPredicates.values()) {
                    long totalCount = p.countToUri.sum();
                    rows.add(new Row("outgoing", p.uri, totalCount));
                }

                // Sort class predicate rows by descending count
                rows.sort((a, b) -> Long.compare(b.count, a.count));

                for (Row r : rows) {
                    writer.write(classObj.uri + "\t" + "direct" + "\t" + r.direction + "\t" + r.predicate + "\t" + r.count + "\n");
                }
            }
        }

        System.out.println("Filtered (no-literals outgoing) output written to " + filteredOutput);

        System.out.println("Finished processing files.");
    }

    private static Map<String, Set<Class>> getEntitiesToClassMap(Path file) throws IOException {        
        Map<String, Set<Class>> entitiesToClassMap = new ConcurrentHashMap<>();

        var time = System.currentTimeMillis();
        System.out.println("Processing file: " + file);

        StreamRDF processor = new StreamRDF() {
            @Override
            public void triple(Triple triple) {
                processTypeTriple(triple, entitiesToClassMap);
            }

            @Override public void start() {}
            @Override public void quad(Quad quad) {
                // Treat quads as triples by ignoring the graph component
                if (quad != null) {
                    processTypeTriple(quad.asTriple(), entitiesToClassMap);
                }
            }
            @Override public void base(String base) {}
            @Override public void prefix(String prefix, String iri) {}
            @Override public void finish() {}
        };

        RDFParser.source(file.toString())
            .lang(Lang.NTRIPLES)
            .parse(processor);

        RDFParser.source(file.toString())
            .lang(Lang.NQUADS)
            .parse(processor);

        System.out.println("Finished processing file: " + file + " in " + (System.currentTimeMillis() - time) + " ms");

        return entitiesToClassMap;
    }

    private static void processTypeTriple(Triple triple, Map<String, Set<Class>> entitiesToClassMap) {
        if (!triple.getSubject().isURI()) 
            return; // Skip blank nodes
        if (!triple.getObject().isURI())
            return; // Skip literals

        String subject = triple.getSubject().getURI();
        String predicate = triple.getPredicate().getURI();
        String object = triple.getObject().getURI();

        if (predicate.contains("http://www.w3.org/1999/02/22-rdf-syntax-ns#type") || predicate.contains("http://www.wikidata.org/prop/direct/P31")) {
            var classObj = classes.get(object);
            if (classObj != null) {
                if (!entitiesToClassMap.containsKey(subject)) {
                    entitiesToClassMap.put(subject, ConcurrentHashMap.newKeySet());
                }
                entitiesToClassMap.get(subject).add(classObj);
            }
        }
    }


    // New: For each triple, count outgoing and incoming predicate popularity per class
    private static void processPredicatesForClasses(Path file) throws IOException {

        var time = System.currentTimeMillis();
        System.out.println("Processing file: " + file);

        StreamRDF processor = new StreamRDF() {
            @Override
            public void triple(Triple triple) {
                if (!triple.getSubject().isURI()) return;
                String subject = triple.getSubject().getURI();
                String predicate = triple.getPredicate().getURI();
                String object = triple.getObject().isURI() ? triple.getObject().getURI() : null;

                // Predicates of classes themselves
                // Outgoing: subject is class, count predicate as outgoing for class
                if (classes.containsKey(subject)) {
                    Class classObj = classes.get(subject);
                    if (triple.getObject().isLiteral()) {
                        classObj.incrementClassPredicateCount(predicate, Target.LITERAL);
                    } else {
                        classObj.incrementClassPredicateCount(predicate, Target.URI);
                    }
                } else if (classes.containsKey(object)) {
                    // Incoming: object is class, count predicate as incoming for class
                    Class classObj = classes.get(object);
                    if (triple.getSubject().isLiteral()) {
                        classObj.incrementClassPredicateCount(predicate, Target.LITERAL);
                    } else {
                        classObj.incrementClassPredicateCount(predicate, Target.URI);
                    }
                } else {
                    // Predicates of entities which are instances of classes
                    // Outgoing: subject has class, count predicate as outgoing for each class
                    if (entityToClassMap.containsKey(subject)) {
                        for (Class classObj : entityToClassMap.get(subject)) {
                            if (triple.getObject().isLiteral()) {
                                classObj.incrementPredicateCount(predicate, Direction.OUTGOING, Target.LITERAL);
                            } else {
                                classObj.incrementPredicateCount(predicate, Direction.OUTGOING, Target.URI);
                            }
                        }
                    }

                    // Incoming: object is URI and has class, count predicate as incoming for each class
                    if (triple.getObject().isURI()) {
                        object = triple.getObject().getURI();
                        if (entityToClassMap.containsKey(object)) {
                            for (Class classObj : entityToClassMap.get(object)) {
                                classObj.incrementPredicateCount(predicate, Direction.INCOMING, Target.URI);
                            }
                        }
                    }
                }
            }
            @Override public void start() {}
            @Override public void quad(Quad quad) {}
            @Override public void base(String base) {}
            @Override public void prefix(String prefix, String iri) {}
            @Override public void finish() {}
        };
        RDFParser.source(file.toString())
            .lang(Lang.NTRIPLES)
            .parse(processor);

        System.out.println("Finished processing file: " + file + " in " + (System.currentTimeMillis() - time) + " ms");
    }

}