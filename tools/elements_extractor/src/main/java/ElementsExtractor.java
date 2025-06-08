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
import java.io.UncheckedIOException;
import java.nio.file.*;
import java.sql.*;
import java.util.*;
import java.util.function.Consumer;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;


// FIXME: Misses some predicates. Check with beastiary, it is apparent when you go through the file.
// FIXME: If there are no labels, maybe we should create a label from the URI? Split the last part of the URI with spaces? But this could mess up IDs.
public class ElementsExtractor {
    private static final Pattern WD_ENTITY_PROP = Pattern.compile("^https?://www\\.wikidata\\.org/entity/(P\\d+)$");
    // Hot-path membership structures as Sets for O(1) contains
    static Set<String> label_predicates = new HashSet<>();
    static Set<String> classes_predicates = new HashSet<>();
    static Set<String> description_predicates = new HashSet<>();
    // Prefix filters remain lists because we check substring contains; use a tight loop helper
    static ArrayList<String> entity_prefixes = new ArrayList<>();
    static ArrayList<String> class_prefixes = new ArrayList<>();
    static boolean filter_entities = false;
    static boolean filter_classes = false;
    private static final String RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    private static final String WD_P31 = "http://www.wikidata.org/prop/direct/P31";

    static boolean no_entity_labels = false;
    static boolean no_predicate_labels = false;
    static boolean no_class_labels = false;

    static boolean memoryOptimized = false;
    static ArrayList<String> commonPrefixes = new ArrayList<>();
    static Path sqliteDb = null;

    private static ArrayList<String> parseList(String listStr) {
        var list = new ArrayList<String>();
        if (listStr != null && !listStr.isEmpty()) {
            // Split the comma-separated list into items and add them to the list
            String[] items = listStr.split(",");
            for (String item : items) {
                list.add(item.trim()); // Trim any leading or trailing spaces
            }
        }
        return list;
    }

    private static void parseLabelOption(String labelsOption) {
        // Now only single-hop label predicates are supported; comma-separated list of URIs
        label_predicates = new HashSet<>();
        if (labelsOption == null || labelsOption.isEmpty()) return;
        for (String item : labelsOption.split(",")) {
            String uri = item.trim();
            if (!uri.isEmpty()) label_predicates.add(uri);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: ElementsExtractor -i <inputDirOrFile> -o <outputDir> -kg <knowledgeGraphType> [-l <LABEL_URIs>] [-d <DESC_URIs>] [-c <CLASS_URIs>] [-t <threads>] [-f] [--entity_prefixes <prefix1,prefix2>] [--class_prefixes <prefix1,prefix2>]");
            System.exit(1);
        }

        // Create the command-line parser
        CommandLineParser parser = new DefaultParser();
        Options options = new Options();

        // Define options for the arguments
        options.addRequiredOption("i", "input", true, "Input directory/file");
        options.addRequiredOption("o", "output", true, "Output directory (the files entities.csv and classes.txt will be added)");
        options.addRequiredOption("kg", "knowledge_graph", true, "The type of knowledge graph (default: generic)");
        options.addOption("l", "labels", true, "The URIs of predicates that denote a node label");
        options.addOption("nel", "no-entity-labels", false, "Used to indicate that entity labels should be extracted from URIs");
        options.addOption("npl", "no-predicate-labels", false, "Used to indicate that predicate labels should be extracted from URIs");
        options.addOption("ncl", "no-class-labels", false, "Used to indicate that class labels should be extracted from URIs");
        options.addOption("d", "descriptions", true, "The URIs of predicates that denote a node description");
        options.addOption("c", "classes", true, "The URIs of predicates that denote that a node is a class");
        options.addOption("t", "threads", true, "Number of threads");
        options.addOption("f", "filter", false, "Filter nodes without labels");
        options.addOption("fc", "filter-classes", false, "Filter classes without labels");
        options.addOption("ep", "entity_prefixes", true, "A comma-separated list of possible substrings in entity URIs (used to filter entities)");
        options.addOption("cp", "class_prefixes", true, "A comma-separated list of possible substrings in class URIs (used to filter classes)");
        
        options.addOption("mem", "memory-optimization", false, "Option to use less memory (still entirealy on RAM).");
        options.addOption("compref", "common-prefixes", true, "A comma-separated list of common prefixes. Used to optimize memory usage");

        Path inputFile = null;
        Path outputDir = null;
        Path outputEntities = null;
        Path outputEntitiesLabels = null;
        Path outputEntitiesFull = null;
        Path outputClasses = null;
        Path outputClassesLabels = null;
        Path outputClassesFull = null;
        Path outputPredicates = null;
        Path outputPredicatesLabels = null;
        Path outputPredicatesFull = null;
        String kgType = "";
        int numThreads = -1;

        try {
            // Parse the command-line arguments
            CommandLine cmd = parser.parse(options, args);

            // Get input and output paths
            inputFile = Paths.get(cmd.getOptionValue("i"));
            outputDir = Paths.get(cmd.getOptionValue("o"));
            Files.createDirectories(outputDir);

            kgType = cmd.getOptionValue("kg");
            if (!kgType.equals("wikidata") && !kgType.equals("dbpedia") && !kgType.equals("freebase") && !kgType.equals("generic")) {
                System.err.println("Unsupported knowledge graph type: " + kgType);
                System.exit(1);
            }

            outputEntities = outputDir.resolve("entities.tsv");
            outputEntitiesLabels = outputDir.resolve("entities_labels.tsv");
            outputEntitiesFull = outputDir.resolve("entities_full.tsv");
            outputClasses = outputDir.resolve("classes.tsv");
            outputClassesLabels = outputDir.resolve("classes_labels.tsv");
            outputClassesFull = outputDir.resolve("classes_full.tsv");
            outputPredicates = outputDir.resolve("predicates.tsv");
            outputPredicatesLabels = outputDir.resolve("predicates_labels.tsv");
            outputPredicatesFull = outputDir.resolve("predicates_full.tsv");

            parseLabelOption(cmd.getOptionValue("l"));
            no_entity_labels = cmd.hasOption("nel");
            no_predicate_labels = cmd.hasOption("npl");
            no_class_labels = cmd.hasOption("ncl");
            classes_predicates = cmd.hasOption("c") ? new HashSet<>(parseList(cmd.getOptionValue("c"))) : new HashSet<>();
            description_predicates = cmd.hasOption("d") ? new HashSet<>(parseList(cmd.getOptionValue("d"))) : new HashSet<>();
            entity_prefixes = cmd.hasOption("ep") ? parseList(cmd.getOptionValue("ep")) : new ArrayList<>();
            class_prefixes = cmd.hasOption("cp") ? parseList(cmd.getOptionValue("cp")) : new ArrayList<>();

            numThreads = cmd.hasOption("t") ? Integer.parseInt(cmd.getOptionValue("t")) : Runtime.getRuntime().availableProcessors();
            filter_entities = cmd.hasOption("f") ? true : false;
            filter_classes = cmd.hasOption("fc") ? true : false;

            memoryOptimized = cmd.hasOption("mem") ? true : false;
            if (cmd.hasOption("compref")) {
                commonPrefixes = parseList(cmd.getOptionValue("compref"));
            }

            // Output parsed values
            System.out.println("Input File: " + inputFile);
            System.out.println("Output Dir: " + outputDir);
            System.out.println("Knowledge Graph Type: " + kgType);
            System.out.println("Number of Threads: " + numThreads);
            System.out.println("Node Labels: " + label_predicates);
            System.out.println("No Entity Labels: " + no_entity_labels);
            System.out.println("No Predicate Labels: " + no_predicate_labels);
            System.out.println("No Class Labels: " + no_class_labels);
            System.out.println("Node Descriptions: " + description_predicates);
            System.out.println("Entity Prefixes: " + entity_prefixes);
            System.out.println("Classes: " + classes_predicates);
            System.out.println("Class Prefixes: " + class_prefixes);
            System.out.println("Filter Entities without Labels: " + filter_entities);
            System.out.println("Filter Classes without Labels: " + filter_classes);
            
            System.out.println("Memory Optimization: " + memoryOptimized);
            System.out.println("Common Prefixes: " + commonPrefixes);
        } catch (Exception e) {
            System.err.println("Error parsing arguments: " + e.getMessage());
            new HelpFormatter().printHelp("EntityExtractor", options);
            System.exit(1);
        }

        // Clear output files
        for (Path p : new Path[]{
                outputEntities, outputEntitiesLabels, outputEntitiesFull,
                outputClasses, outputClassesLabels, outputClassesFull,
                outputPredicates, outputPredicatesLabels, outputPredicatesFull
        }) {
            Files.deleteIfExists(p);
            Files.createFile(p);
        }

        // Process files in parallel
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CompletionService<List<UriData>> completionService = 
            new ExecutorCompletionService<>(executor);

        AtomicInteger fileCount = new AtomicInteger(0);
        if (Files.isDirectory(inputFile)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(inputFile)) {
                for (Path file : stream) {
                    if (Files.isRegularFile(file)) {
                        fileCount.incrementAndGet();
                        System.out.println("Processing input file: " + file);
                        completionService.submit(() -> processFile(file));
                    }
                }
            }
        } else if (Files.isRegularFile(inputFile)) {
            // If input is a file, process it directly
            final var inputFileTemp = inputFile; // just to bypass  the need for final
            fileCount.incrementAndGet();
            System.out.println("Processing input file: " + inputFileTemp);
            completionService.submit(() -> processFile(inputFileTemp));
        } else {
            System.err.println("Input must be a directory or a file.");
            System.exit(1);
        }

        // Aggregate results and then write categorized TSV outputs
        Map<String, UriData> all = new LinkedHashMap<>();
        try {
            for (int i = 0; i < fileCount.get(); i++) {
                try {
                    List<UriData> results = completionService.take().get();
                    System.out.println("Merging results from file " + (i + 1) + "/" + fileCount.get());
                    for (UriData d : results) {
                        var uri = d.getUri();
                        UriData existing = all.get(uri);
                        if (existing == null) {
                            all.put(uri, d);
                        } else {
                            // Merge
                            existing.isPredicate |= d.isPredicate;
                            existing.isClass |= d.isClass;
                            existing.isSubClass |= d.isSubClass;
                            existing.isRedirect |= d.isRedirect;
                            existing.isInternalNode |= d.isInternalNode;
                            var types = d.getTypes();
                            if (types != null) {
                                // existing.types.addAll(d.types);
                                for (String t : types) {
                                    existing.addType(t);
                                }
                            }
                            var superClasses = d.getSuperClasses();
                            if (superClasses != null) {
                                // existing.superClasses.addAll(d.superClasses);
                                for (String t : superClasses) {
                                    existing.addSuperClass(t);
                                }
                            }
                            existing.satisfiesEntityPrefixes |= d.satisfiesEntityPrefixes;
                            existing.satisfiesClassPrefixes |= d.satisfiesClassPrefixes;
                            existing.outgoingLinks += d.outgoingLinks;
                            existing.incomingLinks += d.incomingLinks;
                            if (d.labels != null) {
                                for (String l : d.labels) {
                                    if (!existing.labels.contains(l)) existing.addLabel(l);
                                }
                            }
                            if (existing.description == null || existing.description.isEmpty()) {
                                if (d.description != null && !d.description.isEmpty()) existing.description = d.description;
                                else if (existing.fallbackDescription.isEmpty()) existing.fallbackDescription = d.fallbackDescription;
                            }
                            if (existing.fallbackLabel.isEmpty() && d.fallbackLabel != null && !d.fallbackLabel.isEmpty()) {
                                existing.fallbackLabel = d.fallbackLabel;
                            }
                            existing.finalizeValues();
                        }
                    }
                } catch (ExecutionException e) {
                    System.err.println("Error processing file: " + e.getCause().getMessage());
                }
            }

            // Prepare writers
            try (BufferedWriter entW = Files.newBufferedWriter(outputEntities);
                 BufferedWriter entLblW = Files.newBufferedWriter(outputEntitiesLabels);
                 BufferedWriter entFullW = Files.newBufferedWriter(outputEntitiesFull);
                 BufferedWriter clsW = Files.newBufferedWriter(outputClasses);
                 BufferedWriter clsLblW = Files.newBufferedWriter(outputClassesLabels);
                 BufferedWriter clsFullW = Files.newBufferedWriter(outputClassesFull);
                 BufferedWriter predW = Files.newBufferedWriter(outputPredicates);
                 BufferedWriter predLblW = Files.newBufferedWriter(outputPredicatesLabels);
                 BufferedWriter predFullW = Files.newBufferedWriter(outputPredicatesFull);
                 BufferedWriter fullW = Files.newBufferedWriter(outputDir.resolve("all.tsv"));
                 BufferedWriter fullWithTypesW = Files.newBufferedWriter(outputDir.resolve("all_with_types.tsv"));
                 BufferedWriter uriLabelsW = Files.newBufferedWriter(outputDir.resolve("uri_labels.tsv"))) {

                for (UriData d : all.values()) {
                    // Ensure preferred values
                    d.finalizeValues();

                    System.out.println("Processing URI: " + d.getUri());

                    if (d.isRedirect) {
                        // Skip redirects entirely
                        continue;
                    }

                    if (d.isInternalNode) {
                        // Skip Wikimedia internal nodes entirely
                        continue;
                    }

                    if (d.isPredicate && kgType.equals("wikidata")) {
                        var split_string = d.getUri().split("/");
                        var pid = split_string[split_string.length - 1];
                        if (all.containsKey("http://www.wikidata.org/entity/" + pid)) {
                            var entityData = all.get("http://www.wikidata.org/entity/" + pid);
                            d.bestLabel = entityData.bestLabel;
                            d.bestDescription = entityData.bestDescription;
                        }
                    }

                    if (kgType.equals("wikidata") && d.isClass && !d.isSubClass) {
                        // In Wikidata, classes only if they are also subclasses
                        d.isClass = false;
                    }

                    if (kgType.equals("freebase") && d.isClass) {
                        // In Freebase, we generate labels for classes that are not human readable
                        if (d.getUri().contains("ns/m.") == false) {
                            d.bestLabel = "";
                        }
                    }

                    String label = d.bestLabel;
                    String desc = sanitizeTSV(d.bestDescription);
                    boolean labelIsEmpty = (label == null || label.isEmpty());
                    boolean generate_label = ((d.isPredicate && no_predicate_labels) ||
                                              (d.isClass && no_class_labels) ||
                                              (!d.isPredicate && !d.isClass && no_entity_labels) ||
                                              (labelIsEmpty));

                    // Only generate label from URI for classes and predicates
                    if (generate_label) {
                        String uri = d.getUri();
                        int lastSlash = uri.lastIndexOf('/');
                        int lastHash = uri.lastIndexOf('#');
                        int splitIdx = Math.max(lastSlash, lastHash);
                        String localName = (splitIdx >= 0 && splitIdx < uri.length() - 1) ? uri.substring(splitIdx + 1) : uri;
                        localName = localName.replace('_', ' ');
                        localName = localName.replace('.', ' ');
                        localName = localName.replaceAll("([a-z])([a-z])([A-Z])", "$1$2 $3"); // Split camel case
                        label = localName.trim();
                        d.addLabel(label);
                    } else {
                        label = sanitizeTSV(label);
                    }

                    label = label.toLowerCase();

                    label = sanitizeTSV(label);
                    desc = sanitizeTSV(desc);

                    if (d.isPredicate) {
                        d.bestLabel = label;
                        d.bestDescription = desc;
                        // Always include predicates
                        var uri = d.getUri();
                        predW.write(uri);
                        predW.newLine();
                        predLblW.write(uri + "\t" + label);
                        predLblW.newLine();
                        predFullW.write(uri + "\t" + label + "\t" + desc);
                        predFullW.newLine();
                        uriLabelsW.write(uri + "\t" + label);
                        uriLabelsW.newLine();
                    } else if (d.isClass) {
                        if (!class_prefixes.isEmpty() && !d.satisfiesClassPrefixes) continue;
                        if (filter_classes && labelIsEmpty) continue;
                        d.bestLabel = label;
                        d.bestDescription = desc;
                        var uri = d.getUri();
                        clsW.write(uri);
                        clsW.newLine();
                        clsLblW.write(uri + "\t" + label);
                        clsLblW.newLine();
                        clsFullW.write(uri + "\t" + label + "\t" + desc);
                        clsFullW.newLine();
                        uriLabelsW.write(uri + "\t" + label);
                        uriLabelsW.newLine();
                    } else { // entity
                        System.out.println("Processing entity: " + d.getUri());
                        System.out.println("  Label: " + label);
                        if (!entity_prefixes.isEmpty() && !d.satisfiesEntityPrefixes) continue;
                        if (filter_entities && labelIsEmpty) continue;
                        System.out.println("  Passed filters");
                        var uri = d.getUri();
                        if (kgType.equals("wikidata") && isWikidataPropertyEntity(uri)) continue;
                        d.bestLabel = label;
                        d.bestDescription = desc;
                        entW.write(uri);
                        entW.newLine();
                        entLblW.write(uri + "\t" + d.labels.stream().collect(Collectors.joining("##")));
                        entLblW.newLine();
                        entFullW.write(uri + "\t" + d.labels.stream().collect(Collectors.joining("##")) + "\t" + desc);
                        entFullW.newLine();
                        uriLabelsW.write(uri + "\t" + d.labels.stream().collect(Collectors.joining("##")));
                        uriLabelsW.newLine();
                    }
                }

                for (UriData d : all.values()) {
                    if (d.isRedirect) {
                        // Skip redirects entirely
                        continue;
                    }

                    if (d.isInternalNode) {
                        // Skip Wikimedia internal nodes entirely
                        continue;
                    }

                    if (d.isClass) {
                        if (!class_prefixes.isEmpty() && !d.satisfiesClassPrefixes) continue;
                        if (filter_classes && d.labels.isEmpty()) continue;
                    } else if (!d.isPredicate) {
                        if (!entity_prefixes.isEmpty() && !d.satisfiesEntityPrefixes) continue;
                        if (filter_entities && d.labels.isEmpty()) continue;
                        if (kgType.equals("wikidata") && isWikidataPropertyEntity(d.getUri())) continue;
                    }

                    // Write to full TSV
                    fullW.write(d.getUri() + "\t" + d.getType() + "\t" + d.labels.stream().collect(Collectors.joining("##")) + "\t" + d.bestDescription + "\t" + d.outgoingLinks + "\t" + d.incomingLinks);
                    fullWithTypesW.write(d.getUri() + "\t" + d.getType() + "\t" + d.labels.stream().collect(Collectors.joining("##")) + "\t" + d.bestDescription + "\t" + d.outgoingLinks + "\t" + d.incomingLinks);
                    // Write entity types
                    if (d.isClass) {
                        ArrayList<UriData> top_3_classes = new ArrayList<>();
                        for (var type : d.getSuperClasses()) {
                            var typeData = all.get(type);
                            if (typeData != null) {
                                if (typeData.bestLabel == null || typeData.bestLabel.isEmpty()) continue;
                                var popularity = typeData.popularity();
                                if (top_3_classes.size() < 3) {
                                    top_3_classes.add(typeData);
                                } else {
                                    // Find the least popular in top_3_types
                                    int minIndex = 0;
                                    int minPopularity = top_3_classes.get(0).popularity();
                                    for (int i = 1; i < top_3_classes.size(); i++) {
                                        int currentPopularity = top_3_classes.get(i).popularity();
                                        if (currentPopularity < minPopularity) {
                                            minPopularity = currentPopularity;
                                            minIndex = i;
                                        }
                                    }
                                    if (popularity > minPopularity) {
                                        top_3_classes.set(minIndex, typeData);
                                    }
                                }
                            }
                        }
                        // Sort top_3_types by popularity descending
                        top_3_classes.sort((a, b) -> Integer.compare(b.popularity(), a.popularity()));
                        // Collect top-3 type URIs instead of labels
                        var top_3_types_str = top_3_classes.stream()
                            .map(t -> t.bestLabel)
                            .collect(Collectors.joining("|"));
                        fullW.write("\t" + top_3_types_str);
                        fullWithTypesW.write("\t" + d.getSuperClasses().stream().collect(Collectors.joining("|")));
                    } else if (!d.isPredicate) {
                        ArrayList<UriData> top_3_types = new ArrayList<>();
                        for (var type : d.getTypes()) {
                            var typeData = all.get(type);
                            if (typeData != null) {
                                if (typeData.bestLabel == null || typeData.bestLabel.isEmpty()) continue;
                                var popularity = typeData.popularity();
                                if (top_3_types.size() < 3) {
                                    top_3_types.add(typeData);
                                } else {
                                    // Find the least popular in top_3_types
                                    int minIndex = 0;
                                    int minPopularity = top_3_types.get(0).popularity();
                                    for (int i = 1; i < top_3_types.size(); i++) {
                                        int currentPopularity = top_3_types.get(i).popularity();
                                        if (currentPopularity < minPopularity) {
                                            minPopularity = currentPopularity;
                                            minIndex = i;
                                        }
                                    }
                                    if (popularity > minPopularity) {
                                        top_3_types.set(minIndex, typeData);
                                    }
                                }
                            }
                        }
                        // Sort top_3_types by popularity descending
                        top_3_types.sort((a, b) -> Integer.compare(b.popularity(), a.popularity()));
                        // Collect top-3 type URIs instead of labels
                        var top_3_types_str = top_3_types.stream()
                            .map(t -> t.bestLabel)
                            .collect(Collectors.joining("|"));
                        fullW.write("\t" + top_3_types_str);
                        fullWithTypesW.write("\t" + d.getTypes().stream().collect(Collectors.joining("|")));
                    }
                    fullW.newLine();
                    fullWithTypesW.newLine();
                }
            }
        } finally {
            executor.shutdown();
            executor.awaitTermination(100, TimeUnit.HOURS);
        }
    }

    private static List<UriData> processFile(Path file) throws IOException {
        // Single-threaded per parsed file; HashMap is sufficient and faster
        Map<String, UriData> uris = new HashMap<>();

        System.out.println("Processing file: " + file);

        StreamRDF processor = new StreamRDF() {
            @Override
            public void triple(Triple triple) {
                processTriple(triple, uris);
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

        System.out.println("Finished parsing file: " + file);

        System.out.println("Found " + uris.size() + " unique URIs.");

        return uris.values().stream()
            .peek(UriData::finalizeValues)
            .collect(Collectors.toList());
    }

    private static boolean isSubClassPredicate(String predicate) {
        // Add more subclass predicates if needed
        return "http://www.wikidata.org/prop/direct/P279".equals(predicate) || "http://www.w3.org/2000/01/rdf-schema#subClassOf".equals(predicate);
    }

    private static boolean isClassPredicate(String predicate) {
        if (!classes_predicates.isEmpty()) {
            return classes_predicates.contains(predicate);
        }
        return RDF_TYPE.equals(predicate) || WD_P31.equals(predicate);
    }

    private static boolean isRdfsClass(String object) {
        return "http://www.w3.org/2000/01/rdf-schema#Class".equals(object);
    }

    private static boolean isRedirectPredicate(String predicate) {
        // Add more redirect predicates if needed
        return "http://dbpedia.org/ontology/wikiPageRedirects".equals(predicate);
    }

    private static boolean isInternalClass(String uri) {
        if ("http://www.wikidata.org/entity/Q11266439".equals(uri)) return true; // Wikimedia internal class
        if ("http://www.wikidata.org/entity/Q13406463".equals(uri)) return true; // Wikimedia internal class
        if ("http://www.wikidata.org/entity/Q4167836".equals(uri)) return true; // Wikimedia disambiguation class
        if ("http://www.wikidata.org/entity/Q4167410".equals(uri)) return true;
        if ("http://www.wikidata.org/entity/Q14204246".equals(uri)) return true;
        if ("http://www.wikidata.org/entity/Q15184295".equals(uri)) return true;
        return false;
    }

    private static boolean isWikidataPropertyEntity(String uri) {
		if (uri == null) return false;
		return WD_ENTITY_PROP.matcher(uri).matches();
	}

	private static String extractPid(String uri) {
		Matcher m = WD_ENTITY_PROP.matcher(uri);
		return m.matches() ? m.group(1) : null;
	}

    private static void updatePrefixFlags(UriData d, String uri) {
        if (!entity_prefixes.isEmpty() && matchesAnyPrefix(uri, entity_prefixes)) {
            d.satisfiesEntityPrefixes = true;
        }
        if (!class_prefixes.isEmpty() && matchesAnyPrefix(uri, class_prefixes)) {
            d.satisfiesClassPrefixes = true;
        }
    }

    private static void processTriple(Triple triple, Map<String, UriData> uris) {
        // 1) Subject must be a URI
        if (!triple.getSubject().isURI()) return;

        String s = triple.getSubject().getURI();

        UriData sd = uris.computeIfAbsent(s, UriData::new);
        // Track subject prefix satisfaction for both entity and class filtering (prefixes are URI-based)
        updatePrefixFlags(sd, s);

        sd.outgoingLinks += 1;

        // 2) Predicate bookkeeping
        String p = triple.getPredicate().getURI();
        UriData pd = uris.computeIfAbsent(p, UriData::new);
        pd.isPredicate = true;
        pd.outgoingLinks += 1;
        pd.incomingLinks += 1;

        if (isRedirectPredicate(p)) {
            sd.isRedirect = true;
        }

        if (isSubClassPredicate(p)) {
            sd.isSubClass = true;
            if (triple.getObject().isURI())
                sd.addSuperClass(sanitizeTSV(triple.getObject().getURI()));
        }

        boolean isLabel = label_predicates.contains(p);
        boolean isDesc  = description_predicates.contains(p);
        boolean isClassPred = isClassPredicate(p);

        // 3) Object handling (URI vs literal)
        if (triple.getObject().isURI()) {
            String o = triple.getObject().getURI();
            UriData od = uris.computeIfAbsent(o, UriData::new);

            od.incomingLinks += 1;

            // Track both prefix types for object URIs (independent of class/entity role)
            updatePrefixFlags(od, o);
            // Mark classes
            if (isClassPred) {
                od.isClass = true;
                if (isInternalClass(o)) {
                    sd.isInternalNode = true;
                }
                sd.addType(sanitizeTSV(o));
                if (isRdfsClass(o)) {
                    od.isClass = true;
                    od.isSubClass = true;
                }
            }

        }

        // 4) Early exits for speed
        if (filter_entities && !(isLabel || isDesc)) return;   // ignore non-label/desc data
        if (isClassPred) return;                                // don't treat typing triples as labels

        // 5) Labels / Descriptions (literals only)
        if (!triple.getObject().isLiteral()) return;

        String value = triple.getObject().getLiteralLexicalForm();
        String lang  = triple.getObject().getLiteralLanguage();
        boolean isEnglish = lang.isEmpty() || "en".equals(lang);

        if (isLabel) {
            if (isEnglish && value.length() > 1) {
                if (!sd.labels.contains(value)) sd.addLabel(value);
            } 
            // else if (!isEnglish && sd.fallbackLabel.isEmpty()) {
            //     sd.fallbackLabel = value;
            // }
        } else if (isDesc) {
            if (isEnglish && value.length() > 1) {
                if (sd.description.isEmpty())
                    sd.description = value;
                else if (sd.description.length() < value.length())
                    sd.description = value;
            } 
            // else if (!isEnglish && sd.fallbackDescription.isEmpty()) {
            //     sd.fallbackDescription = value;
            // }
        }
    }

    // Cheap helper to avoid stream overhead in hot path
    private static boolean matchesAnyPrefix(String uri, List<String> prefixes) {
        for (String p : prefixes) {
            if (uri.contains(p)) return true;
        }
        return false;
    }

    public static String sanitizeTSV(String s) {
        if (s == null) return "";
        return s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').trim();
    }

    public static String shortUri(String uri) {
        if (uri == null) return "";
        for (int i = 0; i < commonPrefixes.size(); i++) {
            String prefix = commonPrefixes.get(i);
            if (uri.startsWith(prefix)) {
                // Encode the matched prefix index to make restoration lossless.
                return "@" + i + ":" + uri.substring(prefix.length());
            }
        }
        return uri;
    }

    public static String restoreShortUri(String shortUri) {
        if (shortUri == null) return "";
        if (shortUri.startsWith("@")) {
            int sep = shortUri.indexOf(':', 1);
            if (sep > 1) {
                try {
                    int idx = Integer.parseInt(shortUri.substring(1, sep));
                    if (idx >= 0 && idx < commonPrefixes.size()) {
                        return commonPrefixes.get(idx) + shortUri.substring(sep + 1);
                    }
                } catch (NumberFormatException ignored) {
                    System.err.println("Error restoring short URI: " + shortUri);
                    // Fall through to returning the input as-is.
                }
            }
        }
        return shortUri;
    }

}