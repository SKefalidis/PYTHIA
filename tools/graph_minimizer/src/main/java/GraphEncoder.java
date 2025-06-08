import org.apache.jena.graph.Node;
import org.apache.jena.graph.Triple;
import org.apache.jena.riot.RDFDataMgr;
import org.apache.jena.riot.system.StreamRDF;
import org.apache.jena.riot.system.StreamRDFBase;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

public class GraphEncoder {

    // In-memory map keyed by URI/blank-node string to trim Node object overhead.
    // For BILLIONS of nodes, consider a persistent Map (like MapDB or ChronicleMap).
    private static final Map<String, Integer> nodeMap = new HashMap<>();
    private static int counter = 0;
    
    // Stats
    private static long tripleCount = 0;
    private static long skippedCount = 0;

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java GraphEncoder <input_file_or_dir> <output_dir>");
            return;
        }

        String inputPath = args[0];
        String outputEncoded = args[1] + "/graph_encoded.txt";
        String outputMapping = args[1] + "/entity_mapping.tsv";

        long startTime = System.currentTimeMillis();

        // Ensure output directory exists
        new File(args[1]).mkdirs();

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputEncoded))) {
            
            // Create the StreamRDF handler
            StreamRDF encoder = new StreamRDFBase() {
                @Override
                public void triple(Triple triple) {
                    // Filter: drop triples containing literals or blank nodes anywhere
                    if (triple.getObject().isLiteral()
                            || triple.getSubject().isBlank()
                            || triple.getPredicate().isBlank()
                            || triple.getObject().isBlank()) {
                        skippedCount++;
                        return;
                    }

                    try {
                        int sId = getId(triple.getSubject());
                        int pId = getId(triple.getPredicate());
                        int oId = getId(triple.getObject());

                        // graph-tool CSV format expects: source target edgeProp(s)
                        writer.write(sId + " " + oId + " " + pId);
                        writer.newLine();
                        
                        tripleCount++;
                        if (tripleCount % 500_000 == 0) {
                            System.out.print("Processed " + tripleCount + " triples...\r");
                        }
                    } catch (IOException e) {
                        throw new RuntimeException("Error writing to output file", e);
                    }
                }
            };

            // Handle Input (File or Directory)
            File input = new File(inputPath);
            if (input.isFile()) {
                System.out.println("Parsing single file: " + input.getName());
                RDFDataMgr.parse(encoder, input.getAbsolutePath());
            } else if (input.isDirectory()) {
                try (Stream<Path> paths = Files.list(Paths.get(inputPath))) {
                    paths.filter(p -> p.toString().endsWith(".nt"))
                         .sorted() // Deterministic order
                         .forEach(path -> {
                             System.out.println("Parsing: " + path.getFileName());
                             RDFDataMgr.parse(encoder, path.toString());
                         });
                }
            } else {
                System.err.println("Invalid input path.");
                return;
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        // Write the Dictionary
        System.out.println("\nWriting ID mapping to " + outputMapping + "...");
        saveMapping(outputMapping);

        long endTime = System.currentTimeMillis();
        System.out.println("------------------------------------------------");
        System.out.println("Finished in " + (endTime - startTime) / 1000 + "s");
        System.out.println("Valid Triples: " + tripleCount);
        System.out.println("Skipped Literals: " + skippedCount);
        System.out.println("Unique Entities: " + counter);
    }

    private static int getId(Node node) {
        // computeIfAbsent is slightly slower than get/put pattern in tight loops
        // but cleaner. For max speed, use get -> if null -> put.
        String key = nodeKey(node);
        Integer id = nodeMap.get(key);
        if (id == null) {
            id = counter++;
            nodeMap.put(key, id);
        }
        return id;
    }

    private static String nodeKey(Node node) {
        if (node.isURI()) {
            return node.getURI();
        }
        if (node.isBlank()) {
            return "_:" + node.getBlankNodeLabel();
        }
        return node.toString();
    }

    private static void saveMapping(String filename) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write("ID\tURI\n");
            // Iterate entry set
            for (Map.Entry<String, Integer> entry : nodeMap.entrySet()) {
                writer.write(entry.getValue() + "\t" + entry.getKey());
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}