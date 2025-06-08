import org.apache.commons.cli.*;
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

public class UriLabelExtractor {
	// Set of label predicates (e.g., rdfs:label, skos:prefLabel, etc.)
	static Set<String> labelPredicates = new HashSet<>();

	private static ArrayList<String> parseList(String listStr) {
		var list = new ArrayList<String>();
		if (listStr != null && !listStr.isEmpty()) {
			String[] items = listStr.split(",");
			for (String item : items) {
				list.add(item.trim());
			}
		}
		return list;
	}

	private static class FileResult {
		// Map: URI -> best label (English preferred; if multiple labels exist, keep the first observed)
		final Map<String, String> labels = new HashMap<>();
		// Track fallbacks when no English label is present
		final Map<String, String> fallbackLabels = new HashMap<>();
	}

	public static void main(String[] args) throws Exception {
		// CLI: -i input(file or dir) -o output.tsv -l label1,label2,... [-t threads]
		CommandLineParser parser = new DefaultParser();
		Options options = new Options();
		options.addRequiredOption("i", "input", true, "Input directory or file (N-Triples)");
		options.addRequiredOption("o", "output", true, "Output TSV file (URI\tLabel)");
		options.addOption("l", "labels", true, "Comma-separated list of label predicate URIs");
		options.addOption("t", "threads", true, "Number of parallel threads");
		options.addOption("en", "english-only", false, "Only extract English labels");

		Path inputPath;
		Path outputPath;
		int numThreads;
		boolean englishOnly;
		try {
			CommandLine cmd = parser.parse(options, args);
			inputPath = Paths.get(cmd.getOptionValue("i"));
			outputPath = Paths.get(cmd.getOptionValue("o"));
			labelPredicates = new HashSet<>(parseList(cmd.getOptionValue("l")));
			numThreads = cmd.hasOption("t") ? Integer.parseInt(cmd.getOptionValue("t")) : Runtime.getRuntime().availableProcessors();
			englishOnly = cmd.hasOption("en");

			System.out.println("Input: " + inputPath);
			System.out.println("Output: " + outputPath);
			System.out.println("Threads: " + numThreads);
			System.out.println("Label predicates: " + labelPredicates);
			System.out.println("English only: " + englishOnly);
		} catch (Exception e) {
			System.err.println("Error parsing arguments: " + e.getMessage());
			new HelpFormatter().printHelp("UriLabelExtractor", options);
			return;
		}

		// Ensure output file exists and is empty
		Files.deleteIfExists(outputPath);
		Files.createDirectories(outputPath.getParent() == null ? Paths.get(".") : outputPath.getParent());
		Files.createFile(outputPath);

		// Process files
		List<Path> files = new ArrayList<>();
		if (Files.isDirectory(inputPath)) {
			try (DirectoryStream<Path> stream = Files.newDirectoryStream(inputPath)) {
				for (Path f : stream) {
					if (Files.isRegularFile(f)) files.add(f);
				}
			}
		} else if (Files.isRegularFile(inputPath)) {
			files.add(inputPath);
		} else {
			System.err.println("Input must be a directory or a file");
			return;
		}

		ExecutorService exec = Executors.newFixedThreadPool(numThreads);
		CompletionService<FileResult> cs = new ExecutorCompletionService<>(exec);
		AtomicInteger submitted = new AtomicInteger();
		for (Path f : files) {
			submitted.incrementAndGet();
			cs.submit(() -> processFile(f));
		}

		// Merge all results, preferring English labels when available
		Map<String, String> uriToLabel = new HashMap<>();
		Map<String, String> uriToFallback = new HashMap<>();
		for (int i = 0; i < submitted.get(); i++) {
			try {
				FileResult fr = cs.take().get();
				for (Map.Entry<String, String> e : fr.labels.entrySet()) {
					// English label preferred - if present, set and lock-in
					uriToLabel.putIfAbsent(e.getKey(), e.getValue());
				}
				for (Map.Entry<String, String> e : fr.fallbackLabels.entrySet()) {
					uriToFallback.putIfAbsent(e.getKey(), e.getValue());
				}
			} catch (ExecutionException ee) {
				System.err.println("Error processing a file: " + ee.getCause());
			}
		}
		exec.shutdown();
		exec.awaitTermination(1, TimeUnit.HOURS);

		// Write TSV: URI\tLabel; if no label found, leave blank
		// try (BufferedWriter w = Files.newBufferedWriter(outputPath)) {
		// 	// To cover URIs that may only appear as objects with labels, union keys
		// 	Set<String> allUris = new HashSet<>();
		// 	allUris.addAll(uriToLabel.keySet());
		// 	allUris.addAll(uriToFallback.keySet());

		// 	// If no explicit label list is supplied, we still want to include any subject URIs seen
		// 	// But the spec says: goes through every URI in source files and finds its label. We'll include
		// 	// those for which we saw no label triple as empty lines only when they appeared as subject/predicate/object.
		// 	// This implementation only knows URIs that had a label triple; to expand coverage we need to parse all URIs.
		// 	// Do one lightweight pass to collect all URIs and ensure they appear in output.
		// }

		// Second pass to collect all URIs to ensure an entry even without labels
		Set<String> allUris = collectAllUris(files);

		try (BufferedWriter w = Files.newBufferedWriter(outputPath)) {
			for (String uri : allUris) {
				if (englishOnly && !uriToLabel.containsKey(uri)) {
					// Skip URIs without English labels if the flag is set
					continue;
				}
				String label = uriToLabel.getOrDefault(uri, uriToFallback.getOrDefault(uri, ""));
				w.write(uri);
				w.write("\t");
				if (!label.isEmpty()) {
					// Sanitize newlines
					w.write(label.replace('\n', ' '));
				}
				w.newLine();
			}
		}
	}

	private static FileResult processFile(Path file) throws IOException {
		FileResult result = new FileResult();
		StreamRDF processor = new StreamRDF() {
			@Override
			public void triple(Triple triple) {
				if (!triple.getSubject().isURI()) return; // skip blank nodes
				String subj = triple.getSubject().getURI();
				String pred = triple.getPredicate().getURI();

				if (labelPredicates.isEmpty() || labelPredicates.contains(pred)) {
					if (triple.getObject().isLiteral()) {
						String lang = triple.getObject().getLiteralLanguage();
						String value = triple.getObject().getLiteralLexicalForm();
						if (value != null && value.length() > 0) {
							boolean isEnglish = lang == null || lang.isEmpty() || lang.equals("en");
							if (isEnglish) {
								// Prefer first seen English label
								result.labels.putIfAbsent(subj, value);
							} else {
								// Keep a fallback if no English exists
								result.fallbackLabels.putIfAbsent(subj, value);
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

		RDFParser.source(file.toString()).lang(Lang.NTRIPLES).parse(processor);
		return result;
	}

	private static Set<String> collectAllUris(List<Path> files) throws InterruptedException {
		// Collect all URIs from subject, predicate, and object positions across all files
		Set<String> uris = ConcurrentHashMap.newKeySet();
		int threads = Math.min(files.size(), Math.max(1, Runtime.getRuntime().availableProcessors()));
		ExecutorService exec = Executors.newFixedThreadPool(threads);
		CountDownLatch latch = new CountDownLatch(files.size());

		for (Path file : files) {
			exec.submit(() -> {
				try {
					StreamRDF processor = new StreamRDF() {
						@Override
						public void triple(Triple triple) {
							if (triple.getSubject().isURI()) uris.add(triple.getSubject().getURI());
							uris.add(triple.getPredicate().getURI());
							if (triple.getObject().isURI()) uris.add(triple.getObject().getURI());
						}
						@Override public void start() {}
						@Override public void quad(Quad quad) {}
						@Override public void base(String base) {}
						@Override public void prefix(String prefix, String iri) {}
						@Override public void finish() {}
					};
					RDFParser.source(file.toString()).lang(Lang.NTRIPLES).parse(processor);
				} finally {
					latch.countDown();
				}
			});
		}
		latch.await(1, TimeUnit.HOURS);
		exec.shutdown();
		exec.awaitTermination(1, TimeUnit.HOURS);
		return uris;
	}
}

