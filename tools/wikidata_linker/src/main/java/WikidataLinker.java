import org.apache.commons.cli.*;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Post-processing utility for Wikidata predicate linking.
 *
 * Wikidata property URIs appear as:
 *   - Predicates in triples:  http://www.wikidata.org/prop/direct/PXXXX
 *                             http://www.wikidata.org/prop/direct-normalized/PXXXX
 *   - Subjects describing properties: http://www.wikidata.org/entity/PXXXX
 *
 * ElementsExtractor will put /entity/PXXXX rows (with labels/descriptions) into entities TSVs.
 * This tool moves those rows into the predicates TSVs and writes them as /prop/direct and
 * /prop/direct-normalized with the same label/description. It removes the /entity/PXXXX rows
 * from entity TSVs.
 */
public class WikidataLinker {

	private static final Pattern WD_ENTITY_PROP = Pattern.compile("^https?://www\\.wikidata\\.org/entity/(P\\d+)$");

	public static void main(String[] args) throws Exception {
		Options options = new Options();
		options.addRequiredOption("d", "dir", true, "Directory containing ElementsExtractor TSV outputs");

		CommandLineParser parser = new DefaultParser();
		CommandLine cmd;
		try {
			cmd = parser.parse(options, args);
		} catch (ParseException e) {
			System.err.println("Error: " + e.getMessage());
			new HelpFormatter().printHelp("WikidataLinker", options);
			return;
		}

		Path dir = Paths.get(cmd.getOptionValue("d"));
		if (!Files.isDirectory(dir)) {
			System.err.println("Not a directory: " + dir);
			return;
		}

		Path entitiesLabels = dir.resolve("entities_labels.tsv");
		Path entitiesFull = dir.resolve("entities_full.tsv");
		Path predicatesLabels = dir.resolve("predicates_labels.tsv");
		Path predicatesFull = dir.resolve("predicates_full.tsv");

		if (!Files.exists(entitiesLabels) || !Files.exists(entitiesFull)) {
			System.err.println("Missing entities TSV files in: " + dir);
			return;
		}
		if (!Files.exists(predicatesLabels)) Files.createFile(predicatesLabels);
		if (!Files.exists(predicatesFull)) Files.createFile(predicatesFull);

		// Load entities (use FULL as source of truth)
		LinkedHashMap<String, Pair> entFull = readFullTSV(entitiesFull);
		// Load predicates (merge URIs from labels.tsv as well to know what formats exist)
		LinkedHashMap<String, Pair> predFull = readFullTSV(predicatesFull);
		LinkedHashMap<String, String> predLblOnly = readLabelsTSV(predicatesLabels);
		for (Map.Entry<String, String> e : predLblOnly.entrySet()) {
			predFull.compute(e.getKey(), (k, v) -> {
				if (v == null) return new Pair(e.getValue(), "");
				if ((v.label == null || v.label.isEmpty()) && e.getValue() != null && !e.getValue().isEmpty()) v.label = e.getValue();
				return v;
			});
		}
		Set<String> existingPredicateUris = new HashSet<>(predFull.keySet());

		// Collect property rows from entities
		List<Map.Entry<String, Pair>> propertyRows = new ArrayList<>();
		for (Map.Entry<String, Pair> e : entFull.entrySet()) {
			if (isWikidataPropertyEntity(e.getKey())) {
				propertyRows.add(e);
			}
		}

		if (propertyRows.isEmpty()) {
			System.out.println("No Wikidata property entity rows found. Nothing to do.");
			// Still rewrite labels from entFull to ensure consistency
			writeLabelsTSV(entitiesLabels, entFull);
			writeLabelsTSV(predicatesLabels, predFull);
			writeFullTSV(predicatesFull, predFull);
			return;
		}

		// Move: remove from entities only if a matching predicate URI exists; add/update in predicates accordingly
		for (Map.Entry<String, Pair> e : propertyRows) {
			String entityUri = e.getKey();
			Pair data = e.getValue();
			String pid = extractPid(entityUri);
			if (pid == null) continue;

			String direct = "http://www.wikidata.org/prop/direct/" + pid;
			String directNorm = "http://www.wikidata.org/prop/direct-normalized/" + pid;

			boolean moved = false;
			if (existingPredicateUris.contains(direct)) {
				upsert(predFull, direct, data);
				moved = true;
			}
			if (existingPredicateUris.contains(directNorm)) {
				upsert(predFull, directNorm, data);
				moved = true;
			}
			if (moved) {
				entFull.remove(entityUri);
			}
		}

		// Write back: entities (labels derived from entFull), predicates (labels from predFull)
		writeLabelsTSV(entitiesLabels, entFull);
		writeFullTSV(entitiesFull, entFull);

		writeLabelsTSV(predicatesLabels, predFull);
		writeFullTSV(predicatesFull, predFull);

		System.out.println("WikidataLinker completed. Moved " + propertyRows.size() + " property rows from entities to predicates.");
	}

	private static boolean isWikidataPropertyEntity(String uri) {
		if (uri == null) return false;
		return WD_ENTITY_PROP.matcher(uri).matches();
	}

	private static String extractPid(String uri) {
		Matcher m = WD_ENTITY_PROP.matcher(uri);
		return m.matches() ? m.group(1) : null;
	}

	private static void upsert(LinkedHashMap<String, Pair> map, String uri, Pair src) {
		Pair cur = map.get(uri);
		if (cur == null) {
			map.put(uri, new Pair(src.label, src.description));
		} else {
			if ((cur.label == null || cur.label.isEmpty() || cur.label.startsWith("p")) && src.label != null && !src.label.isEmpty()) {
				cur.label = src.label;
			}
			if ((cur.description == null || cur.description.isEmpty() || cur.description.startsWith("p")) && src.description != null && !src.description.isEmpty()) {
				cur.description = src.description;
			}
		}
	}

	private static LinkedHashMap<String, Pair> readFullTSV(Path file) throws IOException {
		LinkedHashMap<String, Pair> map = new LinkedHashMap<>();
		try (BufferedReader br = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
			String line;
			while ((line = br.readLine()) != null) {
				if (line.isEmpty()) continue;
				String[] parts = line.split("\t", -1);
				if (parts.length < 2) continue; // must have at least uri + label
				String uri = parts[0];
				String label = parts.length > 1 ? parts[1] : "";
				String desc = parts.length > 2 ? parts[2] : "";
				map.put(uri, new Pair(label, desc));
			}
		}
		return map;
	}

	private static void writeLabelsTSV(Path file, LinkedHashMap<String, Pair> map) throws IOException {
		try (BufferedWriter bw = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
			for (Map.Entry<String, Pair> e : map.entrySet()) {
				String uri = e.getKey();
				Pair p = e.getValue();
				bw.write(uri);
				bw.write("\t");
				bw.write(sanitizeTSV(p.label));
				bw.newLine();
			}
		}
	}

	private static void writeFullTSV(Path file, LinkedHashMap<String, Pair> map) throws IOException {
		try (BufferedWriter bw = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
			for (Map.Entry<String, Pair> e : map.entrySet()) {
				String uri = e.getKey();
				Pair p = e.getValue();
				bw.write(uri);
				bw.write("\t");
				bw.write(sanitizeTSV(p.label));
				bw.write("\t");
				bw.write(sanitizeTSV(p.description));
				bw.newLine();
			}
		}
	}

	private static LinkedHashMap<String, String> readLabelsTSV(Path file) throws IOException {
		LinkedHashMap<String, String> map = new LinkedHashMap<>();
		if (!Files.exists(file)) return map;
		try (BufferedReader br = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
			String line;
			while ((line = br.readLine()) != null) {
				if (line.isEmpty()) continue;
				String[] parts = line.split("\t", -1);
				if (parts.length < 1) continue;
				String uri = parts[0];
				String label = parts.length > 1 ? parts[1] : "";
				map.put(uri, label);
			}
		}

		return map;
	}

	private static String sanitizeTSV(String s) {
		if (s == null) return "";
		return s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').trim();
	}

	private static class Pair {
		String label;
		String description;
		Pair(String l, String d) { this.label = l == null ? "" : l; this.description = d == null ? "" : d; }
	}
}
