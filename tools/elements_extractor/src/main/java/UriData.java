import java.util.*;

public class UriData {
    private final String uri;
    ArrayList<String> labels = new ArrayList<>();
    private HashSet<String> types = new HashSet<>();
    private HashSet<String> superClasses = new HashSet<>();
    String description = "";
    String fallbackLabel = "";
    String fallbackDescription = "";
    String bestLabel = "";
    String bestDescription = "";
    boolean isInternalNode = false;
    boolean isClass = false;
    boolean isSubClass = false;
    boolean isPredicate = false;
    boolean isRedirect = false;
    boolean satisfiesEntityPrefixes = false;
    boolean satisfiesClassPrefixes = false;
    int outgoingLinks = 0;
    int incomingLinks = 0;

    UriData(String uri) {
        this.uri = ElementsExtractor.shortUri(ElementsExtractor.sanitizeTSV(uri));
    }

    String getUri() {
        return ElementsExtractor.restoreShortUri(uri);
    }

    void addLabel(String label) {
        if (ElementsExtractor.memoryOptimized) {
            if (labels.size() >= 3) {
                return; // limit to first 3 labels to save memory
            }
        }
        labels.add(ElementsExtractor.sanitizeTSV(label));
    }

    void addType(String typeUri) {
        types.add(ElementsExtractor.shortUri(ElementsExtractor.sanitizeTSV(typeUri)));
    }

    HashSet<String> getTypes() {
        var res = new HashSet<String>();
        for (var t : types) {
            res.add(ElementsExtractor.restoreShortUri(t));
        }
        return res;
    }

    void addSuperClass(String classUri) {
        superClasses.add(ElementsExtractor.shortUri(ElementsExtractor.sanitizeTSV(classUri)));
    }

    HashSet<String> getSuperClasses() {
        var res = new HashSet<String>();
        for (var t : superClasses) {
            res.add(ElementsExtractor.restoreShortUri(t));
        }
        return res;
    }

    void finalizeValues() {
        if (labels.isEmpty() && !fallbackLabel.isEmpty()) {
            labels.add(fallbackLabel);
        }
        if (description.isEmpty() && !fallbackDescription.isEmpty()) {
            description = fallbackDescription;
        }
        // Choose preferred single label/description: English if present, otherwise first available.
        bestLabel = labels.isEmpty() ? "" : labels.get(0);
        bestDescription = description == null ? "" : description;
    }

    int popularity() {
        return outgoingLinks + incomingLinks;
    }

    String getType() {
        if (isPredicate) return "predicate";
        if (isClass) return "class";
        return "entity";
    }
}
