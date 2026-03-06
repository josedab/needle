package dev.needle;

import java.util.List;
import java.util.Map;

/**
 * Represents a single search match.
 */
public class SearchResult {
    private String id;
    private float distance;
    private float score;
    private Map<String, Object> metadata;
    private float[] vector;

    public String getId() {
        return id;
    }

    public float getDistance() {
        return distance;
    }

    public float getScore() {
        return score;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public float[] getVector() {
        return vector;
    }

    @SuppressWarnings("unchecked")
    static SearchResult fromMap(Map<String, Object> map) {
        SearchResult r = new SearchResult();
        if (map.containsKey("id")) r.id = (String) map.get("id");
        if (map.containsKey("distance")) r.distance = ((Number) map.get("distance")).floatValue();
        if (map.containsKey("score")) r.score = ((Number) map.get("score")).floatValue();
        if (map.containsKey("metadata") && map.get("metadata") instanceof Map) {
            r.metadata = (Map<String, Object>) map.get("metadata");
        }
        if (map.containsKey("vector")) r.vector = Vector.toFloatArray(map.get("vector"));
        return r;
    }

    @Override
    public String toString() {
        return "SearchResult{id='" + id + "', distance=" + distance + ", score=" + score + "}";
    }
}
