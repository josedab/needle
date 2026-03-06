package dev.needle;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Cursor for paginated search results.
 */
public class SearchCursor {
    private final float distance;
    private final String id;

    public SearchCursor(float distance, String id) {
        this.distance = distance;
        this.id = id;
    }

    public float getDistance() {
        return distance;
    }

    public String getId() {
        return id;
    }

    Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("distance", distance);
        map.put("id", id);
        return map;
    }

    static SearchCursor fromMap(Map<String, Object> map) {
        if (map == null) return null;
        float dist = map.containsKey("distance") ? ((Number) map.get("distance")).floatValue() : 0f;
        String id = (String) map.get("id");
        return new SearchCursor(dist, id);
    }

    @Override
    public String toString() {
        return "SearchCursor{distance=" + distance + ", id='" + id + "'}";
    }
}
