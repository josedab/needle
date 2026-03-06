package dev.needle;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents a stored vector with optional metadata.
 */
public class Vector {
    private String id;
    private float[] values;
    private Map<String, Object> metadata;

    public Vector() {
    }

    public Vector(String id, float[] values) {
        this.id = id;
        this.values = values;
    }

    public Vector(String id, float[] values, Map<String, Object> metadata) {
        this.id = id;
        this.values = values;
        this.metadata = metadata;
    }

    public String getId() {
        return id;
    }

    public float[] getValues() {
        return values;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }

    Map<String, Object> toInsertMap() {
        return toInsertMap(null);
    }

    Map<String, Object> toInsertMap(Integer ttlSeconds) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("id", id);
        map.put("vector", values);
        if (metadata != null && !metadata.isEmpty()) {
            map.put("metadata", metadata);
        }
        if (ttlSeconds != null) {
            map.put("ttl_seconds", ttlSeconds);
        }
        return map;
    }

    @SuppressWarnings("unchecked")
    static Vector fromMap(Map<String, Object> map) {
        Vector v = new Vector();
        if (map.containsKey("id")) v.id = (String) map.get("id");
        if (map.containsKey("vector")) v.values = toFloatArray(map.get("vector"));
        if (map.containsKey("metadata") && map.get("metadata") instanceof Map) {
            v.metadata = (Map<String, Object>) map.get("metadata");
        }
        return v;
    }

    static float[] toFloatArray(Object obj) {
        if (obj instanceof List) {
            List<?> list = (List<?>) obj;
            float[] arr = new float[list.size()];
            for (int i = 0; i < list.size(); i++) {
                arr[i] = ((Number) list.get(i)).floatValue();
            }
            return arr;
        }
        if (obj instanceof float[]) return (float[]) obj;
        return new float[0];
    }

    @Override
    public String toString() {
        return "Vector{id='" + id + "', dimensions=" + (values != null ? values.length : 0) + "}";
    }
}
