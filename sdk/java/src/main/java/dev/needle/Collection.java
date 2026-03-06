package dev.needle;

import java.util.Map;

/**
 * Represents a Needle vector collection.
 */
public class Collection {
    private String name;
    private int dimensions;
    private String distance;
    private int count;
    private int deletedCount;

    public Collection() {
    }

    public Collection(String name, int dimensions) {
        this.name = name;
        this.dimensions = dimensions;
    }

    public String getName() {
        return name;
    }

    public int getDimensions() {
        return dimensions;
    }

    public String getDistance() {
        return distance;
    }

    public int getCount() {
        return count;
    }

    public int getDeletedCount() {
        return deletedCount;
    }

    @SuppressWarnings("unchecked")
    static Collection fromMap(Map<String, Object> map) {
        Collection c = new Collection();
        if (map.containsKey("name")) c.name = (String) map.get("name");
        if (map.containsKey("dimensions")) c.dimensions = toInt(map.get("dimensions"));
        if (map.containsKey("distance")) c.distance = (String) map.get("distance");
        if (map.containsKey("count")) c.count = toInt(map.get("count"));
        if (map.containsKey("deleted_count")) c.deletedCount = toInt(map.get("deleted_count"));
        return c;
    }

    private static int toInt(Object val) {
        if (val instanceof Number) return ((Number) val).intValue();
        return 0;
    }

    @Override
    public String toString() {
        return "Collection{name='" + name + "', dimensions=" + dimensions +
                ", distance='" + distance + "', count=" + count + "}";
    }
}
