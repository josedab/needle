package dev.needle;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Configures a search request.
 */
public class SearchOptions {
    private float[] vector;
    private int k = 10;
    private Map<String, Object> filter;
    private Map<String, Object> postFilter;
    private boolean includeVectors;
    private boolean explain;
    private String distance;
    private SearchCursor searchAfter;

    public SearchOptions(float[] vector) {
        this.vector = vector;
    }

    public SearchOptions(float[] vector, int k) {
        this.vector = vector;
        this.k = k;
    }

    public float[] getVector() {
        return vector;
    }

    public SearchOptions setVector(float[] vector) {
        this.vector = vector;
        return this;
    }

    public int getK() {
        return k;
    }

    public SearchOptions setK(int k) {
        this.k = k;
        return this;
    }

    public Map<String, Object> getFilter() {
        return filter;
    }

    public SearchOptions setFilter(Map<String, Object> filter) {
        this.filter = filter;
        return this;
    }

    public Map<String, Object> getPostFilter() {
        return postFilter;
    }

    public SearchOptions setPostFilter(Map<String, Object> postFilter) {
        this.postFilter = postFilter;
        return this;
    }

    public boolean isIncludeVectors() {
        return includeVectors;
    }

    public SearchOptions setIncludeVectors(boolean includeVectors) {
        this.includeVectors = includeVectors;
        return this;
    }

    public boolean isExplain() {
        return explain;
    }

    public SearchOptions setExplain(boolean explain) {
        this.explain = explain;
        return this;
    }

    public String getDistance() {
        return distance;
    }

    public SearchOptions setDistance(String distance) {
        this.distance = distance;
        return this;
    }

    public SearchCursor getSearchAfter() {
        return searchAfter;
    }

    public SearchOptions setSearchAfter(SearchCursor searchAfter) {
        this.searchAfter = searchAfter;
        return this;
    }

    Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("vector", vector);
        map.put("k", k > 0 ? k : 10);
        if (filter != null && !filter.isEmpty()) map.put("filter", filter);
        if (postFilter != null && !postFilter.isEmpty()) map.put("post_filter", postFilter);
        if (includeVectors) map.put("include_vectors", true);
        if (explain) map.put("explain", true);
        if (distance != null && !distance.isEmpty()) map.put("distance", distance);
        if (searchAfter != null) map.put("search_after", searchAfter.toMap());
        return map;
    }
}
