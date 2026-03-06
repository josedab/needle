package dev.needle;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Response from a search request, with pagination support.
 */
public class SearchResponse {
    private List<SearchResult> results;
    private SearchCursor nextCursor;
    private boolean hasMore;
    private Object explanation;

    public List<SearchResult> getResults() {
        return results != null ? results : Collections.emptyList();
    }

    public SearchCursor getNextCursor() {
        return nextCursor;
    }

    public boolean isHasMore() {
        return hasMore;
    }

    public Object getExplanation() {
        return explanation;
    }

    @SuppressWarnings("unchecked")
    static SearchResponse fromMap(Map<String, Object> map) {
        SearchResponse resp = new SearchResponse();

        if (map.containsKey("results") && map.get("results") instanceof List) {
            List<?> list = (List<?>) map.get("results");
            resp.results = new ArrayList<>(list.size());
            for (Object item : list) {
                if (item instanceof Map) {
                    resp.results.add(SearchResult.fromMap((Map<String, Object>) item));
                }
            }
        }

        if (map.containsKey("next_cursor") && map.get("next_cursor") instanceof Map) {
            resp.nextCursor = SearchCursor.fromMap((Map<String, Object>) map.get("next_cursor"));
        }

        if (map.containsKey("has_more")) {
            Object hm = map.get("has_more");
            resp.hasMore = Boolean.TRUE.equals(hm);
        }

        if (map.containsKey("explanation")) {
            resp.explanation = map.get("explanation");
        }

        return resp;
    }

    @Override
    public String toString() {
        return "SearchResponse{results=" + (results != null ? results.size() : 0) +
                ", hasMore=" + hasMore + "}";
    }
}
