package dev.needle;

import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Client for the Needle vector database REST API.
 *
 * <p>Thread-safe. Uses Java 11+ {@link HttpClient} with no external dependencies.
 *
 * <pre>{@code
 * NeedleClient client = new NeedleClient("http://localhost:8080");
 * Collection col = client.createCollection("docs", 384);
 * client.insert("docs", new Vector("doc1", embedding));
 * SearchResponse resp = client.search("docs", new SearchOptions(queryVec, 10));
 * }</pre>
 */
public class NeedleClient {
    private final String baseUrl;
    private final HttpClient httpClient;
    private final String apiKey;
    private final int maxRetries;
    private volatile RateLimitInfo lastRateLimitInfo;

    /**
     * Creates a new client targeting the given Needle server URL.
     *
     * @param baseUrl server base URL (e.g. "http://localhost:8080")
     */
    public NeedleClient(String baseUrl) {
        this(baseUrl, null, 3);
    }

    /**
     * Creates a new client with API key authentication.
     *
     * @param baseUrl server base URL
     * @param apiKey  API key for X-API-Key header, or null
     */
    public NeedleClient(String baseUrl, String apiKey) {
        this(baseUrl, apiKey, 3);
    }

    /**
     * Creates a new client with API key authentication and configurable retry count.
     *
     * @param baseUrl    server base URL
     * @param apiKey     API key for X-API-Key header, or null
     * @param maxRetries maximum retries for transient errors (429, 5xx). 0 to disable.
     */
    public NeedleClient(String baseUrl, String apiKey, int maxRetries) {
        this.baseUrl = baseUrl.replaceAll("/+$", "");
        this.apiKey = apiKey;
        this.maxRetries = maxRetries;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();
    }

    // --- Collection operations ---

    /**
     * Creates a new vector collection.
     */
    public Collection createCollection(String name, int dimensions) throws NeedleException {
        return createCollection(name, dimensions, null);
    }

    /**
     * Creates a new vector collection with a specific distance function.
     */
    public Collection createCollection(String name, int dimensions, String distance) throws NeedleException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("name", name);
        body.put("dimensions", dimensions);
        if (distance != null && !distance.isEmpty()) {
            body.put("distance", distance);
        }
        Map<String, Object> resp = requestObject("POST", "/v1/collections", body);
        return Collection.fromMap(resp);
    }

    /**
     * Retrieves information about a collection.
     */
    public Collection getCollection(String name) throws NeedleException {
        Map<String, Object> resp = requestObject("GET", "/v1/collections/" + encode(name), null);
        return Collection.fromMap(resp);
    }

    /**
     * Returns all collections in the database.
     */
    @SuppressWarnings("unchecked")
    public List<Collection> listCollections() throws NeedleException {
        List<Object> arr = requestArray("GET", "/v1/collections", null);
        List<Collection> result = new ArrayList<>(arr.size());
        for (Object item : arr) {
            if (item instanceof Map) {
                result.add(Collection.fromMap((Map<String, Object>) item));
            }
        }
        return result;
    }

    /**
     * Deletes a collection and all its vectors.
     */
    public void deleteCollection(String name) throws NeedleException {
        requestVoid("DELETE", "/v1/collections/" + encode(name), null);
    }

    // --- Vector operations ---

    /**
     * Inserts a vector into the specified collection.
     */
    public void insert(String collection, Vector vector) throws NeedleException {
        insert(collection, vector, null);
    }

    /**
     * Inserts a vector with an optional TTL (in seconds).
     */
    public void insert(String collection, Vector vector, Integer ttlSeconds) throws NeedleException {
        requestVoid("POST",
                "/v1/collections/" + encode(collection) + "/vectors",
                vector.toInsertMap(ttlSeconds));
    }

    /**
     * Inserts multiple vectors in a single batch request.
     *
     * @param collection collection name
     * @param vectors    list of vectors to insert
     * @return number of successfully inserted vectors
     */
    @SuppressWarnings("unchecked")
    public int batchInsert(String collection, List<Vector> vectors) throws NeedleException {
        List<Object> items = new ArrayList<>(vectors.size());
        for (Vector v : vectors) {
            items.add(v.toInsertMap(null));
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("vectors", items);
        Map<String, Object> resp = requestObject("POST",
                "/v1/collections/" + encode(collection) + "/vectors/batch", body);
        Object inserted = resp.get("inserted");
        if (inserted instanceof Number) {
            return ((Number) inserted).intValue();
        }
        return vectors.size();
    }

    /**
     * Retrieves a vector by ID from the specified collection.
     */
    public Vector getVector(String collection, String id) throws NeedleException {
        Map<String, Object> resp = requestObject("GET",
                "/v1/collections/" + encode(collection) + "/vectors/" + encode(id), null);
        return Vector.fromMap(resp);
    }

    /**
     * Deletes a vector by ID from the specified collection.
     */
    public void deleteVector(String collection, String id) throws NeedleException {
        requestVoid("DELETE",
                "/v1/collections/" + encode(collection) + "/vectors/" + encode(id), null);
    }

    /**
     * Updates metadata for a vector in the specified collection.
     *
     * @param collection collection name
     * @param id         vector ID
     * @param metadata   metadata fields to set
     * @param replace    if true, replaces all metadata; if false, merges with existing
     */
    public void updateMetadata(String collection, String id,
                               Map<String, Object> metadata, boolean replace) throws NeedleException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("metadata", metadata);
        if (replace) {
            body.put("replace", true);
        }
        requestVoid("POST",
                "/v1/collections/" + encode(collection) + "/vectors/" + encode(id) + "/metadata",
                body);
    }

    // --- Search ---

    /**
     * Performs approximate nearest neighbor search in the specified collection.
     */
    public SearchResponse search(String collection, SearchOptions options) throws NeedleException {
        Map<String, Object> resp = requestObject("POST",
                "/v1/collections/" + encode(collection) + "/search",
                options.toMap());
        return SearchResponse.fromMap(resp);
    }

    /**
     * Performs paginated search, automatically following next_cursor until all
     * matching results are retrieved or maxResults is reached.
     *
     * @param collection collection to search
     * @param options    search parameters (cursor will be updated automatically)
     * @param maxResults maximum results to return, or &lt;= 0 for all available
     * @return aggregated search results across all pages
     */
    public List<SearchResult> searchAll(String collection, SearchOptions options, int maxResults)
            throws NeedleException {
        List<SearchResult> all = new ArrayList<>();
        SearchOptions current = options;

        while (true) {
            SearchResponse resp = search(collection, current);
            all.addAll(resp.getResults());

            if (maxResults > 0 && all.size() >= maxResults) {
                return all.subList(0, maxResults);
            }
            if (!resp.isHasMore() || resp.getNextCursor() == null) {
                break;
            }
            current.setSearchAfter(resp.getNextCursor());
        }
        return all;
    }

    // --- Health ---

    /**
     * Checks if the Needle server is reachable and healthy.
     *
     * @return true if the server responds with a successful health check
     */
    public boolean isHealthy() {
        try {
            requestObject("GET", "/health", null);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    // --- Rate limit ---

    /**
     * Returns rate limit info from the most recent response, or null.
     */
    public RateLimitInfo getLastRateLimitInfo() {
        return lastRateLimitInfo;
    }

    // --- Internal HTTP layer ---

    private Map<String, Object> requestObject(String method, String path, Map<String, Object> body)
            throws NeedleException {
        String responseBody = doRequest(method, path, body);
        if (responseBody == null || responseBody.isEmpty()) {
            return new LinkedHashMap<>();
        }
        try {
            return SimpleJson.parseObject(responseBody);
        } catch (Exception e) {
            throw new NeedleException("Failed to parse response: " + e.getMessage(), e);
        }
    }

    @SuppressWarnings("unchecked")
    private List<Object> requestArray(String method, String path, Map<String, Object> body)
            throws NeedleException {
        String responseBody = doRequest(method, path, body);
        if (responseBody == null || responseBody.isEmpty()) {
            return new ArrayList<>();
        }
        try {
            return SimpleJson.parseArray(responseBody);
        } catch (Exception e) {
            throw new NeedleException("Failed to parse response: " + e.getMessage(), e);
        }
    }

    private void requestVoid(String method, String path, Map<String, Object> body)
            throws NeedleException {
        doRequest(method, path, body);
    }

    private String doRequest(String method, String path, Map<String, Object> body)
            throws NeedleException {
        try {
            String url = baseUrl + path;
            String jsonBody = body != null ? SimpleJson.toJson(body) : null;

            NeedleException lastError = null;
            for (int attempt = 0; attempt <= maxRetries; attempt++) {
                if (attempt > 0) {
                    long delay = retryDelayMs(attempt);
                    Thread.sleep(delay);
                }

                HttpRequest.Builder reqBuilder = HttpRequest.newBuilder()
                        .uri(URI.create(url))
                        .timeout(Duration.ofSeconds(30))
                        .header("Content-Type", "application/json");

                if (apiKey != null && !apiKey.isEmpty()) {
                    reqBuilder.header("X-API-Key", apiKey);
                }

                if (jsonBody != null) {
                    reqBuilder.method(method, HttpRequest.BodyPublishers.ofString(jsonBody));
                } else {
                    reqBuilder.method(method, HttpRequest.BodyPublishers.noBody());
                }

                HttpResponse<String> response = httpClient.send(
                        reqBuilder.build(),
                        HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));

                extractRateLimit(response);

                int status = response.statusCode();
                if (status < 200 || status >= 300) {
                    lastError = parseError(status, response.body(), response);
                    if (isRetryable(status) && attempt < maxRetries) {
                        continue;
                    }
                    throw lastError;
                }

                return response.body();
            }

            throw lastError != null ? lastError
                    : new NeedleException("Request failed after " + maxRetries + " retries");

        } catch (NeedleException e) {
            throw e;
        } catch (Exception e) {
            throw new NeedleException("Request failed: " + e.getMessage(), e);
        }
    }

    private static boolean isRetryable(int status) {
        return status == 429 || status >= 500;
    }

    private long retryDelayMs(int attempt) {
        if (lastRateLimitInfo != null && lastRateLimitInfo.getRetryAfter() != null
                && lastRateLimitInfo.getRetryAfter() > 0) {
            return lastRateLimitInfo.getRetryAfter() * 1000L;
        }
        long base = Math.min(1000L * (1L << (attempt - 1)), 30000L);
        long jitter = (long) (base * 0.5 * Math.random());
        return base + jitter;
    }

    private void extractRateLimit(HttpResponse<?> response) {
        Integer limit = parseIntHeader(response, "x-ratelimit-limit");
        Integer remaining = parseIntHeader(response, "x-ratelimit-remaining");
        Integer retryAfter = parseIntHeader(response, "retry-after");

        if (limit != null || remaining != null || retryAfter != null) {
            lastRateLimitInfo = new RateLimitInfo(limit, remaining, retryAfter);
        }
    }

    private static Integer parseIntHeader(HttpResponse<?> response, String name) {
        return response.headers().firstValue(name).map(v -> {
            try {
                return Integer.parseInt(v);
            } catch (NumberFormatException e) {
                return null;
            }
        }).orElse(null);
    }

    private static NeedleException parseError(int statusCode, String body, HttpResponse<?> response) {
        Integer retryAfter = response != null ? parseIntHeader(response, "retry-after") : null;
        if (body == null || body.isEmpty()) {
            return new NeedleException("HTTP " + statusCode, statusCode, null, null, retryAfter);
        }
        try {
            Map<String, Object> map = SimpleJson.parseObject(body);
            String message = map.containsKey("error") ? String.valueOf(map.get("error")) : "HTTP " + statusCode;
            String code = map.containsKey("code") ? String.valueOf(map.get("code")) : null;
            String help = map.containsKey("help") ? String.valueOf(map.get("help")) : null;
            return new NeedleException(message, statusCode, code, help, retryAfter);
        } catch (Exception e) {
            return new NeedleException(body, statusCode, null, null, retryAfter);
        }
    }

    private static String encode(String value) {
        return URLEncoder.encode(value, StandardCharsets.UTF_8);
    }
}
