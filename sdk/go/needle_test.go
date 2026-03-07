package needle

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestCreateCollection(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/collections" {
			t.Errorf("expected /v1/collections, got %s", r.URL.Path)
		}
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("failed to decode body: %v", err)
		}
		if body["name"] != "test" {
			t.Errorf("expected name 'test', got %v", body["name"])
		}
		if body["dimensions"] != float64(384) {
			t.Errorf("expected dimensions 384, got %v", body["dimensions"])
		}
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(Collection{Name: "test", Dimensions: 384, Distance: "cosine"})
	}))
	defer server.Close()

	client := NewClient(server.URL)
	col, err := client.CreateCollection(context.Background(), "test", 384)
	if err != nil {
		t.Fatal(err)
	}
	if col.Name != "test" {
		t.Errorf("expected name 'test', got '%s'", col.Name)
	}
	if col.Dimensions != 384 {
		t.Errorf("expected dimensions 384, got %d", col.Dimensions)
	}
	if col.Distance != "cosine" {
		t.Errorf("expected distance 'cosine', got '%s'", col.Distance)
	}
}

func TestListCollections(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			t.Errorf("expected GET, got %s", r.Method)
		}
		if r.URL.Path != "/v1/collections" {
			t.Errorf("expected /v1/collections, got %s", r.URL.Path)
		}
		cols := []Collection{
			{Name: "docs", Dimensions: 384, Count: 100},
			{Name: "images", Dimensions: 512, Count: 50},
		}
		json.NewEncoder(w).Encode(cols)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	cols, err := client.ListCollections(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(cols) != 2 {
		t.Fatalf("expected 2 collections, got %d", len(cols))
	}
	if cols[0].Name != "docs" {
		t.Errorf("expected first collection 'docs', got '%s'", cols[0].Name)
	}
	if cols[1].Count != 50 {
		t.Errorf("expected second collection count 50, got %d", cols[1].Count)
	}
}

func TestGetCollection(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			t.Errorf("expected GET, got %s", r.Method)
		}
		if r.URL.Path != "/v1/collections/my-collection" {
			t.Errorf("expected /v1/collections/my-collection, got %s", r.URL.Path)
		}
		json.NewEncoder(w).Encode(Collection{
			Name: "my-collection", Dimensions: 768, Distance: "euclidean", Count: 42,
		})
	}))
	defer server.Close()

	client := NewClient(server.URL)
	col, err := client.GetCollection(context.Background(), "my-collection")
	if err != nil {
		t.Fatal(err)
	}
	if col.Name != "my-collection" {
		t.Errorf("expected name 'my-collection', got '%s'", col.Name)
	}
	if col.Dimensions != 768 {
		t.Errorf("expected dimensions 768, got %d", col.Dimensions)
	}
	if col.Count != 42 {
		t.Errorf("expected count 42, got %d", col.Count)
	}
}

func TestDeleteCollection(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "DELETE" {
			t.Errorf("expected DELETE, got %s", r.Method)
		}
		if r.URL.Path != "/v1/collections/old-data" {
			t.Errorf("expected /v1/collections/old-data, got %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	err := client.DeleteCollection(context.Background(), "old-data")
	if err != nil {
		t.Fatal(err)
	}
}

func TestInsert(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/collections/docs/vectors" {
			t.Errorf("expected /v1/collections/docs/vectors, got %s", r.URL.Path)
		}
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("failed to decode body: %v", err)
		}
		if body["id"] != "vec1" {
			t.Errorf("expected id 'vec1', got %v", body["id"])
		}
		vec, ok := body["vector"].([]interface{})
		if !ok {
			t.Fatalf("expected vector array, got %T", body["vector"])
		}
		if len(vec) != 3 {
			t.Errorf("expected vector length 3, got %d", len(vec))
		}
		meta, ok := body["metadata"].(map[string]interface{})
		if !ok {
			t.Fatalf("expected metadata map, got %T", body["metadata"])
		}
		if meta["category"] != "test" {
			t.Errorf("expected metadata category 'test', got %v", meta["category"])
		}
		w.WriteHeader(http.StatusCreated)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	err := client.Insert(context.Background(), "docs", &Vector{
		ID:       "vec1",
		Values:   []float32{0.1, 0.2, 0.3},
		Metadata: map[string]interface{}{"category": "test"},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestGetVector(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			t.Errorf("expected GET, got %s", r.Method)
		}
		if r.URL.Path != "/v1/collections/docs/vectors/vec1" {
			t.Errorf("expected /v1/collections/docs/vectors/vec1, got %s", r.URL.Path)
		}
		json.NewEncoder(w).Encode(Vector{
			ID:       "vec1",
			Values:   []float32{0.1, 0.2, 0.3},
			Metadata: map[string]interface{}{"category": "test"},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL)
	v, err := client.Get(context.Background(), "docs", "vec1")
	if err != nil {
		t.Fatal(err)
	}
	if v.ID != "vec1" {
		t.Errorf("expected id 'vec1', got '%s'", v.ID)
	}
	if len(v.Values) != 3 {
		t.Errorf("expected 3 values, got %d", len(v.Values))
	}
	if v.Metadata["category"] != "test" {
		t.Errorf("expected metadata category 'test', got %v", v.Metadata["category"])
	}
}

func TestDeleteVector(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "DELETE" {
			t.Errorf("expected DELETE, got %s", r.Method)
		}
		if r.URL.Path != "/v1/collections/docs/vectors/vec1" {
			t.Errorf("expected /v1/collections/docs/vectors/vec1, got %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	err := client.Delete(context.Background(), "docs", "vec1")
	if err != nil {
		t.Fatal(err)
	}
}

func TestSearch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/collections/docs/search" {
			t.Errorf("expected /v1/collections/docs/search, got %s", r.URL.Path)
		}
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("failed to decode body: %v", err)
		}
		if body["k"] != float64(5) {
			t.Errorf("expected k=5, got %v", body["k"])
		}
		resp := SearchResponse{
			Results: []SearchResult{
				{ID: "vec1", Distance: 0.1, Metadata: map[string]interface{}{"category": "test"}},
				{ID: "vec2", Distance: 0.2},
			},
			HasMore:    true,
			NextCursor: &SearchCursor{Distance: 0.2, ID: "vec2"},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	resp, err := client.Search(context.Background(), "docs", SearchOptions{
		Vector: []float32{0.1, 0.2, 0.3},
		K:      5,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(resp.Results))
	}
	if resp.Results[0].ID != "vec1" {
		t.Errorf("expected first result id 'vec1', got '%s'", resp.Results[0].ID)
	}
	if resp.Results[0].Distance != 0.1 {
		t.Errorf("expected first result distance 0.1, got %f", resp.Results[0].Distance)
	}
	if !resp.HasMore {
		t.Error("expected has_more to be true")
	}
	if resp.NextCursor == nil {
		t.Fatal("expected next_cursor to be non-nil")
	}
	if resp.NextCursor.ID != "vec2" {
		t.Errorf("expected next_cursor id 'vec2', got '%s'", resp.NextCursor.ID)
	}
}

func TestSearchWithCursor(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("failed to decode body: %v", err)
		}
		sa, ok := body["search_after"].(map[string]interface{})
		if !ok {
			t.Fatalf("expected search_after object, got %T", body["search_after"])
		}
		if sa["id"] != "vec2" {
			t.Errorf("expected search_after id 'vec2', got %v", sa["id"])
		}
		if sa["distance"] != float64(0.5) {
			t.Errorf("expected search_after distance 0.5, got %v", sa["distance"])
		}
		resp := SearchResponse{
			Results: []SearchResult{
				{ID: "vec3", Distance: 0.6},
			},
			HasMore: false,
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	resp, err := client.Search(context.Background(), "docs", SearchOptions{
		Vector:      []float32{0.1, 0.2, 0.3},
		K:           5,
		SearchAfter: &SearchCursor{Distance: 0.5, ID: "vec2"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(resp.Results))
	}
	if resp.HasMore {
		t.Error("expected has_more to be false")
	}
}

func TestAPIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{
			"code":  "COLLECTION_NOT_FOUND",
			"error": "collection 'missing' not found",
			"help":  "Create the collection first",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL)
	_, err := client.GetCollection(context.Background(), "missing")
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 404 {
		t.Errorf("expected status 404, got %d", apiErr.StatusCode)
	}
	if apiErr.Code != "COLLECTION_NOT_FOUND" {
		t.Errorf("expected code 'COLLECTION_NOT_FOUND', got '%s'", apiErr.Code)
	}
	if apiErr.Message != "collection 'missing' not found" {
		t.Errorf("expected message 'collection 'missing' not found', got '%s'", apiErr.Message)
	}
	if apiErr.Help != "Create the collection first" {
		t.Errorf("expected help 'Create the collection first', got '%s'", apiErr.Help)
	}
}

func TestRateLimitHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("x-ratelimit-limit", "100")
		w.Header().Set("x-ratelimit-remaining", "0")
		w.Header().Set("retry-after", "30")
		w.WriteHeader(http.StatusTooManyRequests)
		json.NewEncoder(w).Encode(map[string]string{
			"code":  "RATE_LIMITED",
			"error": "rate limit exceeded",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL, WithMaxRetries(0))
	_, err := client.ListCollections(context.Background())
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if apiErr.StatusCode != 429 {
		t.Errorf("expected status 429, got %d", apiErr.StatusCode)
	}

	rl := client.LastRateLimit()
	if rl == nil {
		t.Fatal("expected rate limit info, got nil")
	}
	if rl.Limit == nil || *rl.Limit != 100 {
		t.Errorf("expected limit 100, got %v", rl.Limit)
	}
	if rl.Remaining == nil || *rl.Remaining != 0 {
		t.Errorf("expected remaining 0, got %v", rl.Remaining)
	}
	if rl.RetryAfter == nil || *rl.RetryAfter != 30 {
		t.Errorf("expected retry-after 30, got %v", rl.RetryAfter)
	}
}

func TestHealth(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			t.Errorf("expected GET, got %s", r.Method)
		}
		if r.URL.Path != "/health" {
			t.Errorf("expected /health, got %s", r.URL.Path)
		}
		json.NewEncoder(w).Encode(HealthResponse{
			Status:  "ok",
			Version: "0.1.0",
		})
	}))
	defer server.Close()

	client := NewClient(server.URL)
	resp, err := client.Health(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if resp.Status != "ok" {
		t.Errorf("expected status 'ok', got '%s'", resp.Status)
	}
	if resp.Version != "0.1.0" {
		t.Errorf("expected version '0.1.0', got '%s'", resp.Version)
	}
}

func TestWithAPIKey(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := r.Header.Get("X-API-Key")
		if key != "secret-key-123" {
			t.Errorf("expected X-API-Key 'secret-key-123', got '%s'", key)
		}
		json.NewEncoder(w).Encode(HealthResponse{Status: "ok"})
	}))
	defer server.Close()

	client := NewClient(server.URL, WithAPIKey("secret-key-123"))
	_, err := client.Health(context.Background())
	if err != nil {
		t.Fatal(err)
	}
}

func TestWithTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
		json.NewEncoder(w).Encode(HealthResponse{Status: "ok"})
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	client := NewClient(server.URL)
	_, err := client.Health(ctx)
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
}

func TestRetryOn500(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"error": "temporary failure"})
			return
		}
		json.NewEncoder(w).Encode(HealthResponse{Status: "ok", Version: "0.1.0"})
	}))
	defer server.Close()

	client := NewClient(server.URL, WithMaxRetries(3))
	resp, err := client.Health(context.Background())
	if err != nil {
		t.Fatalf("expected success after retries, got: %v", err)
	}
	if resp.Status != "ok" {
		t.Errorf("expected status 'ok', got '%s'", resp.Status)
	}
	if attempts != 3 {
		t.Errorf("expected 3 attempts, got %d", attempts)
	}
}

func TestNoRetryOn404(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "not found"})
	}))
	defer server.Close()

	client := NewClient(server.URL, WithMaxRetries(3))
	_, err := client.GetCollection(context.Background(), "missing")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if attempts != 1 {
		t.Errorf("expected 1 attempt (no retry on 404), got %d", attempts)
	}
}

func TestSearchAll(t *testing.T) {
	page := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		page++
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)

		if page == 1 {
			json.NewEncoder(w).Encode(SearchResponse{
				Results: []SearchResult{
					{ID: "v1", Distance: 0.1},
					{ID: "v2", Distance: 0.2},
				},
				HasMore:    true,
				NextCursor: &SearchCursor{Distance: 0.2, ID: "v2"},
			})
		} else if page == 2 {
			// Verify cursor was sent
			sa, ok := body["search_after"].(map[string]interface{})
			if !ok {
				t.Fatal("expected search_after in second page request")
			}
			if sa["id"] != "v2" {
				t.Errorf("expected cursor id 'v2', got %v", sa["id"])
			}
			json.NewEncoder(w).Encode(SearchResponse{
				Results: []SearchResult{
					{ID: "v3", Distance: 0.3},
				},
				HasMore: false,
			})
		} else {
			t.Errorf("unexpected page %d", page)
		}
	}))
	defer server.Close()

	client := NewClient(server.URL)
	results, err := client.SearchAll(context.Background(), "docs", SearchOptions{
		Vector: []float32{0.1, 0.2, 0.3},
		K:      2,
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 3 {
		t.Fatalf("expected 3 total results across pages, got %d", len(results))
	}
	if results[0].ID != "v1" || results[2].ID != "v3" {
		t.Errorf("unexpected result ordering: %v", results)
	}
	if page != 2 {
		t.Errorf("expected 2 pages, got %d", page)
	}
}

func TestSearchAllWithMaxResults(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(SearchResponse{
			Results: []SearchResult{
				{ID: "v1", Distance: 0.1},
				{ID: "v2", Distance: 0.2},
				{ID: "v3", Distance: 0.3},
			},
			HasMore:    true,
			NextCursor: &SearchCursor{Distance: 0.3, ID: "v3"},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL)
	results, err := client.SearchAll(context.Background(), "docs", SearchOptions{
		Vector: []float32{0.1},
		K:      3,
	}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected maxResults=2, got %d", len(results))
	}
}

func TestUpdateMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/collections/docs/vectors/vec1/metadata" {
			t.Errorf("expected /v1/collections/docs/vectors/vec1/metadata, got %s", r.URL.Path)
		}
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("failed to decode body: %v", err)
		}
		meta, ok := body["metadata"].(map[string]interface{})
		if !ok {
			t.Fatalf("expected metadata map, got %T", body["metadata"])
		}
		if meta["color"] != "blue" {
			t.Errorf("expected color 'blue', got %v", meta["color"])
		}
		if body["replace"] != true {
			t.Errorf("expected replace=true, got %v", body["replace"])
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	err := client.UpdateMetadata(context.Background(), "docs", "vec1",
		map[string]interface{}{"color": "blue"}, true)
	if err != nil {
		t.Fatal(err)
	}
}

func TestBatchInsert(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/collections/docs/vectors/batch" {
			t.Errorf("expected batch path, got %s", r.URL.Path)
		}
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		vectors, ok := body["vectors"].([]interface{})
		if !ok {
			t.Fatalf("expected vectors array, got %T", body["vectors"])
		}
		if len(vectors) != 2 {
			t.Errorf("expected 2 vectors, got %d", len(vectors))
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"inserted": 2})
	}))
	defer server.Close()

	client := NewClient(server.URL)
	result, err := client.BatchInsert(context.Background(), "docs", []*Vector{
		{ID: "v1", Values: []float32{0.1, 0.2}},
		{ID: "v2", Values: []float32{0.3, 0.4}, Metadata: map[string]interface{}{"tag": "test"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Inserted != 2 {
		t.Errorf("expected 2 inserted, got %d", result.Inserted)
	}
}
