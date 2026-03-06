package needle

// Collection represents a Needle vector collection.
type Collection struct {
	Name         string `json:"name"`
	Dimensions   int    `json:"dimensions"`
	Distance     string `json:"distance,omitempty"`
	Count        int    `json:"count,omitempty"`
	DeletedCount int    `json:"deleted_count,omitempty"`
}

// Vector represents a stored vector with optional metadata.
type Vector struct {
	ID       string                 `json:"id"`
	Values   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// SearchResult represents a single search match.
type SearchResult struct {
	ID       string                 `json:"id"`
	Distance float32                `json:"distance"`
	Score    float32                `json:"score,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	Vector   []float32              `json:"vector,omitempty"`
}

// SearchOptions configures a search request.
type SearchOptions struct {
	Vector         []float32              `json:"vector"`
	K              int                    `json:"k"`
	Filter         map[string]interface{} `json:"filter,omitempty"`
	PostFilter     map[string]interface{} `json:"post_filter,omitempty"`
	IncludeVectors bool                   `json:"include_vectors,omitempty"`
	Explain        bool                   `json:"explain,omitempty"`
	Distance       string                 `json:"distance,omitempty"`
	SearchAfter    *SearchCursor          `json:"search_after,omitempty"`
}

// SearchCursor is used for cursor-based pagination of search results.
type SearchCursor struct {
	Distance float32 `json:"distance"`
	ID       string  `json:"id"`
}

// SearchResponse is the response from a search request.
type SearchResponse struct {
	Results     []SearchResult `json:"results"`
	NextCursor  *SearchCursor  `json:"next_cursor,omitempty"`
	HasMore     bool           `json:"has_more"`
	Explanation interface{}    `json:"explanation,omitempty"`
}

// InsertOptions configures vector insertion.
type InsertOptions struct {
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	TTLSeconds *int                   `json:"ttl_seconds,omitempty"`
}

// RateLimitInfo holds rate limit data extracted from response headers.
type RateLimitInfo struct {
	Limit      *int
	Remaining  *int
	RetryAfter *int
}

// HealthResponse is the response from the health endpoint.
type HealthResponse struct {
	Status  string `json:"status"`
	Version string `json:"version"`
}
