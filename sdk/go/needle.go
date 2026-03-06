// Package needle provides a Go client for the Needle vector database REST API.
//
// Example usage:
//
//	client := needle.NewClient("http://localhost:8080")
//	col, _ := client.CreateCollection(ctx, "docs", 384)
//	_ = client.Insert(ctx, "docs", &needle.Vector{ID: "doc1", Values: vec, Metadata: meta})
//	resp, _ := client.Search(ctx, "docs", needle.SearchOptions{Vector: vec, K: 10})
package needle

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Client is the Needle vector database API client.
type Client struct {
	baseURL       string
	httpClient    *http.Client
	apiKey        string
	lastRateLimit *RateLimitInfo
}

// ClientOption configures the Needle client.
type ClientOption func(*Client)

// WithAPIKey sets the API key for authentication.
func WithAPIKey(key string) ClientOption {
	return func(c *Client) {
		c.apiKey = key
	}
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(hc *http.Client) ClientOption {
	return func(c *Client) {
		c.httpClient = hc
	}
}

// WithTimeout sets the request timeout.
func WithTimeout(d time.Duration) ClientOption {
	return func(c *Client) {
		c.httpClient.Timeout = d
	}
}

// NewClient creates a new Needle API client.
func NewClient(baseURL string, opts ...ClientOption) *Client {
	c := &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// LastRateLimit returns rate limit info from the most recent response, or nil.
func (c *Client) LastRateLimit() *RateLimitInfo {
	return c.lastRateLimit
}

// request performs an HTTP request and decodes the JSON response into dest.
// If dest is nil the response body is discarded.
func (c *Client) request(ctx context.Context, method, path string, body interface{}, dest interface{}) error {
	url := c.baseURL + path

	var reqBody io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("needle: failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return fmt.Errorf("needle: failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("X-API-Key", c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("needle: request failed: %w", err)
	}
	defer resp.Body.Close()

	c.extractRateLimit(resp.Header)

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return c.parseError(resp)
	}

	if dest != nil {
		if err := json.NewDecoder(resp.Body).Decode(dest); err != nil {
			return fmt.Errorf("needle: failed to decode response: %w", err)
		}
	} else {
		_, _ = io.Copy(io.Discard, resp.Body)
	}

	return nil
}

func (c *Client) extractRateLimit(h http.Header) {
	info := &RateLimitInfo{}
	hasAny := false

	if v := h.Get("x-ratelimit-limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			info.Limit = &n
			hasAny = true
		}
	}
	if v := h.Get("x-ratelimit-remaining"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			info.Remaining = &n
			hasAny = true
		}
	}
	if v := h.Get("retry-after"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			info.RetryAfter = &n
			hasAny = true
		}
	}

	if hasAny {
		c.lastRateLimit = info
	}
}

func (c *Client) parseError(resp *http.Response) error {
	apiErr := &APIError{StatusCode: resp.StatusCode}

	body, err := io.ReadAll(resp.Body)
	if err != nil || len(body) == 0 {
		apiErr.Message = resp.Status
		return apiErr
	}

	if err := json.Unmarshal(body, apiErr); err != nil {
		apiErr.Message = string(body)
	}
	if apiErr.Message == "" {
		apiErr.Message = resp.Status
	}

	return apiErr
}
