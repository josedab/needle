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
	maxRetries    int
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

// WithMaxRetries sets the maximum number of retries for transient errors (429, 5xx).
// Default is 3. Set to 0 to disable retries.
func WithMaxRetries(n int) ClientOption {
	return func(c *Client) {
		c.maxRetries = n
	}
}

// NewClient creates a new Needle API client.
func NewClient(baseURL string, opts ...ClientOption) *Client {
	c := &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{Timeout: 30 * time.Second},
		maxRetries: 3,
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
// Retries transient errors (429, 5xx) up to maxRetries times with exponential backoff.
func (c *Client) request(ctx context.Context, method, path string, body interface{}, dest interface{}) error {
	url := c.baseURL + path

	var bodyBytes []byte
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("needle: failed to marshal request body: %w", err)
		}
		bodyBytes = data
	}

	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			delay := retryDelay(attempt, c.lastRateLimit)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
			}
		}

		var reqBody io.Reader
		if bodyBytes != nil {
			reqBody = bytes.NewReader(bodyBytes)
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

		c.extractRateLimit(resp.Header)

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			lastErr = c.parseError(resp)
			resp.Body.Close()
			if isRetryableStatus(resp.StatusCode) && attempt < c.maxRetries {
				continue
			}
			return lastErr
		}

		defer resp.Body.Close()
		if dest != nil {
			if err := json.NewDecoder(resp.Body).Decode(dest); err != nil {
				return fmt.Errorf("needle: failed to decode response: %w", err)
			}
		} else {
			_, _ = io.Copy(io.Discard, resp.Body)
		}

		return nil
	}

	if lastErr != nil {
		return lastErr
	}
	return fmt.Errorf("needle: request failed after %d retries", c.maxRetries)
}

// isRetryableStatus returns true for HTTP status codes safe to retry.
func isRetryableStatus(status int) bool {
	return status == 429 || status >= 500
}

// retryDelay computes an exponential backoff delay with jitter.
func retryDelay(attempt int, rateLimit *RateLimitInfo) time.Duration {
	if rateLimit != nil && rateLimit.RetryAfter != nil && *rateLimit.RetryAfter > 0 {
		return time.Duration(*rateLimit.RetryAfter) * time.Second
	}
	base := time.Duration(1<<uint(attempt-1)) * time.Second
	if base > 30*time.Second {
		base = 30 * time.Second
	}
	jitter := time.Duration(float64(base) * 0.5 * (float64(time.Now().UnixNano()%1000) / 1000.0))
	return base + jitter
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

	if v := resp.Header.Get("retry-after"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			apiErr.RetryAfter = n
		}
	}

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
