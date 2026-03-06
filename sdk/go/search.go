package needle

import (
	"context"
	"net/url"
)

// Search performs an approximate nearest neighbor search in the specified collection.
func (c *Client) Search(ctx context.Context, collection string, opts SearchOptions) (*SearchResponse, error) {
	if opts.K == 0 {
		opts.K = 10
	}
	var resp SearchResponse
	err := c.request(ctx, "POST", "/v1/collections/"+url.PathEscape(collection)+"/search", opts, &resp)
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

// Health checks if the Needle server is reachable and healthy.
func (c *Client) Health(ctx context.Context) (*HealthResponse, error) {
	var resp HealthResponse
	err := c.request(ctx, "GET", "/health", nil, &resp)
	if err != nil {
		return nil, err
	}
	return &resp, nil
}
