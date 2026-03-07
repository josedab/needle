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

// SearchAll performs paginated search, automatically following next_cursor until
// all matching results are retrieved or maxResults is reached.
// If maxResults <= 0, it collects all available results.
func (c *Client) SearchAll(ctx context.Context, collection string, opts SearchOptions, maxResults int) ([]SearchResult, error) {
	if opts.K == 0 {
		opts.K = 10
	}

	var all []SearchResult
	current := opts
	for {
		resp, err := c.Search(ctx, collection, current)
		if err != nil {
			return all, err
		}
		all = append(all, resp.Results...)

		if maxResults > 0 && len(all) >= maxResults {
			all = all[:maxResults]
			break
		}
		if !resp.HasMore || resp.NextCursor == nil {
			break
		}
		current.SearchAfter = resp.NextCursor
	}
	return all, nil
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
