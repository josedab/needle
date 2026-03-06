package needle

import (
	"context"
	"net/url"
)

type insertRequest struct {
	ID         string                 `json:"id"`
	Vector     []float32              `json:"vector"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	TTLSeconds *int                   `json:"ttl_seconds,omitempty"`
}

// Insert adds a vector to the specified collection.
func (c *Client) Insert(ctx context.Context, collection string, v *Vector) error {
	return c.request(ctx, "POST", "/v1/collections/"+url.PathEscape(collection)+"/vectors", insertRequest{
		ID:       v.ID,
		Vector:   v.Values,
		Metadata: v.Metadata,
	}, nil)
}

// InsertWithOptions adds a vector to the specified collection with additional options.
func (c *Client) InsertWithOptions(ctx context.Context, collection string, v *Vector, opts *InsertOptions) error {
	req := insertRequest{
		ID:       v.ID,
		Vector:   v.Values,
		Metadata: v.Metadata,
	}
	if opts != nil {
		if opts.Metadata != nil {
			req.Metadata = opts.Metadata
		}
		req.TTLSeconds = opts.TTLSeconds
	}
	return c.request(ctx, "POST", "/v1/collections/"+url.PathEscape(collection)+"/vectors", req, nil)
}

// Get retrieves a vector by ID from the specified collection.
func (c *Client) Get(ctx context.Context, collection, id string) (*Vector, error) {
	var v Vector
	err := c.request(ctx, "GET", "/v1/collections/"+url.PathEscape(collection)+"/vectors/"+url.PathEscape(id), nil, &v)
	if err != nil {
		return nil, err
	}
	return &v, nil
}

// Delete removes a vector by ID from the specified collection.
func (c *Client) Delete(ctx context.Context, collection, id string) error {
	return c.request(ctx, "DELETE", "/v1/collections/"+url.PathEscape(collection)+"/vectors/"+url.PathEscape(id), nil, nil)
}
