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

// BatchInsertResult is the response from a batch insert operation.
type BatchInsertResult struct {
	Inserted int              `json:"inserted"`
	Errors   []map[string]any `json:"errors,omitempty"`
}

// BatchInsert inserts multiple vectors in a single request.
func (c *Client) BatchInsert(ctx context.Context, collection string, vectors []*Vector) (*BatchInsertResult, error) {
	items := make([]insertRequest, len(vectors))
	for i, v := range vectors {
		items[i] = insertRequest{
			ID:       v.ID,
			Vector:   v.Values,
			Metadata: v.Metadata,
		}
	}
	var result BatchInsertResult
	err := c.request(ctx, "POST", "/v1/collections/"+url.PathEscape(collection)+"/vectors/batch",
		map[string]any{"vectors": items}, &result)
	if err != nil {
		return nil, err
	}
	return &result, nil
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

type updateMetadataRequest struct {
	Metadata map[string]interface{} `json:"metadata"`
	Replace  bool                   `json:"replace,omitempty"`
}

// UpdateMetadata updates the metadata for a vector in the specified collection.
// If replace is true, existing metadata is fully replaced; otherwise it is merged.
func (c *Client) UpdateMetadata(ctx context.Context, collection, id string, metadata map[string]interface{}, replace bool) error {
	return c.request(ctx, "POST",
		"/v1/collections/"+url.PathEscape(collection)+"/vectors/"+url.PathEscape(id)+"/metadata",
		updateMetadataRequest{Metadata: metadata, Replace: replace}, nil)
}
