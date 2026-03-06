package needle

import (
	"context"
	"net/url"
)

type createCollectionRequest struct {
	Name       string `json:"name"`
	Dimensions int    `json:"dimensions"`
	Distance   string `json:"distance,omitempty"`
}

// CreateCollection creates a new vector collection.
func (c *Client) CreateCollection(ctx context.Context, name string, dimensions int) (*Collection, error) {
	var col Collection
	err := c.request(ctx, "POST", "/v1/collections", createCollectionRequest{
		Name:       name,
		Dimensions: dimensions,
	}, &col)
	if err != nil {
		return nil, err
	}
	return &col, nil
}

// CreateCollectionWithDistance creates a new vector collection with a specific distance function.
func (c *Client) CreateCollectionWithDistance(ctx context.Context, name string, dimensions int, distance string) (*Collection, error) {
	var col Collection
	err := c.request(ctx, "POST", "/v1/collections", createCollectionRequest{
		Name:       name,
		Dimensions: dimensions,
		Distance:   distance,
	}, &col)
	if err != nil {
		return nil, err
	}
	return &col, nil
}

// GetCollection retrieves information about a collection.
func (c *Client) GetCollection(ctx context.Context, name string) (*Collection, error) {
	var col Collection
	err := c.request(ctx, "GET", "/v1/collections/"+url.PathEscape(name), nil, &col)
	if err != nil {
		return nil, err
	}
	return &col, nil
}

// ListCollections returns all collections in the database.
func (c *Client) ListCollections(ctx context.Context) ([]Collection, error) {
	var cols []Collection
	err := c.request(ctx, "GET", "/v1/collections", nil, &cols)
	if err != nil {
		return nil, err
	}
	return cols, nil
}

// DeleteCollection removes a collection and all its vectors.
func (c *Client) DeleteCollection(ctx context.Context, name string) error {
	return c.request(ctx, "DELETE", "/v1/collections/"+url.PathEscape(name), nil, nil)
}
