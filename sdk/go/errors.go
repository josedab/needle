package needle

import "fmt"

// APIError represents an error response from the Needle API.
type APIError struct {
	StatusCode int    `json:"-"`
	Code       string `json:"code"`
	Message    string `json:"error"`
	Help       string `json:"help,omitempty"`
}

func (e *APIError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("needle: %s (code: %s, status: %d)", e.Message, e.Code, e.StatusCode)
	}
	return fmt.Sprintf("needle: %s (status: %d)", e.Message, e.StatusCode)
}
