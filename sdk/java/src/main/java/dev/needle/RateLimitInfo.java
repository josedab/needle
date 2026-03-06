package dev.needle;

/**
 * Rate limit information extracted from HTTP response headers.
 */
public class RateLimitInfo {
    private final Integer limit;
    private final Integer remaining;
    private final Integer retryAfter;

    RateLimitInfo(Integer limit, Integer remaining, Integer retryAfter) {
        this.limit = limit;
        this.remaining = remaining;
        this.retryAfter = retryAfter;
    }

    /** Maximum number of requests allowed in the window, or null if not provided. */
    public Integer getLimit() {
        return limit;
    }

    /** Remaining requests in the current window, or null if not provided. */
    public Integer getRemaining() {
        return remaining;
    }

    /** Seconds to wait before retrying, or null if not provided. */
    public Integer getRetryAfter() {
        return retryAfter;
    }

    @Override
    public String toString() {
        return "RateLimitInfo{limit=" + limit + ", remaining=" + remaining +
                ", retryAfter=" + retryAfter + "}";
    }
}
