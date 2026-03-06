package dev.needle;

/**
 * Exception thrown when a Needle API request fails.
 */
public class NeedleException extends Exception {
    private final int statusCode;
    private final String code;
    private final String help;

    public NeedleException(String message, int statusCode, String code, String help) {
        super(message);
        this.statusCode = statusCode;
        this.code = code;
        this.help = help;
    }

    public NeedleException(String message) {
        super(message);
        this.statusCode = 0;
        this.code = null;
        this.help = null;
    }

    public NeedleException(String message, Throwable cause) {
        super(message, cause);
        this.statusCode = 0;
        this.code = null;
        this.help = null;
    }

    public int getStatusCode() {
        return statusCode;
    }

    public String getCode() {
        return code;
    }

    public String getHelp() {
        return help;
    }

    @Override
    public String toString() {
        if (code != null && !code.isEmpty()) {
            return "needle: " + getMessage() + " (code: " + code + ", status: " + statusCode + ")";
        }
        if (statusCode != 0) {
            return "needle: " + getMessage() + " (status: " + statusCode + ")";
        }
        return "needle: " + getMessage();
    }
}
