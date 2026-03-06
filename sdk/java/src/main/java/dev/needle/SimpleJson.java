package dev.needle;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal JSON parser and serializer for the Needle SDK.
 * Handles objects, arrays, strings, numbers, booleans, and null.
 * No external dependencies required.
 */
final class SimpleJson {

    private SimpleJson() {
    }

    // --- Serialization ---

    static String toJson(Object obj) {
        if (obj == null) return "null";
        if (obj instanceof String) return escapeString((String) obj);
        if (obj instanceof Boolean) return obj.toString();
        if (obj instanceof Integer || obj instanceof Long) return obj.toString();
        if (obj instanceof Float) {
            float f = (Float) obj;
            if (f == (long) f) return Long.toString((long) f);
            return Float.toString(f);
        }
        if (obj instanceof Double) {
            double d = (Double) obj;
            if (d == (long) d) return Long.toString((long) d);
            return Double.toString(d);
        }
        if (obj instanceof Number) return obj.toString();
        if (obj instanceof float[]) return floatArrayToJson((float[]) obj);
        if (obj instanceof Map) return mapToJson((Map<?, ?>) obj);
        if (obj instanceof List) return listToJson((List<?>) obj);
        return escapeString(obj.toString());
    }

    private static String escapeString(String s) {
        StringBuilder sb = new StringBuilder(s.length() + 2);
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"':  sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\b': sb.append("\\b"); break;
                case '\f': sb.append("\\f"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        sb.append('"');
        return sb.toString();
    }

    private static String floatArrayToJson(float[] arr) {
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(arr[i]);
        }
        sb.append(']');
        return sb.toString();
    }

    private static String mapToJson(Map<?, ?> map) {
        StringBuilder sb = new StringBuilder();
        sb.append('{');
        boolean first = true;
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            sb.append(escapeString(entry.getKey().toString()));
            sb.append(':');
            sb.append(toJson(entry.getValue()));
        }
        sb.append('}');
        return sb.toString();
    }

    private static String listToJson(List<?> list) {
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (int i = 0; i < list.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append(toJson(list.get(i)));
        }
        sb.append(']');
        return sb.toString();
    }

    // --- Parsing ---

    @SuppressWarnings("unchecked")
    static Map<String, Object> parseObject(String json) {
        Object result = new Parser(json.trim()).parseValue();
        if (result instanceof Map) return (Map<String, Object>) result;
        throw new IllegalArgumentException("Expected JSON object, got: " + (result == null ? "null" : result.getClass().getSimpleName()));
    }

    @SuppressWarnings("unchecked")
    static List<Object> parseArray(String json) {
        Object result = new Parser(json.trim()).parseValue();
        if (result instanceof List) return (List<Object>) result;
        throw new IllegalArgumentException("Expected JSON array, got: " + (result == null ? "null" : result.getClass().getSimpleName()));
    }

    static Object parse(String json) {
        return new Parser(json.trim()).parseValue();
    }

    /**
     * Recursive descent JSON parser.
     */
    private static final class Parser {
        private final String input;
        private int pos;

        Parser(String input) {
            this.input = input;
            this.pos = 0;
        }

        Object parseValue() {
            skipWhitespace();
            if (pos >= input.length()) throw error("Unexpected end of input");
            char c = input.charAt(pos);
            switch (c) {
                case '{': return parseObjectValue();
                case '[': return parseArrayValue();
                case '"': return parseString();
                case 't': case 'f': return parseBoolean();
                case 'n': return parseNull();
                default:
                    if (c == '-' || (c >= '0' && c <= '9')) return parseNumber();
                    throw error("Unexpected character: " + c);
            }
        }

        private Map<String, Object> parseObjectValue() {
            expect('{');
            Map<String, Object> map = new LinkedHashMap<>();
            skipWhitespace();
            if (pos < input.length() && input.charAt(pos) == '}') {
                pos++;
                return map;
            }
            while (true) {
                skipWhitespace();
                String key = parseString();
                skipWhitespace();
                expect(':');
                Object value = parseValue();
                map.put(key, value);
                skipWhitespace();
                if (pos >= input.length()) throw error("Unexpected end of object");
                if (input.charAt(pos) == '}') { pos++; return map; }
                expect(',');
            }
        }

        private List<Object> parseArrayValue() {
            expect('[');
            List<Object> list = new ArrayList<>();
            skipWhitespace();
            if (pos < input.length() && input.charAt(pos) == ']') {
                pos++;
                return list;
            }
            while (true) {
                list.add(parseValue());
                skipWhitespace();
                if (pos >= input.length()) throw error("Unexpected end of array");
                if (input.charAt(pos) == ']') { pos++; return list; }
                expect(',');
            }
        }

        private String parseString() {
            expect('"');
            StringBuilder sb = new StringBuilder();
            while (pos < input.length()) {
                char c = input.charAt(pos++);
                if (c == '"') return sb.toString();
                if (c == '\\') {
                    if (pos >= input.length()) throw error("Unexpected end of string escape");
                    char esc = input.charAt(pos++);
                    switch (esc) {
                        case '"':  sb.append('"'); break;
                        case '\\': sb.append('\\'); break;
                        case '/':  sb.append('/'); break;
                        case 'b':  sb.append('\b'); break;
                        case 'f':  sb.append('\f'); break;
                        case 'n':  sb.append('\n'); break;
                        case 'r':  sb.append('\r'); break;
                        case 't':  sb.append('\t'); break;
                        case 'u':
                            if (pos + 4 > input.length()) throw error("Incomplete unicode escape");
                            String hex = input.substring(pos, pos + 4);
                            sb.append((char) Integer.parseInt(hex, 16));
                            pos += 4;
                            break;
                        default:
                            throw error("Unknown escape: \\" + esc);
                    }
                } else {
                    sb.append(c);
                }
            }
            throw error("Unterminated string");
        }

        private Number parseNumber() {
            int start = pos;
            if (pos < input.length() && input.charAt(pos) == '-') pos++;
            while (pos < input.length() && input.charAt(pos) >= '0' && input.charAt(pos) <= '9') pos++;
            boolean isFloat = false;
            if (pos < input.length() && input.charAt(pos) == '.') {
                isFloat = true;
                pos++;
                while (pos < input.length() && input.charAt(pos) >= '0' && input.charAt(pos) <= '9') pos++;
            }
            if (pos < input.length() && (input.charAt(pos) == 'e' || input.charAt(pos) == 'E')) {
                isFloat = true;
                pos++;
                if (pos < input.length() && (input.charAt(pos) == '+' || input.charAt(pos) == '-')) pos++;
                while (pos < input.length() && input.charAt(pos) >= '0' && input.charAt(pos) <= '9') pos++;
            }
            String numStr = input.substring(start, pos);
            if (isFloat) return Double.parseDouble(numStr);
            long val = Long.parseLong(numStr);
            if (val >= Integer.MIN_VALUE && val <= Integer.MAX_VALUE) return (int) val;
            return val;
        }

        private Boolean parseBoolean() {
            if (input.startsWith("true", pos)) { pos += 4; return Boolean.TRUE; }
            if (input.startsWith("false", pos)) { pos += 5; return Boolean.FALSE; }
            throw error("Expected boolean");
        }

        private Object parseNull() {
            if (input.startsWith("null", pos)) { pos += 4; return null; }
            throw error("Expected null");
        }

        private void expect(char c) {
            skipWhitespace();
            if (pos >= input.length() || input.charAt(pos) != c) {
                throw error("Expected '" + c + "'");
            }
            pos++;
        }

        private void skipWhitespace() {
            while (pos < input.length()) {
                char c = input.charAt(pos);
                if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
                pos++;
            }
        }

        private IllegalArgumentException error(String msg) {
            return new IllegalArgumentException("JSON parse error at position " + pos + ": " + msg);
        }
    }
}
