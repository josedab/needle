#![no_main]

use libfuzzer_sys::fuzz_target;
use needle::Filter;

fuzz_target!(|data: &[u8]| {
    // Fuzz the metadata filter parser with JSON input
    if let Ok(json_str) = std::str::from_utf8(data) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
            // This should not panic on any valid JSON input
            let _ = Filter::parse(&value);
        }
    }
});
