#![no_main]

use libfuzzer_sys::fuzz_target;
use needle::NLFilterParser;

fuzz_target!(|data: &str| {
    // Fuzz the natural language filter parser
    // This should not panic on any input
    let parser = NLFilterParser::new();
    let _ = parser.parse(data);
});
