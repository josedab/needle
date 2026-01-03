#![no_main]

use libfuzzer_sys::fuzz_target;
use needle::QueryParser;

fuzz_target!(|data: &str| {
    // Fuzz the NeedleQL query parser
    // This should not panic on any input
    let _ = QueryParser::parse(data);
});
