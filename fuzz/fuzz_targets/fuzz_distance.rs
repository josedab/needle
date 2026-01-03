#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use needle::DistanceFunction;

#[derive(Arbitrary, Debug)]
struct DistanceInput {
    a: Vec<f32>,
    b: Vec<f32>,
    function: u8,
}

fuzz_target!(|input: DistanceInput| {
    // Only test with matching dimensions
    if input.a.len() != input.b.len() || input.a.is_empty() {
        return;
    }

    // Limit vector size to prevent OOM
    if input.a.len() > 4096 {
        return;
    }

    // Filter out NaN and Inf values for stable testing
    if input.a.iter().any(|x| !x.is_finite()) || input.b.iter().any(|x| !x.is_finite()) {
        return;
    }

    let distance_fn = match input.function % 4 {
        0 => DistanceFunction::Cosine,
        1 => DistanceFunction::Euclidean,
        2 => DistanceFunction::DotProduct,
        _ => DistanceFunction::Manhattan,
    };

    // This should not panic
    let result = distance_fn.compute(&input.a, &input.b);

    // Result should be a valid number (not NaN for finite inputs)
    // Note: Cosine can return NaN for zero vectors, which is acceptable
    if distance_fn != DistanceFunction::Cosine {
        assert!(
            result.is_finite(),
            "Distance returned non-finite value for finite inputs"
        );
    }
});
