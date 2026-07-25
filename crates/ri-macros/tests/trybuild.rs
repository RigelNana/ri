//! Compile-time coverage for generated public-trait adapters.

#[test]
fn generated_wrappers_compile_against_public_traits() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/ui/pass/*.rs");
}
