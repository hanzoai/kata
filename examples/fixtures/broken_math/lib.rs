//! Deliberately broken single-file unit test fixture for `kata run`.
//!
//! Contains a simple arithmetic regression so `kata run` can demonstrate the
//! closed Detect -> Triage -> Synthesize -> Verify loop deterministically.

pub fn add(a: i32, b: i32) -> i32 {
    // Deliberate defect fixed by the closed-loop repair agent: should be `a + b`.
    a - b
}

pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

#[cfg(test)]
mod tests {
    use super::{add, multiply};

    #[test]
    fn adds_positive_integers() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 4), 3);
    }

    #[test]
    fn multiplies_integers() {
        assert_eq!(multiply(3, 4), 12);
    }
}
