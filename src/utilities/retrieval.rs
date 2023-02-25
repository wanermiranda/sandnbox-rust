/// It takes two vectors of floats, and returns the cosine similarity between them
///
/// Arguments:
///
/// * `v1`: The first vector
/// * `v2`: The vector to compare against
///
/// Returns:
///
/// A vector of vectors of floats.
/// ```
/// use sandbox_rust::utilities::retrieval::cosine_similarity;
/// let v1 = vec![0.0, 0.0, 0.0];
/// let v2 = vec![1.0, 1.0, 1.0];
/// assert!(cosine_similarity(&v2, &v2) == 1.0);
/// assert_eq!(cosine_similarity(&v1, &v2), 0.0);
/// ```
#[must_use]
pub fn cosine_similarity(v1: &Vec<f32>, v2: &Vec<f32>) -> f64 {
    let mut dot = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for x in 0..v1.len() {
        if v1[x] > 0.0 && v2[x] > 0.0 {
            dot += v1[x] * v2[x];
        }
        if v1[x] > 0.0 {
            norm1 += v1[x].powi(2);
        }

        if v2[x] > 0.0 {
            norm2 += v2[x].powi(2);
        }
    }

    if norm1 * norm2 == 0.0 {
        0.0
    } else {
        (dot / (norm1.sqrt() * norm2.sqrt())).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cosine_similarity_domain() {
        let v1 = vec![0.0, 0.0, 0.0];
        let v2 = vec![1.0, 1.0, 1.0];
        let v3 = vec![2.0, 2.0, 2.0];

        let v4 = vec![1.0, 2.0, 3.0];

        // Dissimilarity with a Zero Vector
        assert!(cosine_similarity(&v1, &v2) == 0.0);

        // Equal vectors
        assert!(cosine_similarity(&v2, &v2) == 1.0);

        // Equal vectors v3 = v2 * 2
        assert!(cosine_similarity(&v3, &v2) == 1.0);

        // Dissimilar vectors
        let t1 = cosine_similarity(&v4, &v2);
        let t2 = cosine_similarity(&v4, &v3);
        assert!(t1 < 1.0 && t1 > 0.0);
        assert!(t2 < 1.0 && t2 > 0.0);

        assert_eq!(t1, t2);
    }
}
