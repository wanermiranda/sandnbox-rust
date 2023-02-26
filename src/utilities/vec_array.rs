use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr};

/// It takes a 2D array and returns a vector of vectors of floats
///
/// Arguments:
///
/// * `arr`: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>
///
/// Returns:
///
/// A vector of vectors of f32s.
pub fn array2_to_vec(arr: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>) -> Vec<Vec<f32>> {
    let rows = arr
        .to_owned()
        .into_raw_vec()
        .chunks(arr.shape()[1])
        .map(<[f32]>::to_vec)
        .collect::<Vec<_>>();
    rows
}

/// It takes a 3D array and returns a vector of vectors of vectors of floats
///
/// Arguments:
///
/// * `arr`: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>
///
/// Returns:
///
/// A vector of vectors of f32s.
#[must_use]
pub fn array3_to_vec(arr: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>) -> Vec<Vec<Vec<f32>>> {
    let rows = arr
        .to_owned()
        .into_raw_vec()
        .chunks(arr.shape()[1] * arr.shape()[2])
        .map(|chunk1| {
            chunk1
                .chunks(arr.shape()[2])
                .map(<[f32]>::to_vec)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    rows
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    #[test]
    fn test_array2() {
        let arr2: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> =
            array![[1.1, 2.1, 3.], [3.1, 2.1, 1.], [1.1, 2.1, 3.]];
        assert_eq!(
            array2_to_vec(&arr2.into_dyn()),
            vec![vec![1.1, 2.1, 3.], vec![3.1, 2.1, 1.], vec![1.1, 2.1, 3.]]
        )
    }

    #[test]
    fn test_array3() {
        let arr3: ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>> = array![
            [[3.1, 2.1, 1.], [3.1, 2.1, 1.], [3.1, 2.1, 1.]],
            [[1.1, 2.1, 3.], [1.1, 2.1, 3.], [1.1, 2.1, 3.]]
        ];
        assert_eq!(
            array3_to_vec(&arr3.into_dyn()),
            vec![
                vec![vec![3.1, 2.1, 1.], vec![3.1, 2.1, 1.], vec![3.1, 2.1, 1.]],
                vec![vec![1.1, 2.1, 3.], vec![1.1, 2.1, 3.], vec![1.1, 2.1, 3.]]
            ]
        )
    }
}
