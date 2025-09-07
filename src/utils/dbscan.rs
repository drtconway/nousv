
/// Returns a Vec<Vec<usize>> where each vector is a cluster.
#[allow(dead_code)]
pub fn dbscan_1d_sorted<T, F: Fn(&T) -> f64>(
    data: &[T],
    min_points: usize,
    max_dist: f64,
    key: F,
) -> Vec<Vec<usize>> {
    let n = data.len();
    let mut i = 0;
    let mut j;
    let mut result = Vec::new();

    while i < n {
        // Expand window to include all points within max_dist of data[i]
        j = i + 1;
        while j < n && key(&data[j]) - key(&data[i]) <= max_dist {
            j += 1;
        }
        let window_size = j - i;
        if window_size >= min_points {
            result.push(Vec::from_iter(i..j));
        }
        i = j;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dbscan_basic_clusters() {
        let data = vec![0.0, 0.1, 0.2, 1.0, 1.1, 2.5];
        let clusters = dbscan_1d_sorted(&data, 2, 0.2, |x| *x);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0], vec![0, 1, 2]);
        assert_eq!(clusters[1], vec![3, 4]);
    }

    #[test]
    fn test_dbscan_no_clusters() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let clusters = dbscan_1d_sorted(&data, 2, 0.1, |x| *x);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_dbscan_single_point_clusters() {
        let data = vec![0.0, 0.5, 1.0];
        let clusters = dbscan_1d_sorted(&data, 1, 0.0, |x| *x);
        assert_eq!(clusters.len(), 3);
        assert_eq!(clusters[0], vec![0]);
        assert_eq!(clusters[1], vec![1]);
        assert_eq!(clusters[2], vec![2]);
    }

    #[test]
    fn test_dbscan_custom_key() {
        #[derive(Debug)]
        struct Point { x: f64, y: f64 }
        let data = vec![Point { x: 0.0, y: 1.0 }, Point { x: 0.1, y: 2.0 }, Point { x: 1.0, y: 3.0 }];
        let _ = data[0].y;
        let clusters = dbscan_1d_sorted(&data, 2, 0.2, |p| p.x);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0], vec![0, 1]);
    }
}
