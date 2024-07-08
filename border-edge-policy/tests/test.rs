use border_edge_policy::Mat;
use tch::Tensor;

#[test]
fn test_matmul() {
    let x1 = Tensor::from_slice2(&[&[1.0f32, 2., 3.], &[4., 5., 6.]]);
    let y1 = Tensor::from_slice(&[7.0f32, 8., 9.]);
    let z1 = x1.matmul(&y1);

    let x2: Mat = x1.into();
    let y2: Mat = y1.into();
    let z2 = x2.matmul(&y2);

    let z3 = {
        let mut data = vec![0.0f32; 2];
        z1.f_copy_data(&mut data, 2).unwrap();
        Mat {
            shape: vec![2 as _, 1 as _],
            data,
        }
    };

    assert_eq!(z2, z3)
}
