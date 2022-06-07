pub trait Loss {
    type OutputType;
    fn get_loss(output: &Self::OutputType, groundtruth: &Self::OutputType) -> f32;
    fn get_gradient(output: &Self::OutputType, groundtruth: &Self::OutputType) -> Self::OutputType;
}

pub struct MSE1D<const SIZE:usize> {}
impl<const SIZE:usize> Loss for MSE1D<SIZE>{
    type OutputType = [f32; SIZE];
    fn get_loss(output: &Self::OutputType, groundtruth: &Self::OutputType) -> f32 {
        let mut loss = 0.;
        for (o, gt) in output.iter().zip(groundtruth.iter()){
            loss += (o-gt).powi(2);
        }

        return loss/(SIZE as f32);
    }

    fn get_gradient(output: &Self::OutputType, groundtruth: &Self::OutputType) -> Self::OutputType {
        let mut gradient = [0.0f32; SIZE];
        for i in 0..SIZE{
            gradient[i] = 2.*(output[i]-groundtruth[i]) / (SIZE as f32);
        }

        return gradient;
    }
}

#[test]
fn test_mse_1d(){
    let output = [0.2, 0.6, 1., -1., 0.];
    let groundtruth = [0.0, 0.8, -1., 1., 0.4];

    let mse = MSE1D::<5>::get_loss(&output, &groundtruth);
    assert!((mse-1.648).abs() < 0.00001);

    let gradient = MSE1D::<5>::get_gradient(&output, &groundtruth);
    let expected = [0.08, -0.08, 0.8, -0.8, -0.16];
    for i in 0..5 {
        assert!((gradient[i]-expected[i]).abs() < 0.000001);
    }
}