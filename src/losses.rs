pub trait Loss {
    type OutputType;
    fn get_loss(output: &Self::OutputType, groundtruth: &Self::OutputType) -> f32;
    fn get_gradient(output: &Self::OutputType, groundtruth: &Self::OutputType) -> Self::OutputType;
}

pub struct SumSquares1D<const SIZE:usize> {}
impl<const SIZE:usize> Loss for SumSquares1D<SIZE>{
    type OutputType = [f32; SIZE];
    fn get_loss(output: &Self::OutputType, groundtruth: &Self::OutputType) -> f32 {
        let mut loss = 0.;
        for (o, gt) in output.iter().zip(groundtruth.iter()){
            loss += (o-gt).powi(2);
        }

        return loss
    }

    fn get_gradient(output: &Self::OutputType, groundtruth: &Self::OutputType) -> Self::OutputType {
        let mut gradient = [0.0f32; SIZE];
        for i in 0..SIZE{
            gradient[i] = 2.*(output[i]-groundtruth[i]);
        }

        return gradient;
    }
}

pub struct CrossEntropy1D<const SIZE:usize> {}
impl<const SIZE:usize> CrossEntropy1D<SIZE> {
    const EPSILON : f32 = 0.0001; // to prevent ln(0) and division by 0
}
impl<const SIZE:usize> Loss for CrossEntropy1D<SIZE>{
    type OutputType = [f32; SIZE];
    fn get_loss(output: &Self::OutputType, groundtruth: &Self::OutputType) -> f32 {
        let mut loss = 0.;
        for (&o, &gt) in output.iter().zip(groundtruth){
            if gt != 0.{
                loss += - gt * (o+Self::EPSILON).ln();
            }
        }

        return loss
    }

    fn get_gradient(output: &Self::OutputType, groundtruth: &Self::OutputType) -> Self::OutputType {
        let mut gradient = [0.0f32; SIZE];
        for (i, (&o, &gt)) in output.iter().zip(groundtruth).enumerate(){
            gradient[i] = - (gt/(o+Self::EPSILON));
        }

        return gradient;
    }
}

#[test]
fn test_mse_1d(){
    let output = [0.2, 0.6, 1., -1., 0.];
    let groundtruth = [0.0, 0.8, -1., 1., 0.4];

    let mse = SumSquares1D::<5>::get_loss(&output, &groundtruth);
    assert!((mse-1.648*5.).abs() < 0.00001);

    let gradient = SumSquares1D::<5>::get_gradient(&output, &groundtruth);
    let expected = [0.08, -0.08, 0.8, -0.8, -0.16];
    for i in 0..5 {
        assert!((gradient[i]-expected[i]*5.).abs() < 0.000001);
    }
}

#[test]
fn test_crossentropy_1d(){
    let output = [0.05, 0.95, 0.00];
    let groundtruth = [0.,1.,0.];

    let loss = CrossEntropy1D::<3>::get_loss(&output, &groundtruth);
    println!("loss: {}", loss);
    assert!((loss - 0.0513).abs() < 0.005);

    let output = [0.1, 0.8, 0.1];
    let groundtruth = [0.,0.,1.];

    let loss = CrossEntropy1D::<3>::get_loss(&output, &groundtruth);
    assert!((loss - 2.303).abs() < 0.01);

    let gradient = CrossEntropy1D::<3>::get_gradient(&output, &groundtruth);
    let expected = [0., 0., -10.];
    for i in 0..3 {
        assert!((gradient[i]-expected[i]).abs() < 0.01);
    }
}