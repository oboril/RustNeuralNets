use super::Layer;

pub struct Softmax<const SIZE: usize> {
    output : [f32; SIZE],
    deltas : [f32; SIZE]
}

impl<const SIZE: usize> Softmax<SIZE> {
    pub fn new() -> Softmax<SIZE> {
        return Softmax {output: [0.;SIZE], deltas: [0.;SIZE]};
    }
}

impl<const SIZE: usize> Layer for Softmax<SIZE> {
    type InputType = [f32; SIZE];
    type OutputType = [f32; SIZE];

    fn new() -> Self {
        return Softmax::new();
    }

    fn feedforward(&mut self, prev_output: &[f32;SIZE]) {
        let max = prev_output.iter().cloned().fold(0./0., f32::max);

        for i in 0..SIZE {
            self.output[i] = (prev_output[i]-max).exp();
        }

        let sum:f32 = self.output.iter().sum();
        for i in 0..SIZE {
            self.output[i] /= sum;
        }
    }

    fn get_output(&self) -> &[f32;SIZE] {
        return &self.output;
    }

    fn get_deltas(&self) -> &[f32;SIZE] {
        return &self.deltas;
    }

    fn backpropagate(&mut self, next_deltas : &[f32;SIZE]) {
        for i in 0..SIZE {
            self.deltas[i] = next_deltas[i] * self.output[i] * (1. - self.output[i])
        }
    }

    fn update_gradient(&mut self, prev_output: &[f32;SIZE], next_deltas : &[f32;SIZE], batch_size: f32, momentum: f32) {}
    fn update_weights(&mut self, learning_rate: f32, momentum: f32, l2: f32) {}
}

#[test]
fn test_softmax(){
    let mut softmax = Softmax::<3>::new();

    fn assert_equal(num1:f32, num2:f32){
        assert!((num1 - num2).abs() < 0.001, "The numbers are not close enough");
    }

    softmax.feedforward(&[1.0, -1.0, 0.2]);
    let out = softmax.get_output();

    println!("Chk1");
    assert_equal(out[0], 0.6310);
    assert_equal(out[1], 0.0854);
    assert_equal(out[2], 0.2835);
    println!("Chk2");
    assert_equal(out.iter().sum(), 1.);

    softmax.backpropagate(&[1., -2., 3.]);
    let deltas = softmax.get_deltas();

    println!("Chk3");
    assert_equal(deltas[0], 0.2328);
    assert_equal(deltas[1], -0.1562);
    assert_equal(deltas[2], 0.6094);

    softmax.feedforward(&[1000., -1000., 0.]);
    let out = softmax.get_output();
    assert_equal(out[0], 1.0);
    assert_equal(out[1], 0.0);
    assert_equal(out[2], 0.0);

}