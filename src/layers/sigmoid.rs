use super::Layer;

pub struct Sigmoid<const SIZE: usize> {
    output : [f32; SIZE],
    deltas : [f32; SIZE]
}

impl<const SIZE: usize> Sigmoid<SIZE> {
    pub fn new() -> Sigmoid<SIZE> {
        return Sigmoid {output: [0.;SIZE], deltas: [0.;SIZE]};
    }
}

impl<const SIZE: usize> Layer for Sigmoid<SIZE> {
    type InputType = [f32; SIZE];
    type OutputType = [f32; SIZE];

    fn new() -> Self {
        return Sigmoid::new();
    }

    fn feedforward(&mut self, prev_output: &[f32;SIZE]) {
        for i in 0..SIZE {
            self.output[i] = 1./(1.+(-prev_output[i]).exp());
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
fn test_sigmoid(){
    let mut sigm = Sigmoid::<3>::new();

    fn assert_equal(num1:f32, num2:f32){
        assert!((num1 - num2).abs() < 0.001, "The numbers are not close enough");
    }

    sigm.feedforward(&[1.0, -1.0, 0.2]);
    let out = sigm.get_output();

    assert_equal(out[0], 0.7311);
    assert_equal(out[1], 0.2689);
    assert_equal(out[2], 0.5498);

    sigm.backpropagate(&[1., 2., 3.]);
    let deltas = sigm.get_deltas();

    assert_equal(deltas[0], 0.1966);
    assert_equal(deltas[1], 0.3932);
    assert_equal(deltas[2], 0.7426);
}