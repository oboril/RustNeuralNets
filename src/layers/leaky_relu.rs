use super::Layer;

pub struct LeakyRelu1D<const SIZE: usize> {
    output : [f32; SIZE],
    deltas : [f32; SIZE],
    pub leak : f32,
}

impl<const SIZE: usize> LeakyRelu1D<SIZE> {
    pub fn new() -> LeakyRelu1D<SIZE> {
        return LeakyRelu1D {output: [0.;SIZE], deltas: [0.;SIZE], leak: 0.1};
    }
}

impl<const SIZE: usize> Layer for LeakyRelu1D<SIZE> {
    type InputType = [f32; SIZE];
    type OutputType = [f32; SIZE];

    fn new() -> Self {
        return LeakyRelu1D::new();
    }

    fn feedforward(&mut self, prev_output: &[f32;SIZE]) {
        for i in 0..SIZE {
            if prev_output[i] > 0. {
                self.output[i] = prev_output[i];
            }
            else {
                self.output[i] = prev_output[i] * self.leak;
            }
            self.output[i] = prev_output[i].max(0.);
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
            self.deltas[i] = if self.output[i] > 0. {next_deltas[i]} else {next_deltas[i] * self.leak};
        }
    }

    fn update_gradient(&mut self, prev_output: &[f32;SIZE], next_deltas : &[f32;SIZE], batch_size: f32, momentum: f32) {}
    fn update_weights(&mut self, learning_rate: f32, momentum: f32, l2: f32) {}
}