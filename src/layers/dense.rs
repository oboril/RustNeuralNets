use super::*;
use rand_distr::StandardNormal;
use rand::Rng;

pub struct Dense<const INPUT_SIZE:usize, const OUTPUT_SIZE:usize> {
    pub output: [f32; OUTPUT_SIZE],
    pub deltas: [f32; INPUT_SIZE],

    pub weights: [[f32; OUTPUT_SIZE];INPUT_SIZE],
    pub bias: [f32; OUTPUT_SIZE],
    
    pub weights_grad: [[f32; OUTPUT_SIZE];INPUT_SIZE],
    pub bias_grad: [f32;OUTPUT_SIZE]
}

impl<const INPUT_SIZE:usize, const OUTPUT_SIZE:usize> Dense<INPUT_SIZE,OUTPUT_SIZE> {
    /// TODO: add random weights initialization
    pub fn new() -> Dense<INPUT_SIZE, OUTPUT_SIZE> {
        let mut new_dense = Dense {
            output: [0.; OUTPUT_SIZE],
            deltas: [0.; INPUT_SIZE],
            weights: [[0.; OUTPUT_SIZE];INPUT_SIZE],
            bias: [0.; OUTPUT_SIZE],
            weights_grad: [[0.; OUTPUT_SIZE];INPUT_SIZE],
            bias_grad: [0.; OUTPUT_SIZE],
        };

        // Random weight initialization
        let mut rng = rand::thread_rng();
        for i in 0..OUTPUT_SIZE {
            new_dense.bias[i] = rng.sample(StandardNormal);
            for j in 0..INPUT_SIZE {
                new_dense.weights[j][i] = rng.sample(StandardNormal);
            }
        }

        // Orthogonalize the weights
        if INPUT_SIZE >= OUTPUT_SIZE {
            for i in 0..OUTPUT_SIZE {
                for j in 0..i {
                    let mut dot_product:f32 = 0.;
                    for k in 0..INPUT_SIZE {
                        dot_product += new_dense.weights[k][i]*new_dense.weights[k][j];
                    }

                    for k in 0..INPUT_SIZE {
                        new_dense.weights[k][i] -= dot_product*new_dense.weights[k][j];
                    }
                }
                let mut norm = 0.;
                for j in 0..INPUT_SIZE { norm += new_dense.weights[j][i].powi(2); }
                norm = norm.sqrt();
                for j in 0..INPUT_SIZE { new_dense.weights[j][i] /= norm; }
            }
        }
        else {
            for i in 0..INPUT_SIZE {
                for j in 0..i {
                    let mut dot_product:f32 = 0.;
                    for k in 0..OUTPUT_SIZE {
                        dot_product += new_dense.weights[i][k]*new_dense.weights[j][k];
                    }

                    for k in 0..OUTPUT_SIZE {
                        new_dense.weights[i][k] -= dot_product*new_dense.weights[j][k];
                    }
                }
                let mut norm = 0.;
                for j in 0..OUTPUT_SIZE { norm += new_dense.weights[i][j].powi(2); }
                norm = norm.sqrt();
                for j in 0..OUTPUT_SIZE { new_dense.weights[i][j] /= norm; }
            }
        }

        return new_dense;
    }
}

impl<const INPUT_SIZE:usize, const OUTPUT_SIZE:usize> Layer for Dense<INPUT_SIZE,OUTPUT_SIZE> {
    type InputType = [f32; INPUT_SIZE];
    type OutputType = [f32; OUTPUT_SIZE];

    fn new() -> Self {
        return Dense::new();
    }

    fn get_output(&self) -> &[f32; OUTPUT_SIZE] {
        return &self.output;
    }

    fn get_deltas(&self) -> &[f32; INPUT_SIZE] {
        return &self.deltas;
    }

    fn feedforward(&mut self, prev_output: &[f32; INPUT_SIZE]) {
        // x(i) = W(i) * x(i-1)
        // simple matrix multiplication
        for i in 0..OUTPUT_SIZE{
            self.output[i] = 0.;
            for j in 0..INPUT_SIZE {
                self.output[i] += self.weights[j][i] * prev_output[j];
            }
        }

        //bias
        for i in 0..OUTPUT_SIZE {
            self.output[i] += self.bias[i];
        }
    }

    fn backpropagate(&mut self, next_deltas : &[f32; OUTPUT_SIZE]) {
        // d(i) = W.T(i) * d(i+1)
        // simple matrix multiplication
        for i in 0..INPUT_SIZE{
            self.deltas[i] = 0.;
            for j in 0..OUTPUT_SIZE {
                self.deltas[i] += self.weights[i][j] * next_deltas[j];
            }
        }
    }

    fn update_gradient(&mut self, prev_output: &[f32; INPUT_SIZE], next_deltas: &[f32; OUTPUT_SIZE], batch_size: f32, momentum: f32) {

        let factor = (1.-momentum)/batch_size;

        // weights 
        for i in 0..INPUT_SIZE{
            for j in 0..OUTPUT_SIZE {
                self.weights_grad[i][j] += prev_output[i] * next_deltas[j] * factor;
            }
        }

        // biases
        for i in 0..OUTPUT_SIZE {
            self.bias_grad[i] += next_deltas[i] * factor;
        }
    }

    fn update_weights(&mut self, learning_rate: f32, momentum: f32, l2: f32) {
        let l2_compl = 1.-l2;

        //weights
        for i in 0..INPUT_SIZE{
            for j in 0..OUTPUT_SIZE {
                self.weights[i][j] *= l2_compl;
                self.weights[i][j] -= self.weights_grad[i][j] * learning_rate;

                self.weights_grad[i][j] *= momentum;
            }
        }

        //biases
        for i in 0..OUTPUT_SIZE {
            self.bias[i] *= l2_compl;
            self.bias[i] -= self.bias_grad[i] * learning_rate;

            self.bias_grad[i] *= momentum;
        }
    }
}