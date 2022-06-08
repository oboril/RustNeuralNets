use rand::Rng;

use super::layers::{Dense, Relu1D, Layer, LeakyRelu1D, Softmax};
use super::losses::{Loss, SumSquares1D, CrossEntropy1D};

#[macro_export]
macro_rules! create_nn {
    ($nn_name:ident, [$($layers:ident:$layers_t:ty),+], $loss:ty) => {
        create_nn!(@preprocess $nn_name [$($layers : $layers_t),+] [$($layers : $layers_t),*] [] $loss);
    };

    (@preprocess $nn_name:ident [$($forward:ident : $forward_t:ty),+] [$buffer1:ident : $buffer1_t:ty, $($buffer:ident : $buffer_t:ty),*] [] $loss:ty) => {
        create_nn!(@preprocess $nn_name [$($forward : $forward_t),+] [$($buffer : $buffer_t),*] [$buffer1 : $buffer1_t] $loss);
    };

    (@preprocess $nn_name:ident [$($forward:ident : $forward_t:ty),+] [$buffer1:ident : $buffer1_t:ty, $($buffer:ident : $buffer_t:ty),*] [$($reverse:ident : $reverse_t:ty),+] $loss:ty) => {
        create_nn!(@preprocess $nn_name [$($forward : $forward_t),+] [$($buffer : $buffer_t),*] [$buffer1 : $buffer1_t, $($reverse : $reverse_t),*] $loss);
    };

    (@preprocess $nn_name:ident [$($forward:ident : $forward_t:ty),+] [$buffer1:ident : $buffer1_t:ty] [$($reverse:ident : $reverse_t:ty),*] $loss:ty) => {
        create_nn!(@preprocess $nn_name [$($forward : $forward_t),+] [] [$buffer1 : $buffer1_t, $($reverse : $reverse_t),*] $loss);
    };

    (@preprocess $nn_name:ident [$($forward:ident : $forward_t:ty),+] [] [$($reverse:ident : $reverse_t:ty),*] $loss:ty) => {
        create_nn!(@main $nn_name [$($forward : $forward_t),+] [$($reverse : $reverse_t),*] $loss);
    };

    (@main $nn_name:ident [$head1:ident : $head1_t:ty, $($forward:ident : $forward_t:ty),+] [$tail1:ident : $tail1_t:ty,  $($reverse:ident : $reverse_t:ty),+] $loss:ty) => {
        struct $nn_name {
            $head1 : $head1_t,
            $($forward: $forward_t),*,
            input: <$head1_t as Layer>::InputType,
            output_deltas: <$tail1_t as Layer>::OutputType,
        }

        impl NeuralNetwork for $nn_name {
            type InputType = <$head1_t as Layer>::InputType;
            type OutputType = <$tail1_t as Layer>::OutputType;
            type Loss = $loss;

            fn new() -> Self {
                let new_nn = $nn_name {
                    $head1 : <$head1_t>::new(),
                    $($forward: <$forward_t>::new()),*,
                    input: Self::InputType::default(),
                    output_deltas: Self::OutputType::default()
                };

                return new_nn;
            }

            fn get_output(&self) -> &Self::OutputType {
                return self.$tail1.get_output();
            }

            fn feedforward(&mut self, input: &Self::InputType) -> &Self::OutputType {
                self.input = input.clone();

                self.$head1.feedforward(&self.input);
                create_nn!(@feedforward self [$head1 $($forward)+]);

                return self.get_output();
            }

            fn backpropagate(&mut self, groundtruth : &Self::OutputType) {
                self.output_deltas = Self::Loss::get_gradient(self.get_output(), groundtruth);

                self.$tail1.backpropagate(&self.output_deltas);
                create_nn!(@backpropagate self [$tail1 $($reverse)+]);
            }

            fn update_weights(&mut self, learning_rate:f32, momentum:f32, l2: f32) {
                self.$head1.update_weights(learning_rate, momentum, l2);
                $(
                    self.$forward.update_weights(learning_rate, momentum, l2);
                )+
            }

            fn update_gradient(&mut self, batch_size: f32, momentum: f32) {
                create_nn!(@update_gradient [self batch_size momentum] [$head1 $($forward)+]);
            }   

            fn get_loss(&self, groundtruth : &Self::OutputType) -> f32 {
                return Self::Loss::get_loss(self.get_output(), groundtruth);
            }
        }
    };
    // FEEDFORWARD
    (@feedforward $self:ident [$tail:ident $($reversed:ident)+] ) => {
        create_nn!(@feedforward $self [$tail] [$($reversed)+] []);
    };
    (@feedforward $self:ident [$tail:ident] [$buff:ident $($reversed:ident)+] [$($head:ident)*] ) => {
        create_nn!(@feedforward $self [$tail] [$($reversed)+] [$($head)* $buff]);
    };
    (@feedforward $self:ident [$tail:ident] [$head:ident] [$($reversed:ident)*] ) => {
        create_nn!(@feedforward_final $self [$tail $($reversed)*] [$($reversed)* $head]);
    };
    (@feedforward_final $self:ident [$($offset0:ident)+] [$($offset1:ident)+] ) => {
        $(
            $self.$offset1.feedforward($self.$offset0.get_output());
        )+
    };

    // BACKPROPAGATE
    (@backpropagate $self:ident [$head:ident $($body:ident)+] ) => {
        create_nn!(@backpropagate $self [$head] [$($body)+] []);
    };
    (@backpropagate $self:ident [$head:ident] [$buff:ident $($body:ident)+] [$($tail:ident)*] ) => {
        create_nn!(@backpropagate $self [$head] [$($body)+] [$($tail)* $buff]);
    };
    (@backpropagate $self:ident [$head:ident] [$tail:ident] [$($body:ident)*] ) => {
        create_nn!(@backpropagate_final $self [$head $($body)*] [$($body)* $tail]);
    };
    (@backpropagate_final $self:ident [$($offset0:ident)+] [$($offset1:ident)+] ) => {
        $(
            $self.$offset1.backpropagate($self.$offset0.get_deltas());
        )+
    };

    // UPDATE_GRADIENT
    (@update_gradient [$self:ident $batch_size:ident $momentum:ident] [$head1:ident $head2:ident $($body:ident)*] ) => {
        create_nn!(@update_gradient [$self $batch_size $momentum] [$head1 $head2] [$head1 $head2 $($body)*] []);
    };
    (@update_gradient [$self:ident $batch_size:ident $momentum:ident] [$head1:ident $head2:ident] [$buff1:ident $buff2:ident $($body:ident)+] [$($tail:ident)*] ) => {
        create_nn!(@update_gradient [$self $batch_size $momentum] [$head1 $head2] [$buff2 $($body)*] [$($tail)* $buff1]);
    };
    (@update_gradient [$self:ident $batch_size:ident $momentum:ident] [$head1:ident $head2:ident] [$tail2:ident $tail1:ident] [$($body:ident)*] ) => {
        $self.$head1.update_gradient(&$self.input, $self.$head2.get_deltas(), $batch_size, $momentum);

        create_nn!(@update_gradient_count [$self $batch_size $momentum] [$head1 $head2] [$tail2 $tail1] [$($body)*]);

        $self.$tail1.update_gradient($self.$tail2.get_output(), &$self.output_deltas, $batch_size, $momentum);
    };

    (@update_gradient_count [$self:ident $batch_size:ident $momentum:ident] [$head1:ident $head2:ident] [$tail2:ident $tail1:ident] [] ) => {
    };
    (@update_gradient_count [$self:ident $batch_size:ident $momentum:ident] [$head1:ident $head2:ident] [$tail2:ident $tail1:ident] [$body1:ident] ) => {
        $self.$head2.update_gradient($self.$head1.get_output(), $self.$tail1.get_deltas(), $batch_size, $momentum);
    };
    (@update_gradient_count [$self:ident $batch_size:ident $momentum:ident] [$head1:ident $head2:ident] [$tail2:ident $tail1:ident] [$body1:ident $body2:ident $($body:ident)*] ) => {
        create_nn!(@update_gradient_final [$self $batch_size $momentum] [$head1 $head2 $($body)*] [$head2 $($body)* $tail2] [$($body)* $tail2 $tail1]);
    };

    (@update_gradient_final [$self:ident $batch_size:ident $momentum:ident] [$($offset0:ident)+] [$($offset1:ident)+] [$($offset2:ident)+] ) => {
        $(
            $self.$offset1.update_gradient($self.$offset0.get_output(), $self.$offset2.get_deltas(), $batch_size, $momentum);
        )+
    };
}
pub use create_nn as create_neural_net;

pub trait NeuralNetwork {
    type InputType;
    type OutputType;
    type Loss : Loss;

    fn new() -> Self;
    fn feedforward(&mut self, input : &Self::InputType) -> &Self::OutputType;
    fn get_output(&self) -> &Self::OutputType;
    fn backpropagate(&mut self, groundtruth : &Self::OutputType);
    fn update_gradient(&mut self, batch_size: f32, momentum: f32);
    fn update_weights(&mut self, learning_rate:f32, momentum:f32, l2: f32);
    fn get_loss(&self, groundtruth : &Self::OutputType) -> f32;
}

create_nn!(
    MyNN2,
    [layer1:Dense<4,1>, layer2:Dense<1,1>],
    SumSquares1D<1>
);
create_nn!(
    MyNN3,
    [layer1:Dense<4,2>, layer2:Dense<2,1>, layer3:Dense<1,1>],
    SumSquares1D<1>
);
create_nn!(
    MyNN4,
    [dense1:Dense<2,6>, relu1:LeakyRelu1D<6>, dense2:Dense<6,1>, relu2:Relu1D<1>],
    SumSquares1D<1>
);

#[test]
fn test_nn() {
    let x : [f32; 5] = [-1.,-0.5, 0.0, 0.5, 1.];
    let mut x_vec = [[0.0f32;4];5];
    for (i,&xi) in x.iter().enumerate() {
        for j in 0..4{
            x_vec[i][j] = xi.powi(j as i32+1);
        }
    }
    let y = [0.8, 0.1, 0.3, 0.7, 0.5];

    println!("{:?}", x_vec);

    let mut nn2 = MyNN2::new();

    for iter in 1..=5000 {
        let mut mse = 0.0f32;

        for i in 0..5 {
            nn2.feedforward(&x_vec[i]);

            mse += nn2.get_loss(&[y[i]]);

            nn2.backpropagate(&[y[i]]);
            nn2.update_gradient(5., 0.1);
        }
        nn2.update_weights(0.1, 0.1, 0.);

        if iter < 20 {println!("mse: {}", mse);}
        
        if mse < 0.001 {break;}
        assert!(iter != 5000, "NN2 did not converge, mse = {}", mse);
    }

    let mut nn3 = MyNN3::new();

    for iter in 1..=5000 {
        let mut mse = 0.0f32;

        for i in 0..5 {
            nn3.feedforward(&x_vec[i]);

            mse += nn3.get_loss(&[y[i]]);

            nn3.backpropagate(&[y[i]]);
            nn3.update_gradient(5., 0.1);
        }
        nn3.update_weights(0.1, 0.1, 0.);

        if iter < 20 {println!("mse: {}", mse);}
        
        if mse < 0.001 {break;}
        assert!(iter != 5000, "NN3 did not converge, mse = {}", mse);
    }

    let x = [[-1., -1.],[-1., 1.],[1.,-1.],[1.,1.]];
    let y = [1., 0., 0., 1.];

    let mut nn4 = MyNN4::new();

    for iter in 0..=10000 {
        let mut mse = 0.;
        for sample in 0..4 {
            let s = rand::thread_rng().gen_range(0..4usize);

            nn4.feedforward(&x[s]);
            let output = nn4.relu2.get_output();

            let mse1 = SumSquares1D::get_loss(output, &[y[s]]);
            let mse2 = (output[0] - y[s]).powi(2);
            mse += mse1;
            assert!((mse1 - mse2).abs() < 0.00001);
            assert!((mse1 - nn4.get_loss(&[y[s]])).abs() < 0.00001);

            let gradient1 = 2.*(output[0] - y[s]);
            let gradient = SumSquares1D::get_gradient(output, &[y[s]])[0];

            assert!((gradient1 - gradient).abs() < 0.00001);

            nn4.backpropagate(&[y[s]]);
            assert!((gradient1 - nn4.output_deltas[0]).abs() < 0.00001);
            nn4.update_gradient(4., 0.1);
        }
        if iter < 10 && iter%1==0 {println!("nn4: mse: {}", mse);}

        nn4.update_weights(0.1,0.1,0.001);
        
        
        
        if mse < 0.01 {break;}
        assert!(iter != 10000, "NN4 did not converge, mse = {}", mse);
    }

}


create_nn!(
    MyNN5,
    [dense1:Dense<2,2>, relu1:LeakyRelu1D<2>, dense2:Dense<2,2>, relu2:Softmax<2>],
    CrossEntropy1D<2>
);
#[test]
fn test_nn2() {
    let x : [[f32; 2];4]= [[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]];
    let y : [[f32;2];4] = [[1., 0.], [0., 1.], [0., 1.], [1., 0.]];

    let mut nn = MyNN5::new();

    nn.dense1.weights = [[1., -1.],[1., -1.]];
    nn.dense1.bias = [0., 0.];
    nn.dense2.weights = [[1., -1.],[1., -1.]];
    nn.dense2.bias = [0., 2.];


    for iter in 1..=1000 {
        for _ in 0..2 {
            let sample = rand::thread_rng().gen_range(0..4);
            nn.feedforward(&x[sample]);

            nn.backpropagate(&y[sample]);
            nn.update_gradient(2., 0.);
        }
        nn.update_weights(0.1, 0., 0.000005);

        let mut loss = 0.0f32;
        for (xi, yi) in x.iter().zip(y.iter()) {
            nn.feedforward(xi);
            let out = nn.get_output();
            assert!(out[0]<=1. && out[1] <= 1., "out = {:?}, softmax inp: {:?}", out, nn.dense2.get_output());
            loss += nn.get_loss(yi);
        }

        if iter < 20 && iter%1==0 {println!("loss: {}", loss);}
        
        if loss < 0.1 {break;}
        /*if iter == 1000 {
            println!("ITER {}", iter);
            for xi in x.iter() {
                nn.feedforward(xi);
                let out = nn.get_output();
                println!("Input: {:?}, output: {:?}", xi, out);
            }

            println!("{:?} {:?}", nn.dense2.weights, nn.dense2.bias);
            println!("{:?} {:?}", nn.dense1.weights, nn.dense1.bias);
        }*/
        assert!(iter != 1000, "NN did not converge, loss = {}", loss);
    }
}