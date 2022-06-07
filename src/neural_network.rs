use super::layers::{Dense, Relu1D, Layer};
use super::losses::{Loss, MSE1D};

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


/*


    // THIS IS THE INPUT STRUCTURE
    (
        $struct_name:ident;
        $first_layer_name:ident : $first_layer_type:ty,
        $($layer_name:ident : $layer_type:ty),*;
        $last_layer_name:ident : $last_layer_type:ty;
        $loss:ty
    )
    // ENDS HERE
    => {
        struct $struct_name {
            $first_layer_name : $first_layer_type,
            $($layer_name : $layer_type),*,
            $last_layer_name : $last_layer_type,
            input : <$first_layer_type as Layer>::InputType,
            output_deltas : <$last_layer_type as Layer>::OutputType
        }

        impl NeuralNetwork for $struct_name {
            type InputType = <$first_layer_type as Layer>::InputType;
            type OutputType = <$last_layer_type as Layer>::OutputType;
            type Loss = $loss;

            fn new() -> Self {
                let new_nn = $struct_name {
                    $first_layer_name : <$first_layer_type>::new(),
                    $($layer_name: <$layer_type>::new()),+,
                    $last_layer_name : <$last_layer_type>::new(),
                    input: Self::InputType::default(),
                    output_deltas: Self::OutputType::default()
                };

                return new_nn;
            }

            fn update_weights(&mut self, learning_rate:f32, momentum:f32, l2:f32) {
                self.$first_layer_name.update_weights(learning_rate, momentum, l2);
                $(self.$layer_name.update_weights(learning_rate, momentum, l2);)+
                self.$last_layer_name.update_weights(learning_rate, momentum, l2);
            }

            fn predict(&mut self, input : &Self::InputType) -> &Self::OutputType {
                self.input = input.clone();
                self.$first_layer_name.feedforward(&self.input);

                create_nn!(@feedforward self, $first_layer_name,$($layer_name),+; $($layer_name),+, $last_layer_name);

                return self.$last_layer_name.get_output();
            }

            fn get_output(&self) -> &Self::OutputType {
                return self.$last_layer_name.get_output();
            }

            fn backpropagate(&mut self, groundtruth:&Self::OutputType) {
                self.output_deltas = Self::Loss::get_gradient(self.get_output(), groundtruth);
                self.$last_layer_name.backpropagate(&self.output_deltas);
                create_nn!(@backpropagate self, [$first_layer_name, $($layer_name),*, $last_layer_name]);
            }

            fn update_gradient(&mut self, batch_size: f32, momentum: f32) {
                create_nn!(@update_gradient [self batch_size momentum] [$first_layer_name])
            }
        }
    };*/
}

trait NeuralNetwork {
    type InputType;
    type OutputType;
    type Loss : Loss;

    fn new() -> Self;
    fn feedforward(&mut self, input : &Self::InputType) -> &Self::OutputType;
    fn get_output(&self) -> &Self::OutputType;
    fn backpropagate(&mut self, groundtruth : &Self::OutputType);
    fn update_gradient(&mut self, batch_size: f32, momentum: f32);
    fn update_weights(&mut self, learning_rate:f32, momentum:f32, l2: f32);
}

create_nn!(
    MyNN2,
    [layer1:Dense<2,4>, layer2:Relu1D<4>],
    MSE1D<4>
);
create_nn!(
    MyNN3,
    [layer1:Dense<2,4>, layer2:Relu1D<4>, layer3:Dense<4,1>],
    MSE1D<1>
);
create_nn!(
    MyNN4,
    [layer1:Dense<2,4>, layer2:Relu1D<4>, layer3:Dense<4,1>, layer4:Relu1D<1>],
    MSE1D<1>
);

#[test]
fn test() {
    let mut nn = MyNN2::new();

    println!("{:?}", nn.input);
    println!("{:?}", nn.feedforward(&[1., 0.5]));
    
    panic!("Im here");
}
