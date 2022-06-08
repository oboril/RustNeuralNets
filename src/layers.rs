mod dense;
mod relu;
mod leaky_relu;
mod sigmoid;
mod softmax;
pub use dense::Dense;
pub use relu::Relu1D;
pub use leaky_relu::LeakyRelu1D;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;

#[cfg(test)]
mod tests;

pub trait Layer {
    type InputType;
    type OutputType;

    /// Return instance of self
    fn new() -> Self;

    /// Updates the layer output values
    fn feedforward(&mut self, prev_output: &Self::InputType);

    /// Updates the layer deltas values
    fn backpropagate(&mut self, next_deltas : &Self::OutputType);

    /// Accumulates the layer gradient
    fn update_gradient(&mut self, prev_output: &Self::InputType, next_deltas : &Self::OutputType, batch_size: f32, momentum: f32);

    /// Uses the accumulated gradient to update weights. Resets gradient
    fn update_weights(&mut self, learning_rate: f32, momentum: f32, l2: f32);

    /// Returns the layer output - only returns the cached value, does not do any computation
    fn get_output(&self) -> &Self::OutputType;

    /// Returns the gradient dC/dx where C is the cost function and x is the layer input
    fn get_deltas(&self) -> &Self::InputType;
}