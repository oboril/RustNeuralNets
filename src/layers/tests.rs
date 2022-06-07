use crate::losses::{SumSquares1D, Loss};

use super::dense::Dense;
use super::relu::Relu1D;
use super::Layer;


#[test]
fn test_dense_orthogonal_init() {
    let dense = Dense::<4,4>::new();

    // columns are normalized
    for i in 0..4usize {
        let mut norm = 0.;
        for j in 0..4usize {
            norm += dense.weights[i][j].powi(2);
        }
        assert!((norm-1.).abs() < 0.0001);
    }

    // rows are normalized
    for i in 0..4usize {
        let mut norm = 0.;
        for j in 0..4usize {
            norm += dense.weights[j][i].powi(2);
        }
        assert!((norm-1.).abs() < 0.0001);
    }

    // rows are orthogonal
    for row1 in 0..4usize {
        for row2 in 0..row1 {
            let mut dot_product = 0.;
            for col in 0..4usize {dot_product += dense.weights[row1][col] * dense.weights[row2][col]; }
            assert!(dot_product.abs() < 0.00001);
        }
    }

    // columns are orthogonal
    for col1 in 0..4usize {
        for col2 in 0..col1 {
            let mut dot_product = 0.;
            for row in 0..4usize {dot_product += dense.weights[row][col1] * dense.weights[row][col2]; }
            assert!(dot_product.abs() < 0.00001);
        }
    }


    let dense = Dense::<3,4>::new();

    // rows are normalized
    for row in 0..3usize{
        let mut norm = 0.;
        for col in 0..4usize {
            norm += dense.weights[row][col].powi(2);
        }
        assert!((norm-1.).abs() < 0.0001);
    }

    // rows are orthogonal
    for row1 in 0..3usize {
        for row2 in 0..row1 {
            let mut dot_product = 0.;
            for col in 0..4usize {dot_product += dense.weights[row1][col] * dense.weights[row2][col]; }
            assert!(dot_product.abs() < 0.00001);
        }
    }
}

#[test]
fn test_xor_dense_relu() {
    let x = [[-1., -1.],[-1., 1.],[1.,-1.],[1.,1.]];
    let y = [1., 0., 0., 1.];

    let mut dense1 = Dense::<2,2>::new();
    let mut relu1 = Relu1D::<2>::new();
    let mut dense2 = Dense::<2,1>::new();
    let mut relu2 = Relu1D::<1>::new();

    dense1.weights = [[0.8, -1.1],[0.9, -0.6]];
    dense1.bias = [0.3, -0.1];
    dense2.weights = [[0.4], [0.6]];
    dense2.bias = [0.1];


    for iter in 0..1001 {
        let mut mse = 0.;
        for sample in 0..4 {
            dense1.feedforward(&x[sample]);
            relu1.feedforward(dense1.get_output());
            dense2.feedforward(relu1.get_output());
            relu2.feedforward(dense2.get_output());
            let output = relu2.get_output();

            mse += (output[0] - y[sample]).powi(2);
            let gradient1 = 2.*(output[0] - y[sample]);
            let gradient = SumSquares1D::get_gradient(output, &[y[sample]])[0];

            assert!((gradient1 - gradient).abs() < 0.00001);

            relu2.backpropagate(&[gradient]);
            dense2.backpropagate(relu2.get_deltas());
            relu1.backpropagate(dense2.get_deltas());
            dense1.backpropagate(relu1.get_deltas());

            dense1.update_gradient(&x[sample], relu1.get_deltas(), 4., 0.1);
            relu1.update_gradient(dense1.get_output(), dense2.get_deltas(), 4., 0.1);
            dense2.update_gradient(relu1.get_output(), relu2.get_deltas(), 4., 0.1);
            relu2.update_gradient(dense2.get_output(), &[gradient], 4., 0.1);
        }

        dense1.update_weights(0.2, 0.1, 0.001);
        relu1.update_weights(0.2, 0.1, 0.001);
        dense2.update_weights(0.2, 0.1, 0.001);
        relu2.update_weights(0.2, 0.1, 0.001);

        if mse < 0.001 {
            break;
        }
        if iter == 1000 {
            for (i,input) in x.iter().enumerate() {
                dense1.feedforward(input);
                relu1.feedforward(dense1.get_output());
                dense2.feedforward(relu1.get_output());
                relu2.feedforward(dense2.get_output());
                let output = relu2.get_output();

                println!("Input: {:?}, output: {}, gradient: {}", input, output[0], SumSquares1D::get_gradient(output, &[y[i]])[0]);
            }
            panic!("XOR did not converge, MSE = {}", mse);
        }
    }
}

#[test]
fn test_dense_l2() {
    let mut dense = Dense::<1,1>::new();

    for _ in 0..1000 {
        dense.update_weights(0., 0., 0.1)
    }

    assert!(dense.bias[0].abs() < 0.00001);
    assert!(dense.weights[0][0].abs() < 0.00001);
}

#[test]
fn test_dense() {
    const TRAIN: usize = 4;
    const INPUT: usize = 3;

    let mut dense = Dense::<INPUT,1>::new();

    let mut inputs : [[f32; INPUT]; TRAIN] = [[0.; INPUT]; TRAIN];

    let x: [f32;TRAIN] = [-0.5, 0., 0.5, 1.];

    for i in 0..TRAIN {
        for pow in 0..INPUT {
            inputs[i][pow] = x[i].powi((pow+1) as i32);
        }
    }

    let y: [f32; TRAIN] = [0., 5., -3., -4.];//, 7., 1.];

    for iter in 0..10000 {
        let mut total_mse = 0.;
        for i in 0..TRAIN{
            dense.feedforward(&inputs[i]);

            let output = dense.get_output();
            total_mse += (output[0] - y[i]).powi(2);
            let deltas = [2.*(output[0]-y[i])]; //MSE gradient

            dense.backpropagate(&deltas);
            dense.update_gradient(&inputs[i], &deltas, TRAIN as f32, 0.);
        }
        dense.update_weights(0.5, 0., 0.);
        
        if total_mse < 0.00001{
            println!("Iterations {}, mse {}", iter, total_mse);
            break;
        }
        else if iter == 10000-1 {
            panic!("Dense layer did not converge. MSE: {}", total_mse);
        }
    }
}

#[test]
fn test_dense2() {
    let mut dense = Dense::<1,1>::new();

    let x : [[f32;1];2] = [[0.],[1.]];

    let y: [f32; 2] = [1.,2.];

    for iter in 0..100 {
        let mut total_mse = 0.;
        for i in 0..2{
            dense.feedforward(&x[i]);

            let output = dense.get_output();
            total_mse += (output[0] - y[i]).powi(2);
            let deltas = [2.*(output[0]-y[i])]; //MSE gradient

            dense.backpropagate(&deltas);
            dense.update_gradient(&x[i], &deltas, 2 as f32, 0.);
        }
        
        if total_mse < 0.00001 {
            break;
        }
        if iter == 99 {
            panic!("Dense layer did not converge");
        }

        dense.update_weights(0.5, 0., 0.);
    };
}