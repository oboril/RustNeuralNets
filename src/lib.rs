#![allow(dead_code)]
#![allow(unused_variables)]

mod layers;
mod neural_network;
mod losses;


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
