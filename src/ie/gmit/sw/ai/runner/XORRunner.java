package ie.gmit.sw.ai.runner;

import ie.gmit.sw.ai.nn.BackpropagationTrainer;
import ie.gmit.sw.ai.nn.NeuralNetwork;
import ie.gmit.sw.ai.nn.activator.Activator;

public class XORRunner {

    public static long getRoundedValue(double[] vector){
        return Math.round(vector[0]);
    }// getRoundedValue

    public XORRunner() throws Exception {
        // Training Set
        double[][] data = {{0, 0},{1, 0},{0, 1},{1, 1}};// inputs
        double[][] expected = {{0},{1},{1},{0}};// outputs

        // Create and instance of the NeuralNetwork with 2 input nodes, 2 hidden nodes, and 1 output node
        // and a sigmoidal activation function
        NeuralNetwork nn = new NeuralNetwork(Activator.ActivationFunction.Sigmoid, 2, 2, 1);

        // Instantiate back-propagation training algorithm and train the network using the training set, with a
        // learning rate of 0.01 and 1000000 epochs
        BackpropagationTrainer trainer = new BackpropagationTrainer(nn);
        // the fewer epochs with the a vaguer learning rate makes the NN less accurate
        trainer.train(data, expected, 0.01, 1000000);

        // Create 4 data sets that will be used to test the Neural Network after it is fully trained
        // the data sets relate to the truth table permutations for the two operands of the XOR operator
        double[] test1 = {0.0, 0.0};// false
        double[] test2 = {1.0, 0.0};// true
        double[] test3 = {0.0, 1.0};// true
        double[] test4 = {1.0, 1.0};// false

        // Execute each test and round the results using the static function defined above
        System.out.println("00=>" + getRoundedValue(nn.process(test1)));
        System.out.println("01=>" + getRoundedValue(nn.process(test2)));
        System.out.println("02=>" + getRoundedValue(nn.process(test3)));
        System.out.println("03=>" + getRoundedValue(nn.process(test4)));
    }

    public static void main(String[] args) throws Exception {
        new XORRunner();
    }// void
}// XORRunner
