package ie.gmit.sw.ai.runner;

import ie.gmit.sw.ai.nn.BackpropagationTrainer;
import ie.gmit.sw.ai.nn.NeuralNetwork;
import ie.gmit.sw.ai.nn.Utils;
import ie.gmit.sw.ai.nn.activator.Activator;

public class SignalRunner {

    public SignalRunner() throws Exception {
        // 1. Training set data for Neural Network, the size of the data e.g. 1,1,1,0 should match the number of nodes
        // Inputs
        double[][] data = { { 1, 1, 1, 0 }, { 1, 1, 0, 0 }, { 0, 1, 1, 0 }, { 1, 0, 1, 0 }, { 1, 0, 0, 0 },
            { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 1, 1, 1, 1 }, { 1, 1, 0, 1 }, { 0, 1, 1, 1 },
            { 1, 0, 1, 1 }, { 1, 0, 0, 1 }, { 0, 1, 0, 1 }, { 0, 0, 1, 1 } };

        // Expected Outputs, the size of data should match the number of nodes in the output layer
        double expected[][] = {
                { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 } };

        // 2. Create a neural network with activator function and nodes in input,hidden and output layer.
        // change the hidden layers nodes to 18, 17, 28 and 126
        NeuralNetwork nn = new NeuralNetwork(Activator.ActivationFunction.Sigmoid, 4, 126, 14);

        // 3. Instantiate the back-propagation algorithm
        BackpropagationTrainer trainer = new BackpropagationTrainer(nn);
        // start training the Neural Network with the training input data, expected outputs, learning rate and epochs
        trainer.train(data, expected, 0.2, 500);

        // 4. Create and feed in the test data
        double[] test = {1,1,0,0};
        double[] result = nn.process(test);
        System.out.println(Utils.getMaxIndex(result) + 1);
    }// SignalRunner

    public static void main(String[] args) throws Exception {
        new SignalRunner();
    }// main
}// SignalRunner
