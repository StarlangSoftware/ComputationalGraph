package ComputationalGraph;

import Classification.Parameter.Parameter;
import ComputationalGraph.Function.CrossEntropyLoss;
import ComputationalGraph.Function.Function;
import ComputationalGraph.Initialization.Initialization;
import ComputationalGraph.Initialization.RandomInitialization;
import ComputationalGraph.Optimizer.Optimizer;

import java.util.ArrayList;
import java.util.Random;

public class NeuralNetworkParameter extends Parameter {

    private final Optimizer optimizer;
    private final int epoch;
    private final Initialization initialization;
    private final double dropout;
    private final Function lossFunction;
    private final int batchSize;

    public NeuralNetworkParameter(int seed, int epoch, Optimizer optimizer, Initialization initialization, Function lossFunction, double dropout, int batchSize) {
        super(seed);
        this.optimizer = optimizer;
        this.epoch = epoch;
        this.initialization = initialization;
        this.dropout = dropout;
        this.lossFunction = lossFunction;
        this.batchSize = batchSize;
    }

    public NeuralNetworkParameter(int seed, int epoch, Optimizer optimizer) {
        super(seed);
        this.optimizer = optimizer;
        this.epoch = epoch;
        this.initialization = new RandomInitialization();
        this.dropout = 0.0;
        this.lossFunction = new CrossEntropyLoss();
        this.batchSize = 1;
    }

    public NeuralNetworkParameter(int seed, int epoch, Optimizer optimizer, Function lossFunction, double dropout) {
        super(seed);
        this.optimizer = optimizer;
        this.epoch = epoch;
        this.initialization = new RandomInitialization();
        this.dropout = dropout;
        this.lossFunction = lossFunction;
        this.batchSize = 1;
    }

    public Optimizer getOptimizer() {
        return optimizer;
    }

    public int getEpoch() {
        return epoch;
    }

    public ArrayList<Double> initializeWeights(int row, int column, Random random) {
        return initialization.initialize(row, column, random);
    }

    public double getDropout() {
        return dropout;
    }

    public Function getLossFunction() {
        return lossFunction;
    }

    public int getBatchSize() {
        return batchSize;
    }
}
