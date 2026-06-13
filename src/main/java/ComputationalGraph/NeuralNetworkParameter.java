package ComputationalGraph;

import Classification.Parameter.Parameter;
import ComputationalGraph.Loss.CrossEntropyLoss;
import ComputationalGraph.Loss.Loss;
import ComputationalGraph.Initialization.Initialization;
import ComputationalGraph.Initialization.RandomInitialization;
import ComputationalGraph.Optimizer.Optimizer;

import Math.Tensor;
import java.util.Random;

public class NeuralNetworkParameter extends Parameter implements java.io.Serializable {

    private final Optimizer optimizer;
    private final int epoch;
    private final Initialization initialization;
    private final double dropout;
    private final Loss lossFunction;
    private final int batchDimension;

    public NeuralNetworkParameter(int seed, int epoch, Optimizer optimizer, Initialization initialization, Loss lossFunction, double dropout, int batchDimension) {
        super(seed);
        this.optimizer = optimizer;
        this.epoch = epoch;
        this.initialization = initialization;
        this.dropout = dropout;
        this.lossFunction = lossFunction;
        this.batchDimension = batchDimension;
    }

    public NeuralNetworkParameter(int seed, int epoch, Optimizer optimizer) {
        super(seed);
        this.optimizer = optimizer;
        this.epoch = epoch;
        this.initialization = new RandomInitialization();
        this.dropout = 0.0;
        this.lossFunction = new CrossEntropyLoss();
        this.batchDimension = -1;
    }

    public NeuralNetworkParameter(int seed, int epoch, Optimizer optimizer, Loss lossFunction, double dropout) {
        super(seed);
        this.optimizer = optimizer;
        this.epoch = epoch;
        this.initialization = new RandomInitialization();
        this.dropout = dropout;
        this.lossFunction = lossFunction;
        this.batchDimension = -1;
    }

    public Optimizer getOptimizer() {
        return optimizer;
    }

    public int getEpoch() {
        return epoch;
    }

    public Tensor initializeWeights(int[] shape, Random random) {
        return initialization.initialize(shape, random);
    }

    public double getDropout() {
        return dropout;
    }

    public Loss getLossFunction() {
        return lossFunction;
    }

    public int getBatchDimension() {
        return batchDimension;
    }
}
