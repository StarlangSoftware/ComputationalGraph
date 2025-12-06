package ComputationalGraph;

import Classification.Parameter.Parameter;
import ComputationalGraph.Initialization.Initialization;
import ComputationalGraph.Initialization.RandomInitialization;
import ComputationalGraph.Optimizer.Optimizer;

public class NeuralNetworkParameter extends Parameter {

    private final Optimizer optimizer;
    private final int epoch;
    private final Initialization initialization;
    private final double dropout;

    public NeuralNetworkParameter(int seed, int epoch, Optimizer optimizer, Initialization initialization, double dropout) {
        super(seed);
        this.optimizer = optimizer;
        this.epoch = epoch;
        this.initialization = initialization;
        this.dropout = dropout;
    }

    public NeuralNetworkParameter(int seed, int epoch, Optimizer optimizer) {
        super(seed);
        this.optimizer = optimizer;
        this.epoch = epoch;
        this.initialization = new RandomInitialization();
        this.dropout = 0.0;
    }

    public NeuralNetworkParameter(int seed, int epoch, Optimizer optimizer, double dropout) {
        super(seed);
        this.optimizer = optimizer;
        this.epoch = epoch;
        this.initialization = new RandomInitialization();
        this.dropout = dropout;
    }

    public Optimizer getOptimizer() {
        return optimizer;
    }

    public int getEpoch() {
        return epoch;
    }

    public Initialization getInitialization() {
        return initialization;
    }

    public double getDropout() {
        return dropout;
    }
}
