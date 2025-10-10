package ComputationalGraph;

import Classification.Parameter.Parameter;
import ComputationalGraph.Initialization.Initialization;
import ComputationalGraph.Initialization.RandomInitialization;
import ComputationalGraph.Optimizer.Optimizer;

public class NeuralNetworkParameter extends Parameter {

    private final Optimizer optimizer;
    private final int epoch;
    private final Initialization initialization;

    public NeuralNetworkParameter(int seed, int epoch, Optimizer optimizer, Initialization initialization) {
        super(seed);
        this.optimizer = optimizer;
        this.epoch = epoch;
        this.initialization = initialization;
    }

    public NeuralNetworkParameter(int seed, int epoch, Optimizer optimizer) {
        super(seed);
        this.optimizer = optimizer;
        this.epoch = epoch;
        this.initialization = new RandomInitialization();
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
}
