package ComputationalGraph.Scheduler;

import java.io.Serializable;

public abstract class Scheduler implements Serializable {

    protected final double initialLearningRate;
    private int epoch;

    public Scheduler(double initialLearningRate) {
        this.initialLearningRate = initialLearningRate;
        this.epoch = 0;
    }

    public double updateLearningRate() {
        this.epoch++;
        return call();
    }

    protected int getEpoch() {
        return epoch;
    }

    public double getInitialLearningRate() {
        return initialLearningRate;
    }

    protected abstract double call();
}
