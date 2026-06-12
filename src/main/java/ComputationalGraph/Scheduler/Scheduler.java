package ComputationalGraph.Scheduler;

import java.io.Serializable;

public abstract class Scheduler implements Serializable {

    protected double initialLearningRate;
    private int epoch;

    public Scheduler(double initialLearningRate) {
        this.initialLearningRate = initialLearningRate;
        this.epoch = 0;
    }

    public Scheduler() {
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

    protected void setInitialLearningRate(double initialLearningRate) {
        this.initialLearningRate = initialLearningRate;
    }

    protected abstract double call();
}
