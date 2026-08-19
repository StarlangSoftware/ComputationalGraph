package ComputationalGraph.Scheduler;

import java.io.Serializable;

public abstract class Scheduler implements Serializable {

    protected double learningRate;
    private int epoch;

    public Scheduler(double initialLearningRate) {
        this.learningRate = initialLearningRate;
        this.epoch = 0;
    }

    public Scheduler() {
        this.epoch = 0;
    }

    public void updateLearningRate() {
        this.epoch++;
        this.learningRate = call();
    }

    protected int getEpoch() {
        return epoch;
    }

    public double getLearningRate() {
        return learningRate;
    }

    protected abstract double call();
}
