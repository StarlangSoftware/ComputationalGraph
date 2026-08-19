package ComputationalGraph.Scheduler;

import java.io.Serializable;

public abstract class Scheduler implements Serializable {

    private double initialLearningRate;

    public Scheduler(double initialLearningRate) {
        this.initialLearningRate = initialLearningRate;
    }

    public Scheduler() {
        this(Double.MIN_VALUE);
    }

    protected double getInitialLearningRate() {
        return initialLearningRate;
    }

    protected void setInitialLearningRate(double initialLearningRate) {
        if (this.initialLearningRate == Double.MIN_VALUE) {
           this.initialLearningRate = initialLearningRate;
        }
    }

    public abstract double call(int epoch);
}
