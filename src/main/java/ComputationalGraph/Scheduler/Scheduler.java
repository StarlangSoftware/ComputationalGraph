package ComputationalGraph.Scheduler;

import java.io.Serializable;

public abstract class Scheduler implements Serializable {

    private double initialLearningRate;
    private boolean initialized;

    public Scheduler(double initialLearningRate) {
        this.initialLearningRate = initialLearningRate;
        this.initialized = true;
    }

    public Scheduler() {
        this.initialized = false;
    }

    protected double getInitialLearningRate() {
        if (!initialized) {
            throw new IllegalArgumentException("Learning rate must be initialized first.");
        }
        return initialLearningRate;
    }

    protected void setInitialLearningRate(double initialLearningRate) {
        this.initialLearningRate = initialLearningRate;
        this.initialized = true;
    }

    public abstract double call(int epoch);
}
