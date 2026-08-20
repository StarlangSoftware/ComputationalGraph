package ComputationalGraph.Scheduler;

import java.io.Serializable;

public class ExponentialLR extends Scheduler implements Serializable {

    protected final double etaDecrease;

    public ExponentialLR(double initialLearningRate, double etaDecrease) {
        super(initialLearningRate);
        this.etaDecrease = etaDecrease;
    }

    public ExponentialLR(double etaDecrease) {
        super();
        this.etaDecrease = etaDecrease;
    }

    /**
     * Computes the updated learning rate based on an exponential decay schedule.
     * The learning rate decreases exponentially over epochs, where the rate
     * of decay is determined by the `etaDecrease` factor.
     * @param epoch current epoch of the optimizer.
     */
    @Override
    public double call(int epoch) {
        double initialLearningRate = this.getInitialLearningRate();
        return initialLearningRate * Math.pow(etaDecrease, epoch);
    }
}
