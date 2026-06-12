package ComputationalGraph.Scheduler;

import java.io.Serializable;

public class ExponentialLR extends Scheduler implements Serializable {

    protected final double etaDecrease;

    public ExponentialLR(double initialLearningRate, double etaDecrease) {
        super(initialLearningRate);
        this.etaDecrease = etaDecrease;
    }

    /**
     * Computes the updated learning rate based on an exponential decay schedule.
     * The learning rate decreases exponentially over epochs, where the rate
     * of decay is determined by the `etaDecrease` factor.
     * @return The updated learning rate after applying the exponential decay formula.
     */
    @Override
    protected double call() {
        return this.initialLearningRate * Math.pow(etaDecrease, getEpoch());
    }
}
