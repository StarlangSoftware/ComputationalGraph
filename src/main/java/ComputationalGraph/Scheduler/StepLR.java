package ComputationalGraph.Scheduler;

import java.io.Serializable;

public class StepLR extends ExponentialLR implements Serializable {

    private final int stepSize;

    public StepLR(double initialLearningRate, double etaDecrease, int stepSize) {
        super(initialLearningRate, etaDecrease);
        this.stepSize = stepSize;
    }

    public StepLR(double etaDecrease, int stepSize) {
        super(etaDecrease);
        this.stepSize = stepSize;
    }

    /**
     * Computes the updated learning rate based on a step decay schedule.
     * The learning rate is reduced at regular intervals determined by the step size.
     * The decay factor is applied to every `stepSize` epoch to calculate the new learning rate.
     * @param epoch current epoch of the optimizer.
     */
    @Override
    public double call(int epoch) {
        double initialLearningRate = this.getInitialLearningRate();
        if (initialLearningRate == Double.MIN_VALUE) {
            throw new IllegalArgumentException("Learning rate must be initialized first.");
        }
        int period = epoch / this.stepSize;
        return initialLearningRate * Math.pow(etaDecrease, period);
    }
}
