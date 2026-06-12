package ComputationalGraph.Scheduler;

import java.io.Serializable;

public class StepLR extends ExponentialLR implements Serializable {

    private final int stepSize;

    public StepLR(double initialLearningRate, double etaDecrease, int stepSize) {
        super(initialLearningRate, etaDecrease);
        this.stepSize = stepSize;
    }

    /**
     * Computes the updated learning rate based on a step decay schedule.
     * The learning rate is reduced at regular intervals determined by the step size.
     * The decay factor is applied to every `stepSize` epoch to calculate the new learning rate.
     * @return The updated learning rate after applying the step decay formula.
     */
    @Override
    protected double call() {
        int period = getEpoch() / this.stepSize;
        return this.initialLearningRate * Math.pow(etaDecrease, period);
    }
}
