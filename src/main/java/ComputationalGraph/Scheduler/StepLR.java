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
     */
    @Override
    protected double call() {
        int currentEpoch = getEpoch();
        if (currentEpoch / this.stepSize > (currentEpoch - 1) / this.stepSize) {
            return this.learningRate * etaDecrease;
        }
        return this.learningRate;
    }
}
