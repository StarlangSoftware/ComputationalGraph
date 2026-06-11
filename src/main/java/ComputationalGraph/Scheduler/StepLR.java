package ComputationalGraph.Scheduler;

import java.io.Serializable;

public class StepLR extends ExponentialLR implements Serializable {

    private final int stepSize;

    public StepLR(double initialLearningRate, double etaDecrease, int stepSize) {
        super(initialLearningRate, etaDecrease);
        this.stepSize = stepSize;
    }

    @Override
    protected double call() {
        int period = getEpoch() / this.stepSize;
        return this.initialLearningRate * Math.pow(etaDecrease, period);
    }
}
