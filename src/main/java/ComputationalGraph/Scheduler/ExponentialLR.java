package ComputationalGraph.Scheduler;

import java.io.Serializable;

public class ExponentialLR extends Scheduler implements Serializable {

    protected final double etaDecrease;

    public ExponentialLR(double initialLearningRate, double etaDecrease) {
        super(initialLearningRate);
        this.etaDecrease = etaDecrease;
    }

    @Override
    protected double call() {
        return this.initialLearningRate * Math.pow(etaDecrease, getEpoch());
    }
}
