package ComputationalGraph.Scheduler;

import java.io.Serializable;

public class CosineAnnealingLR extends Scheduler implements Serializable {

    private final double etaMin;
    private final double tMax;

    public CosineAnnealingLR(double initialLearningRate, double etaMin, double tMax) {
        super(initialLearningRate);
        this.etaMin = etaMin;
        this.tMax = tMax;
    }

    public CosineAnnealingLR(double etaMin, double tMax) {
        super();
        this.etaMin = etaMin;
        this.tMax = tMax;
    }

    /**
     * Calculates the updated learning rate using the cosine annealing schedule.
     * The learning rate decreases over time according to a cosine function,
     * and is bounded between the minimum learning rate (`etaMin`) and the
     * initial learning rate.
     * @return The updated learning rate for the current epoch.
     */
    @Override
    protected double call() {
        double progress = (double) getEpoch() / this.tMax;
        if (progress > 1.0) {
            progress = 1.0;
        }
        double cosineValue = Math.cos(Math.PI * progress);
        return this.etaMin + 0.5 * (this.initialLearningRate - this.etaMin) * (1.0 + cosineValue);
    }
}
