package ComputationalGraph.Scheduler;

public class SequentialLR extends Scheduler implements java.io.Serializable {

    private final Scheduler[] schedulers;
    private final int[] milestones;
    private int curIndex;
    private double lastLearningRate;

    public SequentialLR(double initialLearningRate, Scheduler[] schedulers, int[] milestones) {
        super(initialLearningRate);
        if (schedulers.length != milestones.length + 1) {
            throw new IllegalArgumentException("Schedulers and milestones must be matching in size.");
        }
        this.schedulers = schedulers;
        this.milestones = milestones;
        this.curIndex = 0;
        schedulers[curIndex].setInitialLearningRate(initialLearningRate);
    }

    /**
     * Calculates and updates the learning rate using a sequence of distinct learning rate schedulers.
     * The current scheduler is selected based on the epoch count and predefined milestones.
     * At each epoch, the method determines whether to use the current scheduler or transition to the next one.
     * Transitions occur when the epoch count surpasses the milestone associated with the current scheduler.
     * The updated learning rate is computed by invoking the `updateLearningRate` method of the current scheduler.
     * @return The updated learning rate computed by the active scheduler.
     */
    @Override
    protected double call() {
        if (curIndex == milestones.length || getEpoch() <= milestones[curIndex]) {
            lastLearningRate = schedulers[curIndex].updateLearningRate();
            return lastLearningRate;
        }
        curIndex++;
        schedulers[curIndex].setInitialLearningRate(lastLearningRate);
        lastLearningRate = schedulers[curIndex].updateLearningRate();
        return lastLearningRate;
    }
}
