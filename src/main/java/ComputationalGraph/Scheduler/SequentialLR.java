package ComputationalGraph.Scheduler;

import java.util.ArrayList;

public class SequentialLR extends Scheduler implements java.io.Serializable {

    private final ArrayList<Scheduler> schedulers;
    private final ArrayList<Integer> milestones;
    private int curIndex;
    private double lastLearningRate;

    public SequentialLR(double initialLearningRate, ArrayList<Scheduler> schedulers, ArrayList<Integer> milestones) {
        super(initialLearningRate);
        if (schedulers.size() != milestones.size() + 1) {
            throw new IllegalArgumentException("Schedulers and milestones must be matching in size.");
        }
        this.schedulers = schedulers;
        this.milestones = milestones;
        this.curIndex = 0;
        schedulers.get(curIndex).setInitialLearningRate(initialLearningRate);
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
        if (curIndex == milestones.size() || getEpoch() <= milestones.get(curIndex)) {
            lastLearningRate = schedulers.get(curIndex).updateLearningRate();
            return lastLearningRate;
        }
        curIndex++;
        schedulers.get(curIndex).setInitialLearningRate(lastLearningRate);
        lastLearningRate = schedulers.get(curIndex).updateLearningRate();
        return lastLearningRate;
    }
}
