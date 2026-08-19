package ComputationalGraph.Scheduler;

public class SequentialLR extends Scheduler implements java.io.Serializable {

    private final Scheduler[] schedulers;
    private final int[] milestones;

    public SequentialLR(double initialLearningRate, Scheduler[] schedulers, int[] milestones) {
        super(initialLearningRate);
        if (schedulers.length != milestones.length + 1) {
            throw new IllegalArgumentException("Schedulers and milestones must be matching in size.");
        }
        this.schedulers = schedulers;
        this.milestones = milestones;
        if (milestones[0] < 0) {
            throw new IllegalArgumentException("Milestones must be positive.");
        }
        for (int i = 0; i < milestones.length - 1; i++) {
            if (milestones[i + 1] < 0) {
                throw new IllegalArgumentException("Milestones must be positive.");
            }
            if (milestones[i + 1] <= milestones[i]) {
                throw new IllegalArgumentException("Milestones must be increasing.");
            }
        }
        schedulers[0].setInitialLearningRate(initialLearningRate);
    }

    private int[] helper(int epoch, int min, int max) {
        int mid = (min + max) / 2;
        if (milestones[mid] == epoch) {
            return new int[]{mid + 1, 0};
        } else if (milestones[mid] > epoch) {
            if (mid - 1 >= 0) {
                if (milestones[mid - 1] < epoch) {
                    return new int[]{mid, epoch - milestones[mid - 1]};
                } else {
                    return helper(epoch, min, mid);
                }
            } else {
                return new int[]{0, epoch};
            }
        } else {
            if (mid + 1 < milestones.length) {
                if (milestones[mid + 1] > epoch) {
                    return new int[]{mid + 1, epoch - milestones[mid]};
                } else {
                    return helper(epoch, mid, max);
                }
            } else {
                return new int[]{mid + 1, epoch - milestones[mid]};
            }
        }
    }

    private int[] getIndexes(int epoch) {
        return helper(epoch, 0, milestones.length);
    }

    @Override
    public double call(int epoch) {
        int[] indexes;
        if (schedulers.length != 1) {
            indexes = getIndexes(epoch);
        } else {
            indexes = new int[]{0, epoch};
        }
        if (indexes[0] > 0 && indexes[1] == 0) {
            if (indexes[0] > 1) {
                schedulers[indexes[0]].setInitialLearningRate(schedulers[indexes[0] - 1].call(epoch - milestones[indexes[0] - 2] - 1));
            } else {
                schedulers[indexes[0]].setInitialLearningRate(schedulers[indexes[0] - 1].call(epoch - 1));
            }
        }
        return schedulers[indexes[0]].call(indexes[1]);
    }
}
