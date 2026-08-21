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
        if (schedulers.length != 1 && milestones[0] < 1) {
            throw new IllegalArgumentException("First milestone must be bigger than 0.");
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
        for (int i = 1; i < schedulers.length; i++) {
            int previousLength;
            if (i == 1) {
                previousLength = milestones[0];
            } else {
                previousLength = milestones[i - 1] - milestones[i - 2];
            }
            schedulers[i].setInitialLearningRate(schedulers[i - 1].call(previousLength));
        }
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
        return schedulers[indexes[0]].call(indexes[1]);
    }
}
