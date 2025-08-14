package ComputationalGraph;

import java.io.Serializable;
import Math.Tensor;

public class MultiplicationNode extends ComputationalNode implements Serializable {

    private final boolean isHadamard;
    private final ComputationalNode priorityNode;

    public MultiplicationNode(boolean learnable, boolean isBiased, boolean isHadamard) {
        super(learnable, isBiased);
        this.isHadamard = isHadamard;
        this.priorityNode = null;
    }

    public MultiplicationNode(boolean learnable, boolean isBiased, boolean isHadamard, ComputationalNode priorityNode) {
        super(learnable, isBiased);
        this.isHadamard = isHadamard;
        this.priorityNode = priorityNode;
    }

    public MultiplicationNode(boolean learnable, boolean isBiased, Tensor value, boolean isHadamard) {
        super(learnable, isBiased);
        this.isHadamard = isHadamard;
        this.value = value;
        this.priorityNode = null;
    }

    public boolean isHadamard() {
        return isHadamard;
    }

    public ComputationalNode getPriorityNode() {
        return priorityNode;
    }
}
