package ComputationalGraph.Node;

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

    public MultiplicationNode(boolean learnable, Tensor value) {
        super(learnable, false, value);
        this.value = value;
        this.priorityNode = null;
        this.isHadamard = false;
    }

    public MultiplicationNode(Tensor value) {
        super(true, false);
        this.value = value;
        this.priorityNode = null;
        this.isHadamard = false;
    }

    public MultiplicationNode(boolean learnable, boolean isBiased) {
        super(learnable, isBiased);
        this.isHadamard = false;
        this.priorityNode = null;
    }

    @Override
    public String toString() {
        StringBuilder details = new StringBuilder();
        if (value != null) {
            if (details.length() > 0) {
                details.append(", ");
            }
            details.append("Value Shape: [").append(value.getShape()[0]);
            for (int i = 1; i < value.getShape().length; i++) {
                details.append(", ").append(value.getShape()[i]);
            }
            details.append("]");
        }
        if (details.length() > 0) details.append(", ");
        details.append("is learnable: ").append(learnable);
        details.append(", is biased: ").append(isBiased);
        return "Node(" + details + ")";
    }

    public boolean isHadamard() {
        return isHadamard;
    }

    public ComputationalNode getPriorityNode() {
        return priorityNode;
    }
}
