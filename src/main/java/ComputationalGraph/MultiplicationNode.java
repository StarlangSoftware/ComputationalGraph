package ComputationalGraph;

import java.io.Serializable;
import Math.Tensor;

public class MultiplicationNode extends ComputationalNode implements Serializable {

    private final boolean isHadamard;

    public MultiplicationNode(boolean learnable, boolean isBiased, boolean isHadamard) {
        super(learnable, isBiased);
        this.isHadamard = isHadamard;
    }

    public MultiplicationNode(boolean learnable, boolean isBiased, Tensor value, boolean isHadamard) {
        super(learnable, isBiased);
        this.isHadamard = isHadamard;
        this.value = value;
    }

    public boolean isHadamard() {
        return isHadamard;
    }
}
