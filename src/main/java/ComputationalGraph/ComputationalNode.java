package ComputationalGraph;

import Math.Tensor;

import java.io.Serializable;

public class ComputationalNode implements Serializable {
    protected Tensor value;
    protected Tensor backward;
    protected final boolean learnable;
    protected final boolean isBiased;
    protected final Function function;

    /**
     * Initializes a ComputationalNode.
     * @param learnable Indicates whether the node is learnable (e.g., weights)
     * @param isBiased Indicates whether the node is biased
     * @param function The function (e.g., activation like SIGMOID)
     * @param value The tensor value associated with the node (optional)
     */
    public ComputationalNode(boolean learnable, boolean isBiased, Function function, Tensor value) {
        this.value = value;
        this.backward = null;
        this.learnable = learnable;
        this.isBiased = isBiased;
        this.function = function;
    }

    /**
     * Constructor overload for function type initialization
     */
    public ComputationalNode(boolean learnable, Function function, boolean isBiased) {
        this(learnable, isBiased, function, null);
    }

    /**
     * Constructor overload for operator initialization
     */
    public ComputationalNode(boolean learnable, boolean isBiased) {
        this(learnable, isBiased, null, null);
    }

    @Override
    public String toString() {
        StringBuilder details = new StringBuilder();
        if (function != null) {
            details.append("Function: ").append(function);
        }
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

    public boolean isBiased() {
        return isBiased;
    }

    public Function getFunction() {
        return function;
    }

    public Tensor getValue() {
        return value;
    }

    public void setValue(Tensor value) {
        this.value = value;
    }

    public void updateValue() {
        if (value.getShape()[value.getShape().length - 1] + 1 == backward.getShape()[value.getShape().length - 1]) {
            int[] endIndexes = new int[backward.getShape().length];
            for (int i = 0; i < endIndexes.length; i++) {
                if (i == endIndexes.length - 1) {
                    endIndexes[i] = backward.getShape()[i] - 1;
                } else {
                    endIndexes[i] = backward.getShape()[i];
                }
            }
            Tensor partial = backward.partial(new int[backward.getShape().length], endIndexes);
            this.setValue(value.add(partial));
        } else {
            this.setValue(value.add(backward));
        }
    }

    public boolean isLearnable() {
        return learnable;
    }

    public Tensor getBackward() {
        return backward;
    }

    public void setBackward(Tensor backward) {
        this.backward = backward;
    }
}