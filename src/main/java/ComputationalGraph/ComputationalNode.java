package ComputationalGraph;

import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

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
        this.setValue(value.add(backward));
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