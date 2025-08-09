package ComputationalGraph;

import Math.Tensor;

import java.io.Serializable;

public class ComputationalNode implements Serializable {
    private Tensor value;
    private Tensor backward;
    private final boolean learnable;
    private final boolean isBiased;
    private final String operator;
    private final Function function;
    private final boolean isConcatenatedNode;
    private final boolean isHadamard;

    /**
     * Initializes a ComputationalNode.
     * @param learnable Indicates whether the node is learnable (e.g., weights)
     * @param isBiased Indicates whether the node is biased
     * @param operator Operator (e.g., '*', '+') for the node
     * @param function The function (e.g., activation like SIGMOID)
     * @param value The tensor value associated with the node (optional)
     */
    public ComputationalNode(boolean learnable, boolean isBiased, String operator, Function function, Tensor value, boolean isConcatenatedNode, boolean isHadamard) {
        this.value = value;
        this.backward = null;
        this.learnable = learnable;
        this.isBiased = isBiased;
        this.operator = operator;
        this.function = function;
        this.isConcatenatedNode = isConcatenatedNode;
        this.isHadamard = isHadamard;
    }

    /**
     * Constructor overload for function type initialization
     */
    public ComputationalNode(boolean learnable, Function function, boolean isBiased) {
        this(learnable, isBiased, null, function, null, false, false);
    }

    /**
     * Constructor overload for operator initialization
     */
    public ComputationalNode(boolean learnable, String operator, boolean isBiased, boolean isHadamard) {
        this(learnable, isBiased, operator, null, null, false, isHadamard);
    }

    public ComputationalNode() {
        this(false, false, null, null, null, true, false);
    }

    @Override
    public String toString() {
        StringBuilder details = new StringBuilder();
        if (function != null) {
            details.append("Function: ").append(function);
        }
        if (operator != null) {
            if (details.length() > 0) {
                details.append(", ");
            }
            details.append("Operator: ").append(operator);
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
        details.append(", is concatenated: ").append(isConcatenatedNode);
        return "Node(" + details + ")";
    }

    public boolean isBiased() {
        return isBiased;
    }

    public Function getFunction() {
        return function;
    }

    public String getOperator() {
        return operator;
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

    public boolean isConcatenatedNode() {
        return isConcatenatedNode;
    }

    public boolean isHadamard() {
        return isHadamard;
    }
}