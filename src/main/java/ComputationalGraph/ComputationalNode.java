package ComputationalGraph;

import Math.Tensor;

public class ComputationalNode {
    private Tensor value;
    private Tensor backward;
    private final boolean learnable;
    private final boolean isBiased;
    private final String operator;
    private final Function function;

    /**
     * Initializes a ComputationalNode.
     * @param learnable Indicates whether the node is learnable (e.g., weights)
     * @param isBiased Indicates whether the node is biased
     * @param operator Operator (e.g., '*', '+') for the node
     * @param function The function (e.g., activation like SIGMOID)
     * @param value The tensor value associated with the node (optional)
     */
    public ComputationalNode(boolean learnable, boolean isBiased, String operator, Function function, Tensor value) {
        this.value = value;
        this.backward = null;
        this.learnable = learnable;
        this.isBiased = isBiased;
        this.operator = operator;
        this.function = function;
    }

    /**
     * Constructor overload for function type initialization
     */
    public ComputationalNode(boolean learnable, Function function, boolean isBiased) {
        this(learnable, isBiased, null, function, null);
    }

    /**
     * Constructor overload for operator initialization
     */
    public ComputationalNode(boolean learnable, String operator, boolean isBiased) {
        this(learnable, isBiased, operator, null, null);
    }

    @Override
    public String toString() {
        StringBuilder details = new StringBuilder();
        if (function != null) {
            details.append("Function: ").append(function);
        }
        if (operator != null) {
            if (details.length() > 0) details.append(", ");
            details.append("Operator: ").append(operator);
        }
        if (value != null) {
            if (details.length() > 0) details.append(", ");
            details.append("Value Shape: [").append(value.getShape()[0])
                  .append(", ").append(value.getShape()[1]).append("]");
        }
        if (details.length() > 0) details.append(", ");
        details.append("is learnable: ").append(learnable);
        details.append(", is biased: ").append(isBiased);
        
        return "Node(" + details.toString() + ")";
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
        if (value != null && backward != null) {
            for (int i = 0; i < value.getShape()[0]; i++) {
                for (int j = 0; j < value.getShape()[1]; j++) {
                    value.set(new int[]{i, j}, value.getValue(new int[]{i, j}) + backward.getValue(new int[]{i, j}));
                }
            }
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