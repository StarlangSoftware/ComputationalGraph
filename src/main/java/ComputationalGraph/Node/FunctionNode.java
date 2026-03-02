package ComputationalGraph.Node;

import ComputationalGraph.Function.Function;

import java.io.Serializable;

public class FunctionNode extends ComputationalNode implements Serializable {

    private final Function function;

    public FunctionNode(boolean isBiased, Function function) {
        super(false, isBiased);
        this.function = function;
    }

    public FunctionNode(boolean learnable, boolean isBiased, Function function) {
        super(learnable, isBiased);
        this.function = function;
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
        details.append(", is biased: ").append(isBiased);
        return "FunctionNode(" + details + ")";
    }

    public Function getFunction() {
        return function;
    }
}
