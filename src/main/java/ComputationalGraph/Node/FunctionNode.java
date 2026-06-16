package ComputationalGraph.Node;

import java.io.Serializable;

import ComputationalGraph.Function.*;
import Math.Tensor;

public class FunctionNode extends ComputationalNode implements Serializable {

    private final Function function;
    private Tensor context;

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

    public boolean isDropout() {
        return function instanceof Dropout;
    }

    public void calculate(Tensor input) {
        FunctionResults results = function.calculate(input);
        this.setValue(results.output());
        if (results.context() != null) {
            this.context = results.context();
        }
    }

    public Tensor derivative() {
        Tensor backward;
        if (isBiased) {
            backward = getBiasedPartial(this.backward);
        } else {
            backward = this.backward;
        }
        if (context != null) {
            return function.derivative(context, backward);
        }
        Tensor value;
        if (isBiased) {
            value = getBiasedPartial(this.value);
        } else {
            value = this.value;
        }
        return function.derivative(value, backward);
    }
}
