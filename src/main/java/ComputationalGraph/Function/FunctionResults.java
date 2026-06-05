package ComputationalGraph.Function;

import Math.Tensor;
import java.io.Serializable;

public final class FunctionResults implements Serializable {

    private final Tensor output;
    private final Tensor context;

    public FunctionResults(Tensor output, Tensor context) {
        this.output = output;
        this.context = context;
    }

    public FunctionResults(Tensor output) {
        this(output, null);
    }

    public Tensor output() {
        return this.output;
    }

    public Tensor context() {
        return this.context;
    }
}