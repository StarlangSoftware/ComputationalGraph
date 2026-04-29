package ComputationalGraph.Function;

import Math.Tensor;

public interface Function {
    Tensor calculate(Tensor matrix);
    Tensor derivative(Tensor value, Tensor backward);
}
