package ComputationalGraph.Function;

import Math.Tensor;

public interface Function {
    FunctionResults calculate(Tensor matrix);
    Tensor derivative(Tensor value, Tensor backward);
}
