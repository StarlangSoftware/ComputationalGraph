package ComputationalGraph;

import Math.*;

public interface Function {
    Tensor calculate(Tensor matrix);
    Tensor derivative(Tensor value, Tensor backward);
}
