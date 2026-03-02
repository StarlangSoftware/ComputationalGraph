package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import Math.*;

public interface Function {
    Tensor calculate(Tensor matrix);
    Tensor derivative(Tensor value, Tensor backward);
    ComputationalNode addEdge(ComputationalNode input, boolean isBiased);
}
