package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import Math.*;

import java.util.ArrayList;

public interface Function {
    Tensor calculate(Tensor matrix);
    Tensor derivative(Tensor value, Tensor backward);
    ComputationalNode addEdge(ArrayList<ComputationalNode> inputNodes, boolean isBiased);
}
