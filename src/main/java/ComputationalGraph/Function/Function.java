package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import Math.*;

public interface Function {

    Tensor calculate(Tensor matrix);
    Tensor derivative(Tensor value, Tensor backward);

    default ComputationalNode addEdge(ComputationalNode inputNode, boolean isBiased) {
        ComputationalNode newNode = new FunctionNode(isBiased, this);
        inputNode.add(newNode);
        return newNode;
    }
}
