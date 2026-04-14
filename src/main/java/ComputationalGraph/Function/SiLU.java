package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import ComputationalGraph.Node.MultiplicationNode;

import java.io.Serializable;

public class SiLU implements CompositeFunction, Serializable {

    public ComputationalNode addEdge(ComputationalNode inputNode, boolean isBiased) {
        ComputationalNode sigmoid = new FunctionNode(false, new Sigmoid());
        inputNode.add(sigmoid);
        ComputationalNode silu = new MultiplicationNode(false, isBiased, true);
        sigmoid.add(silu);
        inputNode.add(silu);
        return silu;
    }
}
