package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import ComputationalGraph.Node.MultiplicationNode;

import java.io.Serializable;

public class SiLU extends Sigmoid implements Serializable {

    public ComputationalNode addEdge(ComputationalNode inputNode, boolean isBiased) {
        ComputationalNode sigmoid = new FunctionNode(false, this);
        inputNode.add(sigmoid);
        ComputationalNode swish = new MultiplicationNode(false, isBiased, true);
        sigmoid.add(swish);
        inputNode.add(swish);
        return swish;
    }
}
