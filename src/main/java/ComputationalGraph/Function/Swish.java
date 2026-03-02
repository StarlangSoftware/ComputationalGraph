package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import ComputationalGraph.Node.MultiplicationNode;

import java.io.Serializable;

public class Swish extends Sigmoid implements Serializable {

    public ComputationalNode addEdge(ComputationalNode input, boolean isBiased) {
        ComputationalNode sigmoid = new FunctionNode(false, this);
        input.add(sigmoid);
        ComputationalNode swish = new MultiplicationNode(false, isBiased, true);
        sigmoid.add(swish);
        input.add(swish);
        return swish;
    }
}
