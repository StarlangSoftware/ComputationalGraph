package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import ComputationalGraph.Node.MultiplicationNode;

import java.io.Serializable;
import java.util.ArrayList;

public class Swish extends Sigmoid implements Serializable {

    public ComputationalNode addEdge(ArrayList<ComputationalNode> inputNodes, boolean isBiased) {
        ComputationalNode sigmoid = new FunctionNode(false, this);
        inputNodes.get(0).add(sigmoid);
        ComputationalNode swish = new MultiplicationNode(false, isBiased, true);
        sigmoid.add(swish);
        inputNodes.get(0).add(swish);
        return swish;
    }
}
