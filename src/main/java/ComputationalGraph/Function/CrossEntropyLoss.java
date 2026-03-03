package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import ComputationalGraph.Node.MultiplicationNode;

import java.io.Serializable;
import java.util.ArrayList;

public class CrossEntropyLoss extends Logarithm implements Serializable {

    @Override
    public ComputationalNode addEdge(ArrayList<ComputationalNode> inputNodes, boolean isBiased) {
        ComputationalNode logy = new FunctionNode(false, this);
        inputNodes.get(0).add(logy);
        ComputationalNode ylogy = new MultiplicationNode(false, isBiased, true);
        inputNodes.get(1).add(ylogy);
        logy.add(ylogy);
        return ylogy;
    }
}
