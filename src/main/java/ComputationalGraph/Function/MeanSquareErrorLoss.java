package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;

import java.io.Serializable;
import java.util.ArrayList;

public class MeanSquareErrorLoss extends Negation implements Serializable {

    @Override
    public ComputationalNode addEdge(ArrayList<ComputationalNode> inputNodes, boolean isBiased) {
        ComputationalNode negatedY = new FunctionNode(false, this);
        inputNodes.get(0).add(negatedY);
        ComputationalNode yMinusNegatedY = new ComputationalNode(false, false);
        negatedY.add(yMinusNegatedY);
        inputNodes.get(1).add(yMinusNegatedY);
        ComputationalNode newNode = new FunctionNode(isBiased, new Power());
        yMinusNegatedY.add(newNode);
        return newNode;
    }
}
