package ComputationalGraph.Loss;

import ComputationalGraph.Function.DivideByDimensionSize;
import ComputationalGraph.Function.Negation;
import ComputationalGraph.Function.Power;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;

import java.io.Serializable;

public class MeanSquaredErrorLoss implements Loss, Serializable {

    @Override
    public ComputationalNode addLoss(ComputationalNode inputNode, ComputationalNode classLabelNode, int batchDimension) {
        ComputationalNode negatedY = new FunctionNode(false, new Negation());
        inputNode.add(negatedY);
        ComputationalNode yMinusNegatedY = new ComputationalNode();
        negatedY.add(yMinusNegatedY);
        classLabelNode.add(yMinusNegatedY);
        ComputationalNode newNode = new FunctionNode(false, new Power());
        yMinusNegatedY.add(newNode);
        ComputationalNode newNodeDivideByDimension = new FunctionNode(false, new DivideByDimensionSize(batchDimension));
        newNode.add(newNodeDivideByDimension);
        return newNodeDivideByDimension;
    }
}
