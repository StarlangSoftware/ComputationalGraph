package ComputationalGraph.Loss;

import ComputationalGraph.Function.DivideByDimensionSize;
import ComputationalGraph.Function.Negation;
import ComputationalGraph.Function.Power;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;

import java.io.Serializable;

public class MeanSquaredErrorLoss implements Loss, Serializable {

    /**
     * Adds a loss computation node to the computational graph based on the mean-squared error loss function.
     * This loss is calculated as the squared difference between the predicted output and the actual class labels,
     * averaged over the batch dimension.
     * @param inputNode The input node representing the predicted output values.
     * @param classLabelNode The node representing the actual class label values.
     * @param batchDimension which dimension of the input tensor represents the batch size.
     * @return A computational node that represents the final loss value normalized by the batch size.
     */
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
