package ComputationalGraph.Loss;

import ComputationalGraph.Function.DivideByDimensionSize;
import ComputationalGraph.Function.Negation;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;

import java.io.Serializable;

public class MeanBiasError implements Loss, Serializable {

    /**
     * Adds a loss computation node to the computational graph based on input and class label nodes.
     * The method incorporates operations including negation, subtraction, and scaling by the batch dimension.
     * @param inputNode The computational node representing the model's predictions.
     * @param classLabelNode The computational node representing the true class labels.
     * @param batchDimension The size of the batch, used for scaling the loss value.
     * @return The computational node that represents the resultant loss value divided by the batch dimension.
     */
    @Override
    public ComputationalNode addLoss(ComputationalNode inputNode, ComputationalNode classLabelNode, int batchDimension) {
        ComputationalNode negatedY = new FunctionNode(false, new Negation());
        inputNode.add(negatedY);
        ComputationalNode yMinusNegatedY = new ComputationalNode();
        negatedY.add(yMinusNegatedY);
        classLabelNode.add(yMinusNegatedY);
        ComputationalNode newNodeDivideByDimension = new FunctionNode(false, new DivideByDimensionSize(batchDimension));
        yMinusNegatedY.add(newNodeDivideByDimension);
        return newNodeDivideByDimension;
    }
}
