package ComputationalGraph.Loss;

import ComputationalGraph.Function.DivideByDimensionSize;
import ComputationalGraph.Function.Logarithm;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import ComputationalGraph.Node.MultiplicationNode;

import java.io.Serializable;

public class CrossEntropyLoss implements Loss, Serializable {

    /**
     * Adds a cross-entropy loss calculation to the computational graph.
     * This involves applying a logarithm to the predicted probabilities,
     * multiplying the result with the class label distributions,
     * and normalizing by the batch dimension size.
     * @param inputNode The computational node representing the predicted probabilities or model outputs.
     * @param classLabelNode The computational node representing the true class labels.
     * @param batchDimension which dimension of the input tensor represents the batch size.
     * @return A computational node representing the result of the cross-entropy loss computation.
     */
    @Override
    public ComputationalNode addLoss(ComputationalNode inputNode, ComputationalNode classLabelNode, int batchDimension) {
        ComputationalNode logy = new FunctionNode(false, new Logarithm());
        inputNode.add(logy);
        ComputationalNode ylogy = new MultiplicationNode(false, false, true);
        classLabelNode.add(ylogy);
        logy.add(ylogy);
        ComputationalNode ylogyDivideByDimension = new FunctionNode(false, new DivideByDimensionSize(batchDimension));
        ylogy.add(ylogyDivideByDimension);
        return ylogyDivideByDimension;
    }
}
