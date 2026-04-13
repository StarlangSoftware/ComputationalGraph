package ComputationalGraph.Loss;

import ComputationalGraph.Function.DivideByDimensionSize;
import ComputationalGraph.Function.Logarithm;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import ComputationalGraph.Node.MultiplicationNode;

import java.io.Serializable;

public class CrossEntropyLoss implements Loss, Serializable {

    @Override
    public ComputationalNode addEdge(ComputationalNode inputNode, ComputationalNode classLabelNode, int batchDimension) {
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
