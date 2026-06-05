package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import ComputationalGraph.Node.MultiplicationNode;

import java.io.Serializable;

public class SiLU implements FunctionCombiner, Serializable {

    private final double beta;

    public SiLU(double beta) {
        this.beta = beta;
    }

    public SiLU() {
        this(1.0);
    }

    public ComputationalNode addEdge(ComputationalNode inputNode, boolean isBiased) {
        ComputationalNode sigmoid = new FunctionNode(false, new Sigmoid());
        if (beta != 1.0) {
            ComputationalNode multiplyWithBeta = new FunctionNode(false, new MultiplyByConstant(beta));
            inputNode.add(multiplyWithBeta);
            multiplyWithBeta.add(sigmoid);
        } else {
            inputNode.add(sigmoid);
        }
        ComputationalNode silu = new MultiplicationNode(false, isBiased, true);
        sigmoid.add(silu);
        inputNode.add(silu);
        return silu;
    }
}
