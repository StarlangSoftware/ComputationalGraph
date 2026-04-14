package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;

import java.io.Serializable;

public class TanhShrink implements CompositeFunction, Serializable {

    public ComputationalNode addEdge(ComputationalNode inputNode, boolean isBiased) {
        ComputationalNode tanh = new FunctionNode(false, new Tanh());
        inputNode.add(tanh);
        ComputationalNode negativeTanh = new FunctionNode(false, new Negation());
        tanh.add(negativeTanh);
        ComputationalNode tanhShrink = new ComputationalNode(false, isBiased);
        inputNode.add(tanhShrink);
        negativeTanh.add(tanhShrink);
        return tanhShrink;
    }
}
