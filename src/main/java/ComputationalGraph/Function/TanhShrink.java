package ComputationalGraph.Function;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;

import java.io.Serializable;
import java.util.ArrayList;

public class TanhShrink extends Tanh implements Serializable {

    public ComputationalNode addEdge(ArrayList<ComputationalNode> inputNodes, boolean isBiased) {
        ComputationalNode tanh = new FunctionNode(false, this);
        inputNodes.get(0).add(tanh);
        ComputationalNode negativeTanh = new FunctionNode(false, new Negation());
        tanh.add(negativeTanh);
        ComputationalNode tanhShrink = new ComputationalNode(false, isBiased);
        inputNodes.get(0).add(tanhShrink);
        negativeTanh.add(tanhShrink);
        return tanhShrink;
    }
}
