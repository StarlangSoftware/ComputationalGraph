import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.ComputationalGraph;
import ComputationalGraph.Loss.Loss;
import ComputationalGraph.Node.ComputationalNode;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

import ComputationalGraph.Function.Softmax;
import ComputationalGraph.Node.MultiplicationNode;
import Math.*;
import ComputationalGraph.*;

public class LinearPerceptronSingleInput extends ComputationalGraph implements Serializable {

    public LinearPerceptronSingleInput(NeuralNetworkParameter parameters) {
        super(parameters);
    }

    private Tensor createInputTensor(Tensor instance) {
        ArrayList<Double> data = new ArrayList<>();
        for (int i = 0; i < instance.getShape()[0] - 1; i++) {
            data.add(instance.getValue(new int[]{i}));
        }
        return new Tensor(data, new int[]{1, instance.getShape()[0] - 1});
    }

    @Override
    public void train(ArrayList<Tensor> trainSet) {
        ComputationalNode input = new MultiplicationNode(false, true, false);
        inputNodes.add(input);
        Tensor weightsTensor = new Tensor(Arrays.asList(1.0, 1.0, 1.0, 1.0), new int[]{2, 2});
        ComputationalNode w = new MultiplicationNode(weightsTensor);
        ComputationalNode a = this.addEdge(input, w, false);
        this.outputNode = this.addEdge(a, new Softmax(), false);
        Tensor dataTensor = new Tensor(Arrays.asList(1.0, 1.0), new int[]{2});
        input.setValue(createInputTensor(dataTensor));
        Loss dummyLoss = (inputNode, classNode, d) -> inputNode;
        this.addLoss(outputNode, null, dummyLoss);
        this.forwardCalculation();
        ArrayList<Integer> classList = new ArrayList<>();
        classList.add((int) dataTensor.getValue(new int[]{dataTensor.getShape()[0] - 1}));
        this.backpropagation();
    }

    @Override
    public ClassificationPerformance test(ArrayList<Tensor> testSet) {
        return null;
    }

    @Override
    protected ArrayList<Double> getOutputValue(ComputationalNode outputNode) {
        return null;
    }
}
