import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.ComputationalGraph;
import ComputationalGraph.Node.ComputationalNode;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import ComputationalGraph.Function.Softmax;
import ComputationalGraph.Node.MultiplicationNode;
import Math.*;
import ComputationalGraph.*;

public class LinearPerceptronSingleInput extends ComputationalGraph implements Serializable {

    private Tensor createInputTensor(Tensor instance) {
        ArrayList<Double> data = new ArrayList<>();
        for (int i = 0; i < instance.getShape()[0] - 1; i++) {
            data.add(instance.getValue(new int[]{i}));
        }
        return new Tensor(data, new int[]{1, instance.getShape()[0] - 1});
    }

    @Override
    public void train(ArrayList<Tensor> trainSet, NeuralNetworkParameter parameters) {
        ComputationalNode input = new MultiplicationNode(false, true, false);
        inputNodes.add(input);
        Tensor weightsTensor = new Tensor(Arrays.asList(1.0, 1.0, 1.0, 1.0), new int[]{2, 2});
        ComputationalNode w = new MultiplicationNode(true, false, weightsTensor, false);
        ComputationalNode a = this.addEdge(input, w, false);
        this.addEdge(a, new Softmax(), false);
        Tensor dataTensor = new Tensor(Arrays.asList(1.0, 1.0), new int[]{2});
        input.setValue(createInputTensor(dataTensor));
        this.forwardCalculation(false);
        ArrayList<Integer> classList = new ArrayList<>();
        classList.add((int) dataTensor.getValue(new int[]{dataTensor.getShape()[0] - 1}));
        this.backpropagation(parameters.getOptimizer(), classList);
    }

    @Override
    public ClassificationPerformance test(ArrayList<Tensor> testSet) {
        return null;
    }

    @Override
    protected ArrayList<Integer> getClassLabels(ComputationalNode outputNode) {
        return null;
    }
}
