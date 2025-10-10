import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.ComputationalGraph;

import ComputationalGraph.Function.ELU;
import ComputationalGraph.Function.Sigmoid;
import ComputationalGraph.Function.Softmax;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.MultiplicationNode;
import Math.*;
import ComputationalGraph.*;

import java.io.Serializable;
import java.util.*;

public class NeuralNet extends ComputationalGraph implements Serializable {

    private Tensor createInputTensor(Tensor instance) {
        ArrayList<Double> data = new ArrayList<>();
        for (int i = 0; i < instance.getShape()[0] - 1; i++) {
            data.add(instance.getValue(new int[]{i}));
        }
        return new Tensor(data, new int[]{1, instance.getShape()[0] - 1});
    }

    @Override
    public void train(ArrayList<Tensor> trainSet, NeuralNetworkParameter parameters) {
        // Input Node
        ComputationalNode input = new MultiplicationNode(false, true, false);
        inputNodes.add(input);
        // First layer weights
        Tensor t1 = new Tensor(parameters.getInitialization().initialize(5, 4, new Random(1)), new int[]{5, 4});
        ComputationalNode w1 = new MultiplicationNode(true, false, t1, false);
        ComputationalNode a1 = this.addEdge(input, w1, false);
        ComputationalNode a1Sigmoid = this.addEdge(a1, new Sigmoid(), true);
        // Second layer weights
        Tensor t2 = new Tensor(parameters.getInitialization().initialize(5, 20, new Random(1)), new int[]{5, 20});
        ComputationalNode w2 = new MultiplicationNode(true, false, t2, false);
        ComputationalNode a2 = this.addEdge(a1Sigmoid, w2, false);
        ComputationalNode a2ELU = this.addEdge(a2, new ELU(3.0), true);
        // Output layer weights
        Tensor t3 = new Tensor(parameters.getInitialization().initialize(21, 3, new Random(1)), new int[]{21, 3});
        ComputationalNode w3 = new MultiplicationNode(true, false, t3, false);
        ComputationalNode a3 = this.addEdge(a2ELU, w3, false);
        this.addEdge(a3, new Softmax(), false);
        // Training
        ArrayList<Integer> classList = new ArrayList<>();
        for (int i = 0; i < parameters.getEpoch(); i++) {
            // Shuffle
            Random random = new Random(parameters.getSeed());
            for (int j = 0; j < trainSet.size(); j++) {
                int i1 = random.nextInt(trainSet.size());
                int i2 = random.nextInt(trainSet.size());
                Tensor tmp = trainSet.get(i1);
                trainSet.set(i1, trainSet.get(i2));
                trainSet.set(i2, tmp);
            }
            for (Tensor instance : trainSet) {
                input.setValue(createInputTensor(instance));
                this.forwardCalculation();
                classList.add((int) instance.getValue(new int[]{instance.getShape()[0] - 1}));
                this.backpropagation(parameters.getOptimizer(), classList);
                classList.clear();
            }
            parameters.getOptimizer().setLearningRate();
        }
    }

    @Override
    public ClassificationPerformance test(ArrayList<Tensor> testSet) {
        int count = 0, total = 0;
        for (Tensor instance : testSet) {
            inputNodes.get(0).setValue(createInputTensor(instance));
            int classLabel = this.predict().get(0);
            if (classLabel == instance.getValue(new int[]{instance.getShape()[0] - 1})) {
                count++;
            }
            total++;
        }
        return new ClassificationPerformance((count + 0.00) / total);
    }

    @Override
    protected ArrayList<Integer> getClassLabels(ComputationalNode outputNode) {
        ArrayList<Integer> classLabelIndices = new ArrayList<>();
        Tensor outputValue = outputNode.getValue();
        if (outputValue != null) {
            int cols = outputValue.getShape()[1];
            double maxVal = Double.NEGATIVE_INFINITY;
            int labelIndex = -1;
            for (int j = 0; j < cols; j++) {
                double val = outputValue.getValue(new int[]{0, j});
                if (maxVal < val) {
                    maxVal = val;
                    labelIndex = j;
                }
            }
            classLabelIndices.add(labelIndex);
        }
        return classLabelIndices;
    }
}
