import Classification.Parameter.Parameter;
import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.ComputationalGraph;

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
        return new Tensor(data, new int[]{1, 1, instance.getShape()[0] - 1});
    }

    @Override
    public void train(ArrayList<Tensor> trainSet, Parameter parameters) {
        // Input Node
        ComputationalNode input = new MultiplicationNode(false, true, false);
        inputNodes.add(input);
        // First layer weights
        ArrayList<Double> w1Data = new ArrayList<>();
        Random rand1 = new Random(1);
        for (int i = 0; i < 5 * 4; i++) {
            w1Data.add(-0.01 + (0.02 * rand1.nextDouble()));
        }
        Tensor t1 = new Tensor(w1Data, new int[]{1, 5, 4});
        ComputationalNode w1 = new MultiplicationNode(true, false, t1, false);
        ComputationalNode a1 = this.addEdge(input, w1, true);
        ComputationalNode a1TanH = this.addEdge(a1, new Tanh(), true);
        // Second layer weights
        ArrayList<Double> w2Data = new ArrayList<>();
        Random rand2 = new Random(1);
        for (int i = 0; i < 5 * 20; i++) {
            w2Data.add(-0.01 + (0.02 * rand2.nextDouble()));
        }
        Tensor t2 = new Tensor(w2Data, new int[]{1, 5, 20});
        ComputationalNode w2 = new MultiplicationNode(true, false, t2, false);
        ComputationalNode a2 = this.addEdge(a1TanH, w2, true);
        ComputationalNode a2Sigmoid = this.addEdge(a2, new Sigmoid(), true);
        // Output layer weights
        ArrayList<Double> w3Data = new ArrayList<>();
        Random rand3 = new Random(1);
        for (int i = 0; i < 21 * 3; i++) {
            w3Data.add(-0.01 + (0.02 * rand3.nextDouble()));
        }
        Tensor t3 = new Tensor(w3Data, new int[]{1, 21, 3});
        ComputationalNode w3 = new MultiplicationNode(true, false, t3, false);
        ComputationalNode a3 = this.addEdge(a2Sigmoid, w3, false);
        this.addEdge(a3, new Softmax(), false);
        // Training
        ArrayList<Integer> classList = new ArrayList<>();
        for (int i = 0; i < ((NeuralNetParameter) parameters).getEpoch(); i++) {
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
                this.backpropagation(((NeuralNetParameter) parameters).getLearningRate(), classList);
                classList.clear();
            }
            ((NeuralNetParameter) parameters).setLearningRate();
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
            int cols = outputValue.getShape()[2];
            double maxVal = Double.NEGATIVE_INFINITY;
            int labelIndex = -1;
            for (int j = 0; j < cols; j++) {
                double val = outputValue.getValue(new int[]{0, 0, j});
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
