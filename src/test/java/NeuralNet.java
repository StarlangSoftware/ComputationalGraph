import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.ComputationalGraph;

import ComputationalGraph.Function.*;
import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.MultiplicationNode;
import Math.*;
import ComputationalGraph.*;

import java.io.Serializable;
import java.util.*;

public class NeuralNet extends ComputationalGraph implements Serializable {

    public NeuralNet(NeuralNetworkParameter parameters) {
        super(parameters);
        // Input Node
        ComputationalNode input = new MultiplicationNode();
        this.addInputNode(input);
        // First layer weights
        int numberOfInputUnitsWithBiased = 5;
        int numberOfHiddenUnitsInLayer1 = 4;
        Tensor t1 = new Tensor(parameters.initializeWeights(numberOfInputUnitsWithBiased, numberOfHiddenUnitsInLayer1, new Random(parameters.getSeed())), new int[]{numberOfInputUnitsWithBiased, 4});
        ComputationalNode w1 = new MultiplicationNode(t1);
        ComputationalNode a1 = this.addEdge(input, w1);
        ComputationalNode a1Sigmoid = this.addEdge(a1, new Sigmoid());
        ComputationalNode a1SigmoidDropout = this.addEdge(a1Sigmoid, new Dropout(parameters.getDropout(), new Random(parameters.getSeed())), true);
        // Second layer weights
        int numberOfHiddenUnitsInLayer2 = 20;
        Tensor t2 = new Tensor(parameters.initializeWeights(numberOfHiddenUnitsInLayer1 + 1, numberOfHiddenUnitsInLayer2, new Random(parameters.getSeed())), new int[]{numberOfHiddenUnitsInLayer1 + 1, numberOfHiddenUnitsInLayer2});
        ComputationalNode w2 = new MultiplicationNode(t2);
        ComputationalNode a2 = this.addEdge(a1SigmoidDropout, w2);
        ComputationalNode a2ELU = this.addEdge(a2, new ELU(3.0));
        ComputationalNode a2ELUDropout = this.addEdge(a2ELU, new Dropout(parameters.getDropout(), new Random(parameters.getSeed())), true);
        // Output layer weights
        int numberOfClasses = 3;
        Tensor t3 = new Tensor(parameters.initializeWeights(numberOfHiddenUnitsInLayer2 + 1, numberOfClasses, new Random(parameters.getSeed())), new int[]{21, 3});
        ComputationalNode w3 = new MultiplicationNode(t3);
        ComputationalNode a3 = this.addEdge(a2ELUDropout, w3);
        ComputationalNode classLabelNode = this.addLoss(this.addEdge(a3, new Softmax()));
        this.addInputNode(classLabelNode);
    }

    private Tensor createInputTensor(Tensor instance) {
        ArrayList<Double> data = new ArrayList<>();
        for (int i = 0; i < instance.getShape()[0] - 1; i++) {
            data.add(instance.getValue(new int[]{i}));
        }
        return new Tensor(data, new int[]{1, instance.getShape()[0] - 1});
    }

    private Tensor setClassLabelNode(int classLabel) {
        ArrayList<Double> data = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            if (i == classLabel) {
                data.add(1.0);
            } else {
                data.add(0.0);
            }
        }
        return new Tensor(data, new int[]{1, 3});
    }

    @Override
    public void train(ArrayList<Tensor> trainSet) {
        // Training
        for (int i = 0; i < parameters.getEpoch(); i++) {
            this.shuffle(trainSet, new Random(parameters.getSeed()));
            for (Tensor instance : trainSet) {
                this.getInputNode(0).setValue(createInputTensor(instance));
                this.getInputNode(1).setValue(setClassLabelNode((int) instance.getValue(new int[]{instance.getShape()[0] - 1})));
                this.forwardCalculation();
                this.backpropagation();
            }
            parameters.getOptimizer().setLearningRate();
        }
    }

    @Override
    public ClassificationPerformance test(ArrayList<Tensor> testSet) {
        int count = 0, total = 0;
        for (Tensor instance : testSet) {
            this.getInputNode(0).setValue(createInputTensor(instance));
            int classLabel = this.predict().get(0).intValue();
            if (classLabel == instance.getValue(new int[]{instance.getShape()[0] - 1})) {
                count++;
            }
            total++;
        }
        return new ClassificationPerformance((count + 0.00) / total);
    }

    @Override
    protected ArrayList<Double> getOutputValue(Tensor outputValue) {
        ArrayList<Double> classLabelIndices = new ArrayList<>();
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
            classLabelIndices.add(labelIndex + 0.0);
        }
        return classLabelIndices;
    }
}
