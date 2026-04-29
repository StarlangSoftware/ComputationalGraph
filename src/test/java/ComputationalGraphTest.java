import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.*;
import ComputationalGraph.Loss.CrossEntropyLoss;
import ComputationalGraph.Loss.Loss;
import ComputationalGraph.Node.*;
import ComputationalGraph.Optimizer.*;
import org.junit.Test;
import Math.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

import static org.junit.Assert.*;

public class ComputationalGraphTest {

    @Test
    public void testLinearPerceptronSingleInput() {
        Loss dummyLoss = (inputNode, classNode, d) -> inputNode;
        LinearPerceptronSingleInput graph = new LinearPerceptronSingleInput(new NeuralNetworkParameter(1, 100, new StochasticGradientDescent(0.1, 0.99), dummyLoss, 0));
        graph.train(new ArrayList<>());
    }

    @Test
    public void testNeuralNet() throws FileNotFoundException {
        HashMap<String, Integer> labelMap = new HashMap<>();
        ArrayList<String[]> dataSet = new ArrayList<>();
        Scanner source = new Scanner(new File("iris.txt"));
        while (source.hasNextLine()) {
            String[] instance = source.nextLine().split(",");
            dataSet.add(instance);
            if (!labelMap.containsKey(instance[instance.length - 1])) {
                labelMap.put(instance[instance.length - 1], labelMap.size());
            }
        }
        source.close();
        Collections.shuffle(dataSet, new Random(1));
        ArrayList<Tensor> trainList = new ArrayList<>();
        ArrayList<Tensor> testList = new ArrayList<>();
        for (int i = 0; i < dataSet.size(); i++) {
            ArrayList<Double> values = new ArrayList<>();
            if (i >= 120) {
                for (int j = 0; j < dataSet.get(i).length - 1; j++) {
                    values.add(Double.parseDouble(dataSet.get(i)[j]));
                }
                values.add(labelMap.get(dataSet.get(i)[dataSet.get(i).length - 1]) + 0.00);
                testList.add(new Tensor(values, new int[]{values.size()}));
            } else {
                for (int j = 0; j < dataSet.get(i).length - 1; j++) {
                    values.add(Double.parseDouble(dataSet.get(i)[j]));
                }
                values.add(labelMap.get(dataSet.get(i)[dataSet.get(i).length - 1]) + 0.00);
                trainList.add(new Tensor(values, new int[]{values.size()}));
            }
        }
        NeuralNet graph = new NeuralNet(new NeuralNetworkParameter(1, 4, new AdamW(0.002, 0.99, 0.9, 0.999, 1e-10, 0.5), new CrossEntropyLoss(), 0));
        graph.train(trainList);
        ClassificationPerformance performance = graph.test(testList);
        System.out.println("Accuracy: " + performance.getAccuracy());
        assertEquals(1.0, performance.getAccuracy(), 0.01);
    }

    @Test
    public void testFeatures() {
        Loss dummyLoss = (inputNode, classNode, d) -> inputNode;
        ComputationalGraph graph = new ComputationalGraph(new NeuralNetworkParameter(1, 1, new StochasticGradientDescent(0.1, 0.99), dummyLoss, 0)) {
            @Override
            public void train(ArrayList<Tensor> trainSet) {
                ComputationalNode input = new MultiplicationNode(false, false);
                inputNodes.add(input);
                input.setValue(new Tensor(Arrays.asList(1.0, 2.0, 3.0, 4.0), new int[]{2, 1, 2}));
                ArrayList<ComputationalNode> nodes = new ArrayList<>();
                for (int i = 0; i < 4; i++) {
                    MultiplicationNode w = new MultiplicationNode(new Tensor(Arrays.asList(6.0, 5.0, 4.0, 3.0, 2.0, 1.0), new int[]{1, 2, 3}));
                    nodes.add(this.addEdge(input, w));
                }
                ComputationalNode c = this.concatEdges(nodes, 1);
                MultiplicationNode w = new MultiplicationNode(new Tensor(Arrays.asList(6.0, 5.0, 1.0), new int[]{1, 3, 1}));
                this.outputNode = this.addEdge(c, w);
                this.addLoss(null);
                this.forwardCalculation();
                this.backpropagation();
                input.setValue(new Tensor(Arrays.asList(4.0, 3.0, 2.0, 1.0), new int[]{2, 1, 2}));
                this.forwardCalculation();
                ArrayList<Double> output = (ArrayList<Double>) this.outputNode.getValue().getData();
                ArrayList<Double> expected = new ArrayList<>();
                expected.add(2202.44);
                expected.add(2202.44);
                expected.add(2202.44);
                expected.add(2202.44);
                expected.add(973.6400000000001);
                expected.add(973.6400000000001);
                expected.add(973.6400000000001);
                expected.add(973.6400000000001);
                assertEquals(expected, output);
            }

            @Override
            public ClassificationPerformance test(ArrayList<Tensor> testSet) {
                return null;
            }

            @Override
            protected ArrayList<Double> getOutputValue() {
                return null;
            }
        };
        graph.train(null);
    }
}