import Classification.Parameter.Parameter;
import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.*;
import org.junit.Test;
import Math.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

import static org.junit.Assert.*;

public class ComputationalGraphTest {

    @Test
    public void test1() throws FileNotFoundException {
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
        NeuralNet graph = new NeuralNet();
        graph.train(trainList, new NeuralNetworkParameter(1, 100, new StochasticGradientDescent(0.1, 0.99)));
        ClassificationPerformance performance = graph.test(testList);
        System.out.println("Accuracy: " + performance.getAccuracy());
        assertEquals(1.0, performance.getAccuracy(), 0.01);
    }


    @Test
    public void test2() {
        ComputationalGraph graph = new ComputationalGraph() {
            @Override
            public void train(ArrayList<Tensor> trainSet, NeuralNetworkParameter parameters) {
                ComputationalNode a0 = new ComputationalNode(false, null, false);
                ComputationalNode a1 = new ComputationalNode(true, null, false);
                ComputationalNode a2 = this.addAdditionEdge(a0, a1, false);
                this.addEdge(a2, new Softmax(), false);
                List<Double> data = new ArrayList<>();
                Random rand = new Random(1);
                for (int i = 0; i < 3; i++) {
                    data.add(rand.nextDouble() * 100);
                }
                a0.setValue(new Tensor(data, new int[]{1, 3}));
                a1.setValue(new Tensor(data, new int[]{1, 3}));
                ArrayList<Integer> classList = new ArrayList<>();
                classList.add(1);
                this.forwardCalculation();
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
        };
        graph.train(null, new NeuralNetworkParameter(1, 100, new StochasticGradientDescent(0.1, 0.99)));
    }

    @Test
    public void test3() {
        ComputationalGraph graph = new ComputationalGraph() {
            @Override
            public void train(ArrayList<Tensor> trainSet, NeuralNetworkParameter parameters) {
                ArrayList<ComputationalNode> nodes = new ArrayList<>();
                ComputationalNode input = new MultiplicationNode(false, false, false);
                inputNodes.add(input);
                ArrayList<Double> w1Data = new ArrayList<>();
                Random rand1 = new Random(1);
                for (int i = 0; i < 5 * 4; i++) {
                    w1Data.add(-0.01 + (0.02 * rand1.nextDouble()));
                }
                Tensor t1 = new Tensor(w1Data, new int[]{1, 5, 4});
                ComputationalNode w1 = new MultiplicationNode(true, false, t1, false);
                ComputationalNode a1 = this.addEdge(input, w1, false);
                ComputationalNode a1TanH = this.addEdge(a1, new Tanh(), false);
                nodes.add(a1TanH);
                ComputationalNode x1 = new MultiplicationNode(false, false, false);
                ArrayList<Double> v1Data = new ArrayList<>();
                for (int i = 0; i < 5 * 4; i++) {
                    v1Data.add(-0.01 + (0.02 * rand1.nextDouble()));
                }
                Tensor k1 = new Tensor(v1Data, new int[]{1, 5, 4});
                ComputationalNode v1 = new MultiplicationNode(true, false, k1, false);
                ComputationalNode m1 = this.addEdge(x1, v1, false);
                ComputationalNode m1TanH = this.addEdge(m1, new Tanh(), false);
                nodes.add(m1TanH);
                ConcatenatedNode concatenatedNode = (ConcatenatedNode) this.concatEdges(nodes, 2);
                ArrayList<Double> w3Data = new ArrayList<>();
                for (int i = 0; i < 8 * 3; i++) {
                    w3Data.add(-0.01 + (0.02 * rand1.nextDouble()));
                }
                Tensor t3 = new Tensor(w3Data, new int[]{1, 8, 3});
                ComputationalNode w3 = new MultiplicationNode(true, false, t3, false);
                ComputationalNode a3 = this.addEdge(concatenatedNode, w3, false);
                this.addEdge(a3, new Softmax(), false);
                for (int j = 0; j < 10; j++) {
                    ArrayList<Double> data = new ArrayList<>();
                    for (int i = 0; i < 5; i++) {
                        data.add(-0.01 + (0.02 * rand1.nextDouble()));
                    }
                    input.setValue(new Tensor(data, new int[]{1, 1, 5}));
                    data = new ArrayList<>();
                    for (int i = 0; i < 5; i++) {
                        data.add(-0.01 + (0.02 * rand1.nextDouble()));
                    }
                    x1.setValue(new Tensor(data, new int[]{1, 1, 5}));
                    ArrayList<Integer> classList = new ArrayList<>();
                    classList.add(1);
                    this.forwardCalculation();
                    this.backpropagation(parameters.getOptimizer(), classList);
                }
            }

            @Override
            public ClassificationPerformance test(ArrayList<Tensor> testSet) {
                return null;
            }

            @Override
            protected ArrayList<Integer> getClassLabels(ComputationalNode outputNode) {
                return null;
            }
        };
        graph.train(null, new NeuralNetworkParameter(1, 100, new StochasticGradientDescent(0.1, 0.99)));
    }

    @Test
    public void test4() {
        ComputationalGraph graph = new ComputationalGraph() {
            @Override
            public void train(ArrayList<Tensor> trainSet, NeuralNetworkParameter parameters) {
                ComputationalNode input = new MultiplicationNode(false, false, false);
                inputNodes.add(input);
                ArrayList<Double> w1Data = new ArrayList<>();
                Random rand1 = new Random(1);
                for (int i = 0; i < 5 * 4; i++) {
                    w1Data.add(-0.01 + (0.02 * rand1.nextDouble()));
                }
                Tensor t1 = new Tensor(w1Data, new int[]{1, 5, 4});
                ComputationalNode w1 = new MultiplicationNode(true, false, t1, false);
                ComputationalNode a1 = this.addEdge(input, w1, false);
                ComputationalNode a1TanH = this.addEdge(a1, new Tanh(), false);
                ArrayList<Double> w2Data = new ArrayList<>();
                rand1 = new Random(1);
                for (int i = 0; i < 4; i++) {
                    w2Data.add(-0.01 + (0.02 * rand1.nextDouble()));
                }
                Tensor t2 = new Tensor(w2Data, new int[]{1, 1, 4});
                ComputationalNode w2 = new MultiplicationNode(true, false, t2, true);
                ComputationalNode a2 = this.addEdge(a1TanH, w2, false);
                this.addEdge(a2, new Softmax(), false);
                ArrayList<Double> data = new ArrayList<>();
                for (int i = 0; i < 5; i++) {
                    data.add(-0.01 + (0.02 * rand1.nextDouble()));
                }
                input.setValue(new Tensor(data, new int[]{1, 1, 5}));
                ArrayList<Integer> classList = new ArrayList<>();
                classList.add(1);
                this.forwardCalculation();
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
        };
        graph.train(null, new NeuralNetworkParameter(1, 100, new StochasticGradientDescent(0.1, 0.99)));
    }

    @Test
    public void test5() {
        ComputationalNode node = new MultiplicationNode(false, false, new Tensor(Arrays.asList(1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0), new int[]{1, 3, 4}), false);
        node.setBackward(new Tensor(Arrays.asList(1.0, 2.0, 23.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 12.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 15.0, 6.0, 7.0, 8.0), new int[]{2, 3, 4}));
        node.updateValue();
    }
}