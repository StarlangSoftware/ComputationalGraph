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
        List<Double> trainList = new ArrayList<>();
        List<Double> testList = new ArrayList<>();
        for (int i = 0; i < dataSet.size(); i++) {
            if (i >= 120) {
                for (int j = 0; j < dataSet.get(i).length - 1; j++) {
                    testList.add(Double.parseDouble(dataSet.get(i)[j]));
                }
                testList.add(labelMap.get(dataSet.get(i)[dataSet.get(i).length - 1]) + 0.00);
            } else {
                for (int j = 0; j < dataSet.get(i).length - 1; j++) {
                    trainList.add(Double.parseDouble(dataSet.get(i)[j]));
                }
                trainList.add(labelMap.get(dataSet.get(i)[dataSet.get(i).length - 1]) + 0.00);
            }
        }
        NeuralNet graph = new NeuralNet();
        graph.train(new Tensor(trainList, new int[]{120, 5}), new NeuralNetParameter(100, 0.99, 0.1, 1));
        ClassificationPerformance performance = graph.test(new  Tensor(testList, new int[]{30, 5}));
        System.out.println("Accuracy: " + performance.getAccuracy());
        assertEquals(1.0, performance.getAccuracy(), 0.01);
    }


    @Test
    public void test2() {
        ComputationalGraph graph = new ComputationalGraph() {
            @Override
            public void train(Tensor trainSet, Parameter parameters) {
                ComputationalNode a0 = new ComputationalNode(false, false, "+", null, null);
                ComputationalNode a1 = new ComputationalNode(true, false, "+", null, null);
                ComputationalNode a2 = this.addEdge(a0, a1, false);
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
                this.backpropagation(0.01, classList);
            }

            @Override
            public ClassificationPerformance test(Tensor testSet) {
                return null;
            }

            @Override
            protected ArrayList<Integer> getClassLabes(ComputationalNode outputNode) {
                return null;
            }
        };
        graph.train(null, null);
    }
}