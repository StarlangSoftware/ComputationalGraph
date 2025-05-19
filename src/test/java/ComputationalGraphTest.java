import ComputationalGraph.*;
import org.junit.Test;
import Math.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

import static org.junit.Assert.*;

public class ComputationalGraphTest {

    private Tensor createInputTensor(String[] instance) {
        List<Double> data = new ArrayList<>();
        for (int i = 0; i < instance.length - 1; i++) {
            data.add(Double.parseDouble(instance[i]));
        }
        return new Tensor(data, new int[]{1, instance.length - 1});
    }

    @Test
    public void test1() throws FileNotFoundException {
        HashMap<String, Integer> labelMap = new HashMap<>();
        ArrayList<String[]> instances = new ArrayList<>();
        ArrayList<String[]> testSet = new ArrayList<>();
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
        for (int i = 0; i < dataSet.size(); i++) {
            if (i >= 120) {
                testSet.add(dataSet.get(i));
            } else {
                instances.add(dataSet.get(i));
            }
        }

        ComputationalGraph graph = new ComputationalGraph();

        // Input Node
        ComputationalNode input = new ComputationalNode(false, "*", true);

        // First layer weights
        List<Double> w1Data = new ArrayList<>();
        Random rand1 = new Random(1);
        for (int i = 0; i < 5 * 4; i++) {
            w1Data.add(-0.01 + (0.02 * rand1.nextDouble()));
        }
        Tensor t1 = new Tensor(w1Data, new int[]{5, 4});
        ComputationalNode w1 = new ComputationalNode(true, false, "*", null, t1);
        ComputationalNode a1 = graph.addEdge(input, w1, true);
        ComputationalNode a1Sigmoid = graph.addEdge(a1, new Sigmoid(), true);

        // Second layer weights
        List<Double> w2Data = new ArrayList<>();
        Random rand2 = new Random(1);
        for (int i = 0; i < 5 * 20; i++) {
            w2Data.add(-0.01 + (0.02 * rand2.nextDouble()));
        }
        Tensor t2 = new Tensor(w2Data, new int[]{5, 20});
        ComputationalNode w2 = new ComputationalNode(true, false, "*", null, t2);
        ComputationalNode a2 = graph.addEdge(a1Sigmoid, w2, true);
        ComputationalNode a2Sigmoid = graph.addEdge(a2, new Sigmoid(), true);

        // Output layer weights
        List<Double> w3Data = new ArrayList<>();
        Random rand3 = new Random(1);
        for (int i = 0; i < 21 * labelMap.size(); i++) {
            w3Data.add(-0.01 + (0.02 * rand3.nextDouble()));
        }
        Tensor t3 = new Tensor(w3Data, new int[]{21, labelMap.size()});
        ComputationalNode w3 = new ComputationalNode(true, false, "*", null, t3);
        ComputationalNode a3 = graph.addEdge(a2Sigmoid, w3, false);
        graph.addEdge(a3, new Softmax(), false);

        // Training
        int epoch = 1000;
        double etaDecrease = 0.99;
        double learningRate = 0.1;
        ArrayList<Integer> classList = new ArrayList<>();

        for (int i = 0; i < epoch; i++) {
            Collections.shuffle(instances, new Random(1));
            for (String[] instance : instances) {
                input.setValue(createInputTensor(instance));
                graph.forwardCalculation();
                classList.add(labelMap.get(instance[instance.length - 1]));
                graph.backpropagation(learningRate, classList);
                classList.clear();
            }
            learningRate *= etaDecrease;
        }

        // Testing
        int count = 0;
        for (String[] instance : testSet) {
            input.setValue(createInputTensor(instance));
            int classLabel = graph.predict().get(0);
            if (classLabel == labelMap.get(instance[instance.length - 1])) {
                count++;
            }
        }
        System.out.println("Accuracy: " + (count + 0.0) / testSet.size());
        assertEquals(0.9666666666666667, (count + 0.0) / testSet.size(), 0.01);
    }


    @Test
    public void test2() {
        ComputationalGraph graph = new ComputationalGraph();
        ComputationalNode a0 = new ComputationalNode(false, false, "+", null, null);
        ComputationalNode a1 = new ComputationalNode(true, false, "+", null, null);
        ComputationalNode a2 = graph.addEdge(a0, a1, false);
        graph.addEdge(a2, new Softmax(), false);
        List<Double> data = new ArrayList<>();
        Random rand = new Random(1);
        for (int i = 0; i < 3; i++) {
            data.add(rand.nextDouble() * 100);
        }
        a0.setValue(new Tensor(data, new int[]{1, 3}));
        a1.setValue(new Tensor(data, new int[]{1, 3}));
        graph.forwardCalculation();
        ArrayList<Integer> classList = new ArrayList<>();
        classList.add(1);
        graph.backpropagation(0.01, classList);
    }
}