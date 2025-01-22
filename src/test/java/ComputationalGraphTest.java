import ComputationalGraph.*;
import org.junit.Test;
import Math.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

import static org.junit.Assert.*;

public class ComputationalGraphTest {

    private Matrix createInputMatrix(String[] instance) {
        Matrix matrix = new Matrix(1, instance.length - 1);
        for (int i = 0; i < instance.length - 1; i++) {
            matrix.setValue(0, i, Double.parseDouble(instance[i]));
        }
        return matrix;
    }

    @Test
    public void test1() throws MatrixDimensionMismatch, MatrixRowColumnMismatch, FileNotFoundException {
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
        ComputationalNode input = new ComputationalNode(false, '*', true);
        Matrix m1 = new Matrix(5, 4, -0.01, 0.01, new Random(1));
        ComputationalNode w1 = new ComputationalNode(m1, '*');
        ComputationalNode a1 = graph.addEdge(input, w1, true);
        ComputationalNode a1Sigmoid = graph.addEdge(a1, FunctionType.SIGMOID, true);
        Matrix m2 = new Matrix(5, 20, -0.01, 0.01, new Random(1));
        ComputationalNode w2 = new ComputationalNode(m2, '*');
        ComputationalNode a2 = graph.addEdge(a1Sigmoid, w2, true);
        ComputationalNode a2Sigmoid = graph.addEdge(a2, FunctionType.SIGMOID, true);
        Matrix m3 = new Matrix(21, labelMap.size(), -0.01, 0.01, new Random(1));
        ComputationalNode w3 = new ComputationalNode(m3, '*');
        ComputationalNode a3 = graph.addEdge(a2Sigmoid, w3, false);
        graph.addEdge(a3, FunctionType.SOFTMAX, false);
        int epoch = 1000;
        double etaDecrease = 0.99;
        double learningRate = 0.1;
        ArrayList<Integer> classList = new ArrayList<>();
        for (int i = 0; i < epoch; i++) {
            Collections.shuffle(instances, new Random(1));
            for (String[] instance : instances) {
                input.setValue(createInputMatrix(instance));
                graph.forwardCalculation();
                classList.add(labelMap.get(instance[instance.length - 1]));
                graph.backpropagation(learningRate, classList);
                classList.clear();
            }
            learningRate *= etaDecrease;
        }
        int count = 0;
        for (String[] strings : testSet) {
            input.setValue(createInputMatrix(strings));
            int classLabel = graph.predict().get(0);
            if (classLabel == labelMap.get(strings[strings.length - 1])) {
                count++;
            }
        }
        assertEquals(0.9666666666666667, (count + 0.0) / testSet.size(), 0.001);
    }

    @Test
    public void test2() throws MatrixDimensionMismatch, MatrixRowColumnMismatch {
        ComputationalGraph graph = new ComputationalGraph();
        ComputationalNode a0 = new ComputationalNode(false, '+', false);
        ComputationalNode a1 = new ComputationalNode(true, '+', false);
        ComputationalNode a2 = graph.addEdge(a0, a1, false);
        ComputationalNode output = graph.addEdge(a2, FunctionType.SOFTMAX, false);
        a0.setValue(new Matrix(1, 3, 0, 100, new Random(1)));
        a1.setValue(new Matrix(1, 3, 0, 100, new Random(1)));
        graph.forwardCalculation();
        ArrayList<Integer> classList = new ArrayList<>();
        classList.add(1);
        graph.backpropagation(0.01, classList);
    }
}
