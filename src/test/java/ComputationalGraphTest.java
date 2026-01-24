import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.*;
import ComputationalGraph.Optimizer.*;
import org.junit.Test;
import Math.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

import static org.junit.Assert.*;

public class ComputationalGraphTest {

    @Test
    public void testLinearPerceptronSingleInput(){
        LinearPerceptronSingleInput graph = new LinearPerceptronSingleInput();
        graph.train(new ArrayList<>(), new NeuralNetworkParameter(1, 100, new StochasticGradientDescent(0.1, 0.99), 0));
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
        NeuralNet graph = new NeuralNet();
        graph.train(trainList, new NeuralNetworkParameter(1, 100, new StochasticGradientDescent(0.1, 0.99), 0));
        ClassificationPerformance performance = graph.test(testList);
        System.out.println("Accuracy: " + performance.getAccuracy());
        assertEquals(1.0, performance.getAccuracy(), 0.01);
    }
}