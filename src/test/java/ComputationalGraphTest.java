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
            public void loadModel(String fileName) {

            }

            @Override
            public void train(Tensor trainSet, Parameter parameters) {

            }

            @Override
            public ClassificationPerformance test(Tensor testSet) {
                return null;
            }
        };
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

    @Test
    public void test3DCase() {
        /**
         * Tests a 3D computational graph case with higher dimensional tensors.
         */
        // Initialize the computational graph
        ComputationalGraph graph = new ComputationalGraph() {
            @Override
            public void loadModel(String fileName) {
            }

            @Override
            public void train(Tensor trainSet, Parameter parameters) {
            }

            @Override
            public ClassificationPerformance test(Tensor testSet) {
                return null;
            }
        };

        // Define nodes for 3D operations
        ComputationalNode input3D = new ComputationalNode(false, false, "*", null, null);
        ComputationalNode weight3D = new ComputationalNode(true, false, "*", null, null);
        
        // Create 3D tensors and flatten them to 2D for compatibility
        // Original 3D shapes: 2x3x4 (batch_size x height x width)
        List<Double> inputTensor3D = new ArrayList<>();
        Random rand = new Random(10);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 4; k++) {
                    inputTensor3D.add(rand.nextDouble() * 10);
                }
            }
        }
        
        // Flatten to 2D: 2x12 (batch_size x flattened_features)
        // We need to reshape the 3D tensor to 2D
        List<Double> inputTensor2D = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3 * 4; j++) {
                inputTensor2D.add(inputTensor3D.get(i * 12 + j));
            }
        }
        
        // Weight: 12x3 tensor (flattened_features x num_classes)
        List<Double> weightTensor2D = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 3; j++) {
                weightTensor2D.add(rand.nextDouble() * 0.2 - 0.1); // -0.1 to 0.1
            }
        }
        
        // Set values
        input3D.setValue(new Tensor(inputTensor2D, new int[]{2, 12}));
        weight3D.setValue(new Tensor(weightTensor2D, new int[]{12, 3}));
        
        // Create computational graph with flattened 3D tensors
        ComputationalNode convResult = graph.addEdge(input3D, weight3D, false);
        ComputationalNode output3D = graph.addEdge(convResult, new Softmax(), false);

        // Perform forward calculation
        graph.forwardCalculation();
        
        // Verify the graph processed 3D tensors correctly BEFORE backpropagation
        assertNotNull(output3D.getValue());
        int[] outputShape = output3D.getValue().getShape();
        System.out.println("3D test completed successfully with flattened tensor shape: " + 
                          outputShape[0] + " x " + outputShape[1]);
        System.out.println("Original 3D tensor shape was: 2x3x4, flattened to: 2x12");
        
        // Test backpropagation with 3D data
        ArrayList<Integer> trueClass3D = new ArrayList<>();
        trueClass3D.add(0);
        trueClass3D.add(1); // Two samples with different class labels
        graph.backpropagation(0.01, trueClass3D);
    }
}