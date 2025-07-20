package ComputationalGraph;

import Math.Tensor;
import java.util.ArrayList;
import java.util.List;

public class Softmax implements Function {
    /**
     * Implements the Softmax activation function.
     */

    @Override
    public Tensor calculate(Tensor tensor) {
        int[] shape = tensor.getShape();
        if (shape.length < 1) {
            throw new IllegalArgumentException("Softmax requires at least 1D tensor.");
        }

        int lastDim = shape[shape.length - 1];
        int[] outerShape = new int[shape.length - 1];
        System.arraycopy(shape, 0, outerShape, 0, shape.length - 1);
        int outerDim = tensor.computeNumElements(outerShape);
        
        List<Double> resultData = new ArrayList<>();

        for (int i = 0; i < outerDim; i++) {
            // Build base index prefix: e.g. (batch, time) part
            int[] baseIdx = tensor.unflattenIndex(i, tensor.computeStrides(outerShape));
            
            // Get the row values for this outer index
            List<Double> row = new ArrayList<>();
            double maxVal = Double.NEGATIVE_INFINITY;
            
            for (int j = 0; j < lastDim; j++) {
                int[] fullIdx = new int[baseIdx.length + 1];
                System.arraycopy(baseIdx, 0, fullIdx, 0, baseIdx.length);
                fullIdx[fullIdx.length - 1] = j;
                
                double val = tensor.getValue(fullIdx);
                row.add(val);
                maxVal = Math.max(maxVal, val);
            }

            // Compute exp with stability
            List<Double> expRow = new ArrayList<>();
            double sumExp = 0.0;
            for (double val : row) {
                double expVal = Math.exp(val - maxVal);
                expRow.add(expVal);
                sumExp += expVal;
            }

            // Normalize
            for (double expVal : expRow) {
                resultData.add(expVal / sumExp);
            }
        }

        return new Tensor(resultData, shape);
    }

    @Override
    public Tensor derivative(Tensor tensor) {
        // Compute the softmax output first (like in the Python version)
        Tensor softmaxOutput = calculate(tensor);

        int[] shape = softmaxOutput.getShape();
        int rows = shape[0];
        int cols = shape[1];

        List<List<List<Double>>> initialData = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<List<Double>> rowList = new ArrayList<>();
            for (int j = 0; j < cols; j++) {
                List<Double> colList = new ArrayList<>();
                for (int k = 0; k < cols; k++) {
                    colList.add(0.0);
                }
                rowList.add(colList);
            }
            initialData.add(rowList);
        }

        Tensor result = new Tensor(initialData, new int[]{rows, cols, cols});

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double s_i = softmaxOutput.getValue(new int[]{i, j});
                for (int k = 0; k < cols; k++) {
                    double s_k = softmaxOutput.getValue(new int[]{i, k});
                    double value = (j == k) ? s_i * (1 - s_k) : -s_i * s_k;
                    result.set(new int[]{i, j, k}, value);
                }
            }
        }

        return result;
    }
}
