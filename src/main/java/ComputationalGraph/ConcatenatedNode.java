package ComputationalGraph;

import java.io.Serializable;
import java.util.HashMap;

public class ConcatenatedNode extends ComputationalNode implements Serializable {

    private final HashMap<ComputationalNode, Integer> indexMap;
    private final int dimension;

    public ConcatenatedNode(int dimension) {
        super(false, false, null, null);
        this.indexMap = new HashMap<>();
        this.dimension = dimension;
    }

    public int getDimension() {
        return dimension;
    }

    public void clear() {
        this.indexMap.clear();
    }

    public int getIndex(ComputationalNode node) {
        return indexMap.get(node);
    }

    public void addNode(ComputationalNode node) {
        indexMap.put(node, indexMap.size());
    }
}
