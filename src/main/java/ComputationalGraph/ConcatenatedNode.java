package ComputationalGraph;

import java.io.Serializable;
import java.util.HashMap;

public class ConcatenatedNode extends ComputationalNode implements Serializable {

    private final HashMap<ComputationalNode, Integer> indexMap;

    public ConcatenatedNode() {
        super(false, false, null, null);
        this.indexMap = new HashMap<>();
    }

    public int getIndex(ComputationalNode node) {
        return indexMap.get(node);
    }

    public void addNode(ComputationalNode node) {
        indexMap.put(node, indexMap.size());
    }
}
