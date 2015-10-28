import java.util.Map;

public class DecisionTreeNode {

	private DecisionTreeNode parent;
	private Map<Integer, DecisionTreeNode> children;
	private int featureIndex;
	private Matrix features;
	private Matrix labels;

	public DecisionTreeNode(Matrix features, Matrix labels) {
		this.features = features;
		this.labels = labels;
	}

	public void addChild(DecisionTreeNode child, int featureIndex) {
		child.featureIndex = featureIndex;
		child.parent = this;
		children.put(child.featureIndex, child);
	}
}
