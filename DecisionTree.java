import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Stack;

public class DecisionTree extends SupervisedLearner {

	private final static double validationSetSize = 0.25;
	private final static double purityThreshold = 1;
	private final static boolean prune = false;
	private Random random;
	private double accuracy;
	private Map<Integer, DecisionTree> children;
	private int featureIndex = -1;
	private Matrix featuresMatrix;
	private Matrix labelsMatrix;
	private List<double[]> features;
	private List<double[]> labels;
	private int[] skipArray;
	private int category = -1;

	public DecisionTree(Random random) {
		this.random = random;
	}

	public void train(Matrix features, Matrix labels) throws Exception {
		Matrix validationFeatures = null;
		Matrix trainingFeatures = null;
		Matrix validationLabels = null;
		Matrix trainingLabels = null;
		if (prune) {
			features.shuffle(random, labels);
			int validationSetNumber = (int) (validationSetSize * features
					.rows());
			validationFeatures = new Matrix(features, 0, 0,
					validationSetNumber, features.cols());
			trainingFeatures = new Matrix(features, validationSetNumber, 0,
					features.rows() - validationSetNumber, features.cols());
			validationLabels = new Matrix(labels, 0, 0, validationSetNumber,
					labels.cols());
			trainingLabels = new Matrix(labels, validationSetNumber, 0,
					labels.rows() - validationSetNumber, labels.cols());

			featuresMatrix = trainingFeatures;
			labelsMatrix = trainingLabels;
			this.features = trainingFeatures.m_data;
			this.labels = trainingLabels.m_data;
		} else {
			featuresMatrix = features;
			labelsMatrix = labels;
			this.features = features.m_data;
			this.labels = labels.m_data;
		}
		children = new HashMap<>();
		skipArray = new int[] {};
		visit();
		if (prune) {
			// System.out.println(this);
			// System.out.println(getDepth());
			accuracy = getAccuracy(validationFeatures, validationLabels);
			// System.out.println("Accuracy: " + accuracy);
			prune(accuracy, validationFeatures, validationLabels);
			System.out.println(getNodeCount());
			System.out.println(this);
			// System.out.println("Accuracy: " + accuracy);
		}
	}

	public double getAccuracy(Matrix features, Matrix labels) throws Exception {
		double accuracy = 0;
		double[] predictions = new double[labels.cols()];

		for (int i = 0; i < features.rows(); i++) {
			predict(features.row(i), predictions);
			if (predictions[0] == labels.get(i, 0))
				accuracy++;
		}
		accuracy /= features.rows();

		return accuracy;
	}

	public void prune(double accuracy, Matrix validationFeatures,
			Matrix validationLabels) throws Exception {
		Stack<DecisionTree> stack = new Stack<>();
		stack.push(this);
		while (!stack.isEmpty()) {
			DecisionTree node = stack.pop();
			if (node.category >= 0)
				continue;
			Map<Integer, DecisionTree> childrenTmp = node.children;
			node.children = new HashMap<>();
			double[] histo = new double[node.labelsMatrix.valueCount(0)];
			for (int i = 0; i < node.labels.size(); i++) {
				histo[(int) node.labels.get(i)[0]]++;
			}
			double max = 0;
			int maxIndex = -1;
			for (int i = 0; i < histo.length; i++) {
				if (histo[i] > max) {
					max = histo[i];
					maxIndex = i;
				}
			}
			node.category = maxIndex;
			double newAccuracy = getAccuracy(validationFeatures,
					validationLabels);
			if (newAccuracy >= accuracy)
				accuracy = newAccuracy;
			else {
				node.category = -1;
				node.children = childrenTmp;
			}

			for (int i : node.children.keySet())
				stack.push(node.children.get(i));
		}
	}

	public void visit() {

		double[] histo = new double[labelsMatrix.valueCount(0)];
		for (int i = 0; i < labels.size(); i++) {
			histo[(int) labels.get(i)[0]]++;
		}
		double max = 0;
		int maxIndex = -1;
		for (int i = 0; i < histo.length; i++) {
			if (histo[i] > max) {
				max = histo[i];
				maxIndex = i;
			}
		}
		if (max / labels.size() >= purityThreshold
				|| features.get(0).length == skipArray.length)
			category = maxIndex;
		else {
			chooseFeature();
			for (int i = 0; i < featuresMatrix.valueCount(featureIndex); i++) {
				addChild(i);
			}

			for (Integer i : children.keySet()) {
				children.get(i).visit();
			}
		}
	}

	public void predict(double[] features, double[] labels) throws Exception {
		double[] probabilities = getCategoryProbabilities(features, labels);
		double max = 0;
		double maxIndex = -1;
		for (int i = 0; i < probabilities.length; i++) {
			if (probabilities[i] > max) {
				max = probabilities[i];
				maxIndex = i;
			}
		}

		labels[0] = maxIndex;
	}

	public double[] getCategoryProbabilities(double[] features, double[] labels) {
		double[] probabilities = new double[labelsMatrix.valueCount(0)];
		if (category >= 0) {
			probabilities[category] = 1;
		} else {
			DecisionTree child = children.get((int) features[featureIndex]);
			if (child != null) {
				probabilities = child
						.getCategoryProbabilities(features, labels);
			} else {

				int missingValues = 0;
				for (int i = 0; i < this.features.size(); i++) {
					if (this.features.get(i)[featureIndex] == Matrix.MISSING)
						missingValues++;
				}

				double[] childProbabilities = null;
				for (Integer i : children.keySet()) {
					childProbabilities = children.get(i)
							.getCategoryProbabilities(features, labels);
					for (int j = 0; j < childProbabilities.length; j++)
						probabilities[j] += childProbabilities[j]
								* children.get(i).features.size()
								/ (this.features.size() - missingValues);
				}
			}
		}

		return probabilities;
	}

	public void addChild(int value) {
		DecisionTree child = new DecisionTree(random);
		child.features = new ArrayList<>();
		child.labels = new ArrayList<>();
		child.featuresMatrix = featuresMatrix;
		child.labelsMatrix = labelsMatrix;
		child.children = new HashMap<>();
		child.skipArray = new int[skipArray.length + 1];
		for (int i = 0; i < skipArray.length; i++)
			child.skipArray[i] = skipArray[i];
		child.skipArray[child.skipArray.length - 1] = featureIndex;

		double[] histo = new double[featuresMatrix.valueCount(featureIndex)];
		for (int i = 0; i < features.size(); i++) {
			if (features.get(i)[featureIndex] < Matrix.MISSING)
				histo[(int) features.get(i)[featureIndex]]++;
		}
		double max = 0;
		int maxIndex = -1;
		for (int i = 0; i < histo.length; i++) {
			if (histo[i] > max) {
				max = histo[i];
				maxIndex = i;
			}
		}

		for (int i = 0; i < features.size(); i++) {
			if (features.get(i)[featureIndex] == value) {
				child.features.add(features.get(i));
				child.labels.add(labels.get(i));
			} else if (features.get(i)[featureIndex] == Matrix.MISSING
					&& value == maxIndex) {
				child.features.add(features.get(i));
				child.labels.add(labels.get(i));
			}
		}

		if (child.features.size() > 0)
			children.put(value, child);
	}

	public void chooseFeature() {
		double min_info = Double.MAX_VALUE;
		int min_idx = -1;
		for (int k = 0; k < features.get(0).length; k++) {
			boolean skip = false;
			for (int j = 0; j < skipArray.length; j++) {
				if (skipArray[j] == k) {
					skip = true;
				}
			}
			if (skip) {
				continue;
			}
			double feature_info = info(k);
			if (feature_info < min_info) {
				min_idx = k;
				min_info = feature_info;
			}
		}
		featureIndex = min_idx;
	}

	private double info(int feature_idx) {
		double[] histo = new double[labelsMatrix.valueCount(0)];
		double score = 0;
		for (int k = 0; k < featuresMatrix.valueCount(feature_idx); k++) {
			double total = 0;
			for (int j = 0; j < histo.length; j++)
				histo[j] = 0;
			for (int j = 0; j < features.size(); j++) {
				if (features.get(j)[feature_idx] == k) {
					int label = (int) labels.get(j)[0];
					histo[label]++;
					total++;
				}
			}
			if (total != 0) {
				for (int j = 0; j < histo.length; j++) {
					histo[j] /= total;
				}
			}

			double entropy = entropy(histo);
			score += (total / features.size()) * entropy;
		}
		return score;
	}

	private double entropy(double[] histo) {
		double logSum = 0;
		for (int i = 0; i < histo.length; i++) {
			if (histo[i] != 0)
				logSum -= histo[i] * Math.log(histo[i]) / Math.log(2);
		}

		return logSum;
	}

	public int getNodeCount() {
		int count = 1;
		for (int i : children.keySet())
			count += children.get(i).getNodeCount();
		return count;
	}

	public int getDepth() {
		return getDepth(0);
	}

	public int getDepth(int level) {
		int maxDepth = level;
		for (int i : children.keySet()) {
			int childDepth = children.get(i).getDepth(level + 1);
			if (childDepth > maxDepth)
				maxDepth = childDepth;
		}

		return maxDepth;
	}

	public String toString() {
		return toString(0);
	}

	public String toString(int level) {
		StringBuilder sb = new StringBuilder();
		String tabs = "";
		for (int i = 0; i < level; i++)
			tabs = tabs + "\t";
		if (featureIndex >= 0)
			sb.append("\n" + tabs + "featureIndex=" + featureIndex + " ("
					+ featuresMatrix.attrName(featureIndex) + ")");
		sb.append("\n" + tabs + "features="
				+ Arrays.deepToString(features.toArray()));
		sb.append("\n" + tabs + "labels="
				+ Arrays.deepToString(labels.toArray()));
		sb.append("\n" + tabs + "skipArray=" + Arrays.toString(skipArray));
		if (category >= 0)
			sb.append("\n" + tabs + "category=" + category + " ("
					+ labelsMatrix.attrValue(0, category) + ")");

		for (int i : children.keySet()) {
			sb.append("\n" + tabs + "{\n");
			sb.append(tabs + "\tChild " + i + " ("
					+ featuresMatrix.attrValue(featureIndex, i) + "):");
			sb.append(children.get(i).toString(level + 1));
			sb.append("\n" + tabs + "}");
		}

		return sb.toString();
	}
}
