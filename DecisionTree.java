import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DecisionTree extends SupervisedLearner {

	private DecisionTree parent;
	private Map<Integer, DecisionTree> children;
	private int featureIndex = -1;
	private List<double[]> features;
	private List<double[]> labels;
	private int[] featureValueCounts;
	private int[] labelValueCounts;
	private int[] skipArray;
	private int category = -1;

	public void train(Matrix features, Matrix labels) throws Exception {
		this.features = features.m_data;
		this.labels = labels.m_data;
		featureValueCounts = new int[features.cols()];
		for (int i = 0; i < featureValueCounts.length; i++)
			featureValueCounts[i] = features.valueCount(i);
		labelValueCounts = new int[labels.cols()];
		for (int i = 0; i < labelValueCounts.length; i++)
			labelValueCounts[i] = labels.valueCount(i);
		children = new HashMap<>();
		skipArray = new int[] {};
		visit();
		// System.out.println(this);
	}

	public void visit() {

		boolean isPure = true;
		for (int i = 0; i < labels.size(); i++) {
			if (labels.get(i)[0] != labels.get(0)[0])
				isPure = false;
		}
		if (isPure)
			category = (int) labels.get(0)[0];
		else {
			chooseFeature();
			for (int i = 0; i < featureValueCounts[featureIndex]; i++) {
				addChild(i);
			}

			if (children.isEmpty()) {
				double[] histo = new double[labelValueCounts[0]];
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
				category = maxIndex;
			}

			for (Integer i : children.keySet()) {
				children.get(i).visit();
			}
		}
	}

	public void predict(double[] features, double[] labels) throws Exception {

	}

	public void addChild(int value) {
		DecisionTree child = new DecisionTree();
		child.features = new ArrayList<>();
		child.labels = new ArrayList<>();
		child.featureValueCounts = featureValueCounts;
		child.labelValueCounts = labelValueCounts;
		child.children = new HashMap<>();
		child.skipArray = new int[skipArray.length + 1];
		for (int i = 0; i < skipArray.length; i++)
			child.skipArray[i] = skipArray[i];
		child.skipArray[child.skipArray.length - 1] = featureIndex;
		for (int i = 0; i < features.size(); i++) {
			if (features.get(i)[featureIndex] == value) {
				child.features.add(features.get(i));
				child.labels.add(labels.get(i));
			}
		}
		child.parent = this;

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
		double[] histo = new double[labelValueCounts[0]];
		double score = 0;
		for (int k = 0; k < featureValueCounts[feature_idx]; k++) {
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

	public String toString() {
		return toString(0);
	}

	public String toString(int level) {
		StringBuilder sb = new StringBuilder();
		String tabs = "";
		for (int i = 0; i < level; i++)
			tabs = tabs + "\t";
		sb.append(tabs + "featureIndex=" + featureIndex);
		sb.append("\n" + tabs + "features="
				+ Arrays.deepToString(features.toArray()));
		sb.append("\n" + tabs + "labels="
				+ Arrays.deepToString(labels.toArray()));
		sb.append("\n" + tabs + "featureValueCounts="
				+ Arrays.toString(featureValueCounts));
		sb.append("\n" + tabs + "labelValueCounts="
				+ Arrays.toString(labelValueCounts));
		sb.append("\n" + tabs + "skipArray=" + Arrays.toString(skipArray));
		sb.append("\n" + tabs + "category=" + category);

		for (int i : children.keySet()) {
			sb.append("\n" + tabs + "{\n");
			sb.append(tabs + "\tChild (value " + i + "):\n");
			sb.append(children.get(i).toString(level + 1));
			sb.append("\n" + tabs + "}");
		}

		return sb.toString();
	}
}
