class DecisionTree extends SupervisedLearner {

	private DecisionTreeNode tree;

	public void train(Matrix features, Matrix labels) throws Exception {
		tree = new DecisionTreeNode(features, labels);
		chooseFeature(features, labels, new int[] {});
	}

	private int chooseFeature(Matrix features, Matrix labels, int[] skipArray) {
		double min_info = Double.MAX_VALUE;
		int min_idx = -1;
		for (int k = 0; k < features.cols(); k++) {
			boolean skip = false;
			for (int j = 0; j < skipArray.length; j++) {
				if (skipArray[j] == k) {
					skip = true;
				}
			}
			if (skip) {
				continue;
			}
			double feature_info = info(features, labels, k);
			if (feature_info < min_info) {
				min_idx = k;
				min_info = feature_info;
			}
		}
		return min_idx;
	}

	private double info(Matrix features, Matrix labels, int feature_idx) {
		// Map<Integer, Double> histo = new TreeMap<>();
		double[] histo = new double[labels.valueCount(0)];
		double score = 0;
		for (int k = 0; k < features.valueCount(feature_idx); k++) {
			double total = 0;
			for (int j = 0; j < histo.length; j++)
				histo[j] = 0;
			for (int j = 0; j < features.rows(); j++) {
				if (features.get(j, feature_idx) == k) {
					int label = (int) labels.get(j, 0);
					histo[label]++;
					total++;
				}
			}
			for (int j = 0; j < histo.length; j++) {
				histo[j] /= total;
			}

			double entropy = entropy(histo);
			score += (total / features.rows()) * entropy;
		}
		return score;
	}

	private double entropy(double[] histo) {
		double sum = 0;
		for (int i = 0; i < histo.length; i++)
			sum += histo[i];
		double logSum = 0;
		for (int i = 0; i < histo.length; i++) {
			if (histo[i] != 0)
				logSum -= histo[i] * Math.log(histo[i]) / Math.log(2);
		}

		return logSum;
	}

	public void predict(double[] features, double[] labels) throws Exception {

	}
}
