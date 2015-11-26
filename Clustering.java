import java.io.FileNotFoundException;

public class Clustering {

	private static final boolean normalize = false;
	private static final String fileName = "laborWithID.arff";
	private static final int k = 5;
	private static Matrix data;

	public static void main(String[] args) throws FileNotFoundException,
			Exception {
		data = new Matrix();
		data.loadArff(fileName);
		data = new Matrix(data, 0, 1, data.rows(), data.cols() - 2);
		if (normalize) {
			System.out.println("Using normalized data\n");
			data.normalize();
		}

		double[][] centroids = new double[k][];
		for (int i = 0; i < k; i++) {
			centroids[i] = new double[data.cols()];
			for (int j = 0; j < data.cols(); j++)
				centroids[i][j] = data.get(i, j);
		}

		System.out.println("Computing Centroids:");
		for (int i = 0; i < k; i++) {
			System.out.print("Centroid " + i + " = ");
			for (int j = 0; j < data.cols(); j++) {
				if (centroids[i][j] == Matrix.MISSING)
					System.out.print("?");
				else if (data.valueCount(j) == 0)
					System.out.printf("%.3f", centroids[i][j]);
				else
					System.out.print(data.attrValue(j, (int) centroids[i][j]));
				if (j != data.cols() - 1)
					System.out.print(", ");
			}
			System.out.println();
		}

		System.out.println("Making Assignments");
		double sse = 0;

		int[] clusters = new int[data.rows()];
		for (int i = 0; i < data.rows(); i++) {
			double minSquaredDistance = Double.MAX_VALUE;
			int closestK = -1;
			for (int j = 0; j < k; j++) {
				double squaredDistance = squaredDistance(data.row(i),
						centroids[j]);
				if (squaredDistance < minSquaredDistance) {
					minSquaredDistance = squaredDistance;
					closestK = j;
				}
			}
			clusters[i] = closestK;
			sse += minSquaredDistance;
			if (i % 10 == 0) {
				if (i > 0)
					System.out.println();
				System.out.print("\t");
			}
			System.out.print(i + "=" + closestK + " ");
		}
		System.out.printf("\nSSE: %.3f", sse);
	}

	private static double squaredDistance(double[] a, double[] b) {
		double squaredDistance = 0;
		for (int i = 0; i < a.length; i++) {
			if (a[i] == Matrix.MISSING || b[i] == Matrix.MISSING)
				squaredDistance++;
			else {
				if (data.valueCount(i) == 0)
					squaredDistance += (a[i] - b[i]) * (a[i] - b[i]);
				else if (a[i] != b[i])
					squaredDistance++;
			}
		}
		return squaredDistance;
	}
}
