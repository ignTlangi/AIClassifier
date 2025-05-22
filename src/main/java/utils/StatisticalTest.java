package utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Comparator;

public class StatisticalTest {
    public static class WilcoxonResult {
        public final double statistic;
        public final double pValue;
        public final boolean isSignificant;
        public WilcoxonResult(double statistic, double pValue, boolean isSignificant) {
            this.statistic = statistic;
            this.pValue = pValue;
            this.isSignificant = isSignificant;
        }
    }

    public static WilcoxonResult wilcoxonSignedRankTest(double[] sample1, double[] sample2, double alpha) {
        if (sample1.length != sample2.length) {
            throw new IllegalArgumentException("Samples must have the same length");
        }
        int n = sample1.length;
        List<Double> differences = new ArrayList<>();
        List<Double> absoluteDifferences = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            double diff = sample1[i] - sample2[i];
            if (diff != 0) {
                differences.add(diff);
                absoluteDifferences.add(Math.abs(diff));
            }
        }
        Double[] absDiffs = absoluteDifferences.toArray(new Double[0]);
        Integer[] indices = new Integer[absDiffs.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, Comparator.comparingDouble(i -> absDiffs[i]));
        double[] ranks = new double[indices.length];
        int currentRank = 1;
        int i = 0;
        while (i < indices.length) {
            int j = i;
            while (j < indices.length && absDiffs[indices[j]].equals(absDiffs[indices[i]])) {
                j++;
            }
            double rank = (currentRank + j - 1) / 2.0;
            for (int k = i; k < j; k++) {
                ranks[indices[k]] = rank;
            }
            currentRank = j;
            i = j;
        }
        double W = 0;
        for (i = 0; i < differences.size(); i++) {
            if (differences.get(i) > 0) {
                W += ranks[i];
            } else {
                W -= ranks[i];
            }
        }
        double nEffective = differences.size();
        double mean = 0;
        double variance = (nEffective * (nEffective + 1) * (2 * nEffective + 1)) / 6.0;
        double stdDev = Math.sqrt(variance);
        double z = Math.abs(W) / stdDev;
        double pValue = 2 * (1 - normalCDF(z));
        return new WilcoxonResult(Math.abs(W), pValue, pValue < alpha);
    }
    private static double normalCDF(double x) {
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;
        int sign = 1;
        if (x < 0) {
            sign = -1;
        }
        x = Math.abs(x) / Math.sqrt(2.0);
        double t = 1.0 / (1.0 + p * x);
        double erf = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        return 0.5 * (1.0 + sign * erf);
    }
} 