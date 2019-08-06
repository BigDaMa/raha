/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

package qa.qcri.nadeef.tools;

import uk.ac.shef.wit.simmetrics.similaritymetrics.*;

/**
 * Metrics provides methods to do similarity test.
 */
public final class Metrics {
    public static double getEuclideanDistance(String a, String b) {
        return template(a, b, new EuclideanDistance());
    }

    public static double getQGramsDistance(String a, String b) {
        return template(a, b, new QGramsDistance());
    }

    public static double getSoundex(String a, String b) {
        return template(a, b, new Soundex());
    }

    public static double getLevenshtein(String a, String b) {
        return template(a, b, new Levenshtein());
    }

    public static double getSmithWaterman(String a, String b) {
        return template(a, b, new SmithWaterman());
    }

    public static double getSmithWatermanGotoWindowedAffine(String a, String b) {
        return template(a, b, new SmithWatermanGotohWindowedAffine());
    }

    public static double getMongeElkan(String a, String b) {
        return template(a, b, new MongeElkan());
    }

    public static double getEqual(String a, String b) {
        if (a == null || b == null) {
            return 0.0;
        }
        return a.equals(b) ? 1.0f : 0.0f;
    }


    private static double template(String a, String b, AbstractStringMetric metric) {
        if (a == null || b == null) {
            return 0.0;
        }
        return metric.getSimilarity(a, b);
    }
}
