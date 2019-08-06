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

package qa.qcri.nadeef.core.datamodel;

import com.google.common.base.Preconditions;

/**
 * Tuple Pair represents a pair of tuples.
 *
 */
public class TuplePair {
    private Tuple left;
    private Tuple right;

    /**
     * Constructor.
     * @param left left tuple.
     * @param right right tuple.
     */
    public TuplePair(Tuple left, Tuple right) {
        Preconditions.checkArgument(
            left != null && right != null, "Pair cannot have null values."
        );
        this.left = left;
        this.right = right;
    }

    /**
     * Factory method.
     * @param values values.
     * @return new TuplePair.
     */
    public static TuplePair of(Tuple[] values) {
        Preconditions.checkArgument(values != null && values.length == 2);
        return new TuplePair(values[0], values[1]);
    }

    //<editor-fold desc="Getters">

    /**
     * Get the left tuple.
     * @return the left tuple.
     */
    public Tuple getLeft() {
        return left;
    }

    /**
     * Gets the right tuple.
     * @return the right tuple.
     */
    public Tuple getRight() {
        return right;
    }

    //</editor-fold>
}
