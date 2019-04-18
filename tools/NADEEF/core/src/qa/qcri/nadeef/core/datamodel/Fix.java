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

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;

/**
 * Fix represents a suggestion of changing a cell to a value. This is the result from repairing
 * violations.
 */
public class Fix {

    //<editor-fold desc="Private fields">
    private Operation operation;
    private Cell left;
    private Cell right;
    private String rightValue;
    private int vid;

    //</editor-fold>

    /**
     * Builder class.
     */
    public static class Builder {
        private Cell left;
        private Cell right;
        private String rightValue;
        private int vid;
        private Optional<Operation> operation;

        public Builder() {
            operation = Optional.absent();
        }

        public Builder(Violation violation) {
            this();
            this.vid = violation.getVid();
        }

        public Builder vid(int vid) {
            this.vid = vid;
            return this;
        }

        public Builder left(Cell left) {
            this.left = Preconditions.checkNotNull(left);
            return this;
        }

        public Builder right(Cell right) {
            this.right = Preconditions.checkNotNull(right);
            this.rightValue = null;
            return this;
        }

        public Builder right(String right) {
            this.rightValue = Preconditions.checkNotNull(right);
            this.right = null;
            return this;
        }

        public Builder right(Object right) {
            Preconditions.checkNotNull(right);
            this.rightValue = right.toString();
            this.right = null;
            return this;
        }

        public Builder op(Operation operation) {
            this.operation = Optional.of(operation);
            return this;
        }

        public Fix build() {
            Fix fix;
            if (operation.isPresent()) {
                if (right != null) {
                    fix = new Fix(vid, left, right, operation.get());
                } else {
                    fix = new Fix(vid, left, rightValue, operation.get());
                }
            } else {
                if (rightValue != null) {
                    fix = new Fix(vid, left, rightValue, Operation.EQ);
                } else {
                    fix = new Fix(vid, left, right, Operation.EQ);
                }
            }
            return fix;
        }
    }

    //<editor-fold desc="Constructor">
    protected Fix(int vid, Cell left, Operation operation) {
        this.left = Preconditions.checkNotNull(left);
        this.operation = Preconditions.checkNotNull(operation);
        this.vid = vid;
    }

    /**
     * Constructor.
     * @param left left operator.
     * @param right right operator.
     * @param operation operation value.
     */
    protected Fix(int vid, Cell left, Cell right, Operation operation) {
        this(vid, left, operation);
        this.right = Preconditions.checkNotNull(right);
    }

    /**
     * Constructor.
     * @param left left cell.
     * @param right right constant value in string.
     * @param operation operation value.
     */
    protected Fix(int vid, Cell left, String right, Operation operation) {
        this(vid, left, operation);
        this.rightValue = Preconditions.checkNotNull(right);
    }

    //</editor-fold>

    //<editor-fold desc="Getters">

    /**
     * Gets the operation.
     * @return operation.
     */
    public Operation getOperation() {
        return operation;
    }

    /**
     * Gets the left cell.
     * @return left cell.
     */
    public Cell getLeft() {
        return left;
    }

    /**
     * Gets the right value as a cell.
     * @return right value as a cell.
     */
    public Cell getRight() {
        return right;
    }

    /**
     * Gets the right value in string.
     * @return value in string.
     */
    public String getRightValue() {
        return rightValue;
    }

    /**
     * Gets of vid.
     * @return vid.
     */
    public int getVid() {
        return vid;
    }

    /**
     * Returns <code>True</code> when the right cell is a constant value.
     * @return <code>True</code> when the right cell is a constant value.
     */
    public boolean isRightConstant() {
        return this.right == null;
    }
    //</editor-fold>
}
