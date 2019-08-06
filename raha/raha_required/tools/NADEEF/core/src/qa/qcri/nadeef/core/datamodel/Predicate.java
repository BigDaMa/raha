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
import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.Maps;
import qa.qcri.nadeef.tools.Logger;

import java.util.Collections;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Expression class describes a simple expression used in the scope.
 * An expression contains a left operator and a assigned value.
 */
public class Predicate {

    /**
     * Initialize BiMap for operation and corresponding SQL strings.
     */
    private static final BiMap<Operation, String> operationMap;
    static {
        Map<Operation, String> realMap = Maps.newHashMap();
        realMap.put(Operation.EQ, "=");
        realMap.put(Operation.GT, ">");
        realMap.put(Operation.LT, "<");
        realMap.put(Operation.NEQ, "!=");
        realMap.put(Operation.GTE, ">=");
        realMap.put(Operation.LTE, "<=");
        operationMap = ImmutableBiMap.copyOf(Collections.unmodifiableMap(realMap));
    }

    private static Logger tracer = Logger.getLogger(Predicate.class);

    private Operation operation;
    private Column left;
    private Column right;
    private Object value;
    private boolean isSingle;

    public static class PredicateBuilder {
        private Operation operation;
        private Column left;
        private Column right;
        private Object value;
        private boolean isSingle = false;

        public PredicateBuilder left(Column column) {
            Preconditions.checkNotNull(column);
            left = column;
            return this;
        }

        public PredicateBuilder isSingle() {
            isSingle = true;
            return this;
        }

		public PredicateBuilder isSingle(boolean isSingle) {
            this.isSingle = isSingle;
            return this;
        }


        public PredicateBuilder right(Column column) {
            right = column;
            return this;
        }

        public PredicateBuilder op(Operation operation) {
            this.operation = operation;
            return this;
        }

        public PredicateBuilder op(String operation) {
            this.operation = operationMap.inverse().get(operation);
            return this;
        }

        public PredicateBuilder constant(Object value) {
            this.value = value;
            return this;
        }

        public PredicateBuilder isNull() {
            this.value = null;
            return this;
        }

        public Predicate build() {
            Predicate exp = new Predicate();
            exp.left = left;
            exp.right = right;
            exp.value = value;
            exp.operation = operation;
            exp.isSingle = isSingle;
            return exp;
        }
    }

    private Predicate() {}

    /**
     * Returns the expression in SQL string.
     * @return SQL String.
     */
    // TODO: move this to SQLTable
    public String toSQLString() {
        StringBuilder builder = new StringBuilder(left.getFullColumnName());

        // http://en.wikipedia.org/wiki/SQL
        if (operation == Operation.NEQ && value == null) {
            builder.append(" is not ");
        } else {
            if (operation == Operation.NEQ)
                builder.append("<>");
            else
                builder.append(operationMap.get(operation));
        }

        if (value == null) {
            builder.append("null");
        } else if (!(value instanceof String)) {
            builder.append(value.toString());
        } else {
            builder.append("'");
            builder.append(value);
            builder.append("'");
        }
        return builder.toString();
    }

    public static Predicate createEq(Column leftColumn, Object constant) {
        return new PredicateBuilder()
            .left(leftColumn)
            .constant(constant)
            .op(Operation.EQ)
            .isSingle()
            .build();
    }

    public static Predicate valueOf(String value, String tableName) {
        final String patternRegx = "([^>=<!\\s]+)\\s*(>|>=|=|<=|<|!=)\\s*([^>=<!\\s]+)";
        final Pattern pattern = Pattern.compile(patternRegx);

        Predicate result;
        Matcher m = pattern.matcher(value);
        if (m.find()) {
            String leftHandSide = m.group(1);
            String operationStr = m.group(2);
            String rightHandSide = m.group(3);
            String leftColumnName = null;
            String rightColumnName = null;
            Pattern leftPattern = Pattern.compile("t1\\.(.*)");
            Matcher mLeft = leftPattern.matcher(leftHandSide);
            if (mLeft.find()) {
                leftColumnName = mLeft.group(1);
            } else {
                throw new IllegalArgumentException("illegal syntax" + leftHandSide);
            }

            Pattern singleRightPattern = Pattern.compile("t1\\.(.*)");
            Matcher mSingleRightMatcher = singleRightPattern.matcher(rightHandSide);
            Pattern doubleRightPattern = Pattern.compile("t2\\.(.*)");
            Matcher mDoubleRightMatcher = doubleRightPattern.matcher(rightHandSide);
            boolean isRightConstant = false;
            boolean isSingle = false;
            if (mSingleRightMatcher.find()) {
                rightColumnName = mSingleRightMatcher.group(1);
                isSingle = true;
                isRightConstant = false;
            } else if (mDoubleRightMatcher.find()) {
                rightColumnName = mDoubleRightMatcher.group(1);
                isSingle = false;
                isRightConstant = false;
            } else {
                rightColumnName = rightHandSide;
                isRightConstant = true;
                isSingle = true;
            }

            if (isRightConstant) {
                result =
                    new PredicateBuilder()
                        .left(new Column(tableName, leftColumnName))
                        .op(operationStr)
                        .constant(rightColumnName)
                        .isSingle()
                        .build();
            } else if (isSingle) {
                result =
                    new PredicateBuilder()
                        .left(new Column(tableName, leftColumnName))
                        .right(new Column(tableName, rightColumnName))
                        .isSingle()
                        .op(operationStr)
                        .build();
            } else {
                result =
                    new PredicateBuilder()
                        .left(new Column(tableName, leftColumnName))
                        .right(new Column(tableName, rightColumnName))
                        .op(operationStr)
                        .build();
            }
        } else {
            throw new IllegalArgumentException("illegal expression " + value);
        }
        return result;
    }

    /**
     * Returns <code>True</code> when the tuple matches the given predicate.
     * @param tuple tuple.
     * @return <code>True</code> when the tuple matches the given predicate.
     */
    @SuppressWarnings("unchecked")
    public boolean isValid(Tuple tuple) {
        Preconditions.checkArgument(isSingle());
        Object leftValue = tuple.get(left);
        Comparable leftComparable = (Comparable)leftValue;
        int compareResult;
        if (!isRightConstant()) {
            Object rightValue = tuple.get(right);

            if (leftValue == null || rightValue == null) {
                tracer.info("Tuple attribute contains NULL value.");
            }

            if (leftValue == null && rightValue == null) {
                compareResult = 0;
            } else if (leftValue == null || rightValue == null) {
                compareResult = 1;
            } else {
                compareResult = leftComparable.compareTo(rightValue);
            }
        } else {
            if (leftValue == null) {
                tracer.info("Tuple attribute contains NULL value.");
                compareResult = 1;
            } else {
                DataType type = tuple.getSchema().getType(left);
                // TODO: should move during parsing?
                switch (type) {
                    case INTEGER:
                        compareResult =
                            leftComparable.compareTo(Integer.parseInt((String)value));
                        break;
                    case FLOAT:
                        // special case for postgres when reading real is always DOUBLE.
                        if (leftValue instanceof Double) {
                            compareResult =
                                leftComparable.compareTo(Double.parseDouble((String) value));
                        } else {
                            compareResult =
                                leftComparable.compareTo(Float.parseFloat((String)value));
                        }
                        break;
                    case DOUBLE:
                        compareResult = leftComparable.compareTo(Double.parseDouble((String)value));
                        break;
                    default:
                        compareResult = leftComparable.compareTo(value);
                }
            }
        }
        return validResult(compareResult);
    }

    /**
     * Returns <code>True</code> when the tuple matches the given predicate.
     * @param tupleLeft left tuple.
     * @param tupleRight right tuple.
     * @return <code>True</code> when the tuple matches the given predicate.
     */
    @SuppressWarnings("unchecked")
    public boolean isValid(Tuple tupleLeft, Tuple tupleRight) {
        Object leftValue = tupleLeft.get(left);
        Comparable leftComparable = (Comparable)leftValue;
        int compareResult;
        if (isRightConstant()) {
            if (leftValue == null) {
                tracer.info("Tuple attribute contains NULL value.");
                compareResult = 1;
            } else {
                DataType type = tupleLeft.getSchema().getType(left);
                // TODO: should move during parsing?
                switch (type) {
                    case INTEGER:
                        compareResult =
                            leftComparable.compareTo(Integer.parseInt((String)value));
                        break;
                    case FLOAT:
                        compareResult =
                            leftComparable.compareTo(Float.parseFloat((String)value));
                        break;
                    case DOUBLE:
                        compareResult =
                            leftComparable.compareTo(Double.parseDouble((String)value));
                        break;
                    default:
                        compareResult = leftComparable.compareTo(value);
                }
            }
        } else {
            Object rightValue = tupleRight.get(right);
            if (leftValue == null || rightValue == null) {
                tracer.info("Tuple attribute contains NULL value.");
            }

            if (leftValue == null && rightValue == null) {
                compareResult = 0;
            } else if (leftValue == null || rightValue == null) {
                compareResult = 1;
            } else {
                compareResult = ((Comparable)leftValue).compareTo(rightValue);
            }
        }
        return validResult(compareResult);
    }

    // TODO: to improve
    @SuppressWarnings("unchecked")
    public boolean isValid(Cell left, Cell right) {
        Object leftValue = left.getValue();
        Comparable leftComparable = (Comparable)leftValue;
        int compareResult = leftComparable.compareTo(right.getValue());
        return validResult(compareResult);
    }

    /**
     * Gets the {@link Operation}.
     * @return operation.
     */
    public Operation getOperation() {
        return operation;
    }

    /**
     * Gets the left operator column.
     * @return left operator column.
     */
    public Column getLeft() {
        return left;
    }

    public Object getValue() {
        return value;
    }

    /**
     * Gets the right operator column.
     * @return right operator column.
     */
    public Column getRight() {
        return right;
    }

    public boolean isRightConstant() {
        return value != null;
    }

    public boolean isSingle() {
        return isSingle;
    }

    private boolean validResult(int compareResult) {
        boolean result;
        switch (operation){
            case EQ:
                result = compareResult == 0;
                break;
            case GT:
                result = compareResult > 0;
                break;
            case GTE:
                result = compareResult >= 0;
                break;
            case NEQ:
                result = compareResult != 0;
                break;
            case LT:
                result = compareResult < 0;
                break;
            case LTE:
                result = compareResult <= 0;
                break;
            default:
                throw new UnsupportedOperationException("unsupported operation: " + operation);
        }
        return result;
    }
}
