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

package qa.qcri.nadeef.lab.hc;

import com.google.common.collect.Maps;
import qa.qcri.nadeef.core.datamodel.Cell;
import qa.qcri.nadeef.core.datamodel.Fix;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

class FixExtensions {
    public static boolean isValidFix(HashSet<Fix> repairContext, List<Fix> fixes) {
        HashMap<Cell, String> modifiedCell = Maps.newHashMap();
        for (Fix fix : fixes) {
            modifiedCell.put(fix.getLeft(), fix.getRightValue());
        }

        for (Fix fix : repairContext) {
            Cell leftCell = fix.getLeft();
            double leftValue;
            if (modifiedCell.containsKey(leftCell)) {
                String obj = modifiedCell.get(leftCell);
                leftValue = parseValue(obj);
            } else {
                leftValue = getValue(leftCell.getValue());
            }

            double rightValue;
            if (!fix.isRightConstant()) {
                Cell rightCell = fix.getRight();
                if (modifiedCell.containsKey(rightCell)) {
                    String obj = modifiedCell.get(rightCell);
                    rightValue = parseValue(obj);
                } else {
                    rightValue = getValue(rightCell.getValue());
                }
            } else {
                String obj = fix.getRightValue();
                rightValue = parseValue(obj);
            }

            boolean result = false;
            switch (fix.getOperation()) {
                case EQ:
                    result = leftValue == rightValue;
                    break;
                case LT:
                    result = leftValue < rightValue;
                    break;
                case LTE:
                    result = leftValue <= rightValue;
                    break;
                case GT:
                    result = leftValue > rightValue;
                    break;
                case GTE:
                    result = leftValue >= rightValue;
                    break;
                case NEQ:
                    result = leftValue != rightValue;
                    break;
                default:
                    assert true;
            }

            // inverse the result to break constraints
            if (!result)
                return false;
        }
        return true;
    }

    private static double getValue(Object v) {
        if (v instanceof Double)
            return (Double)v;
        else if (v instanceof Float) {
            return (Float)v;
        } else if (v instanceof Integer) {
            return (Integer)v;
        }

        return parseValue((String)v);
    }

    private static double parseValue(String str) {
        double v = 0.0;
        try {
            v = (double)Integer.parseInt(str);
        } catch (Exception ex) {}

        try {
            v = Double.parseDouble(str);
        } catch (Exception ex) {}

        return v;
    }

}
