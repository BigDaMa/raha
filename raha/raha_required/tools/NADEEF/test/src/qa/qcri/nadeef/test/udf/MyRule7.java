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
package qa.qcri.nadeef.test.udf;

import qa.qcri.nadeef.core.datamodel.*;

import java.util.*;

public class MyRule7 extends SingleTupleRule {
    private List<Predicate> predicates;

    public MyRule7() {
        predicates = new ArrayList<>();
    }

    @Override
    public Collection<Violation> detect(Tuple tuple) {
        String tableName = getTableNames().get(0);
        predicates.add(Predicate.valueOf("t1.B!=b1", tableName));
        boolean isValid = true;
        List<Violation> result = new ArrayList<>();
        Set<Cell> infectedCells = new HashSet<>();
        for (Predicate predicate : predicates){
            if (!predicate.isValid(tuple)) {
                isValid = false;
                break;
            }

            Cell leftCell = tuple.getCell(predicate.getLeft());
            infectedCells.add(leftCell);
            if (!predicate.isRightConstant()){
                Cell rightCell = tuple.getCell(predicate.getRight());
                infectedCells.add(rightCell);
            }
        }
        // all the predicates are valid, then the DC is violated
        if (isValid) {
            Violation violation = new Violation(getRuleName());
            for (Cell cell : infectedCells){
                violation.addCell(cell);
            }
            //violation.addCell(cell);
            result.add(violation);
        }
        return result;
    }

    @Override
    public Collection<Fix> repair(Violation violation) {
        List<Cell> cells = new ArrayList<>(violation.getCells());
        List<Fix> result = new ArrayList<>();
        HashMap<Column, Cell> columnMap = new HashMap<>();

        Fix.Builder builder = new Fix.Builder(violation);
        for (Cell cell : cells)
            columnMap.put(cell.getColumn(), cell);

        for (Predicate predicate : predicates) {
            Column leftColumn = predicate.getLeft();
            if (predicate.isRightConstant()) {
                result.add(builder
                    .left(columnMap.get(leftColumn))
                    .op(repairOperation(predicate.getOperation()))
                    .right(predicate.getValue())
                    .build()
                );
            } else {
                Column rightColumn = predicate.getRight();
                result.add(builder
                    .left(columnMap.get(leftColumn))
                    .op(repairOperation(predicate.getOperation()))
                    .right(columnMap.get(rightColumn))
                    .build()
                );
            }
        }
        return result;
    }

    private Operation repairOperation(Operation op) {
        Operation result = null;
        switch (op) {
            case EQ:
                result = Operation.NEQ;
                break;
            case GT:
                result = Operation.LTE;
                break;
            case GTE:
                result = Operation.LT;
                break;
            case LT:
                result = Operation.GTE;
                break;
            case LTE:
                result = Operation.GT;
                break;
            default:
                assert true : "unknown operations";
        }
        return result;
    }
}