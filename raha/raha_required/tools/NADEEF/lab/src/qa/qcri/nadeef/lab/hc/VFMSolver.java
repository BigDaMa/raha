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

import com.google.common.collect.*;
import qa.qcri.nadeef.core.datamodel.Cell;
import qa.qcri.nadeef.core.datamodel.Fix;
import qa.qcri.nadeef.core.datamodel.Operation;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * Value Frequency Solver. It is a solver to deal with EQ / NEQ only
 * {@link qa.qcri.nadeef.core.datamodel.Fix}. The implementation is by counting the
 * frequency of values and set the new value with the highest count.
 */
public class VFMSolver extends SolverBase {
    @Override
    public List<Fix> solve(HashSet<Fix> repairContext, HashSet<Cell> changed) {
        HashSet<Cell> cells = Sets.newHashSet();
        HashMap<Object, Integer> countMap = Maps.newHashMap();

        // executing VFM
        // we start with counting both the EQ and NEQ.
        // TODO: we don't know how to deal with NEQ.
        // When there are only NEQ, we need to assign an variable class.
        for (Fix fix : repairContext) {
            if (fix.isRightConstant()) {
                createOrAdd(countMap, fix.getRightValue());
                cells.add(fix.getLeft());
            } else {
                cells.add(fix.getLeft());
                createOrAdd(countMap, fix.getLeft().getValue());
                cells.add(fix.getRight());
                createOrAdd(countMap, fix.getRight().getValue());
            }
        }

        // count all the cells in the context.
        for (Cell cell : cells)
            createOrAdd(countMap, cell.getValue());

        // pick the highest count first
        int maxCount = 0;
        for (Integer count : countMap.values())
            maxCount = Math.max(count, maxCount);

        Object target = null;
        if (cells.size() == 1) {
            // A very special case when there is only one cell involved,
            // in that case we pick the highest value which is not equal to original value.
            Object original = cells.iterator().next().getValue();
            for (Map.Entry<Object, Integer> entry : countMap.entrySet()) {
                if (entry.getValue().equals(maxCount) && !entry.getKey().equals(original))
                    target = entry.getKey();
            }
        } else {
            // In normal cases we just pick the highest occurrence one.
            for (Map.Entry<Object, Integer> entry : countMap.entrySet())
                if (entry.getValue().equals(maxCount))
                    target = entry.getKey();
        }

        List<Fix> result = Lists.newArrayList();
        if (target != null) {
            Fix.Builder builder = new Fix.Builder();
            for (Cell cell : cells) {
                if (cell.getValue().equals(target))
                    continue;
                result.add(
                    builder
                        .left(cell)
                        .right(target)
                        .op(Operation.EQ)
                        .build()
                );
            }
        }
        return result;
    }

    private void createOrAdd(HashMap<Object, Integer> map, Object key) {
        if (map.containsKey(key)) {
            int count = map.get(key);
            map.put(key, count + 1);
        } else {
            map.put(key, 1);
        }
    }
}
